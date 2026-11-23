"""Shared experiment orchestration and optimizer loop for all training methods."""
from itertools import islice
import json
import math
import random
import time

import torch

from train.checkpointing import CheckpointManager, prepare_output_directory
from train.preparation import IGNORE_INDEX, make_loaders, prepare_data
from train.reporting import (
    MemoryTracker, build_report, checkpoint_sizes, environment_info,
    parameter_counts, print_report, source_hashes, write_history, write_json,
)
from train.runtime import ModelContext, autocast, pick_device, resolve_precision, synchronize
from train import validation
from train.validation import timed_validation


def build_optimizer(name, model, lr, weight_decay):
    if name == "adafactor":
        from transformers.optimization import Adafactor
        return Adafactor(
            model.parameters(), lr=lr, weight_decay=weight_decay,
            scale_parameter=False, relative_step=False, warmup_init=False,
        )
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def training_plan(batches_per_epoch, grad_accum, epochs, max_steps=None):
    """Plan complete/partial epochs without borrowing batches from another pass."""
    steps_per_epoch = math.ceil(batches_per_epoch / grad_accum)
    if max_steps is not None:
        return [batches_per_epoch] * math.ceil(max_steps / steps_per_epoch), max_steps
    batches = math.ceil(batches_per_epoch * epochs)
    full_epochs, remainder = divmod(batches, batches_per_epoch)
    limits = [batches_per_epoch] * full_epochs
    if remainder:
        limits.append(remainder)
    return limits, sum(math.ceil(limit / grad_accum) for limit in limits)


def normalize_gradients(model, response_tokens):
    """Turn accumulated summed token losses into a mean over response tokens."""
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad.div_(response_tokens)


def run_training(model, train_loader, val_loader, args, device, autocast_dtype, output_dir, checkpoints=None,
                 generation_evaluator=None):
    """Train, persist step history, and measure training separately from validation."""
    from transformers import get_scheduler
    from train.validation import is_better, validation_due

    epoch_limits, total_steps = training_plan(
        len(train_loader), args.grad_accum, args.epochs, args.max_steps
    )
    print(f"Optimizer steps: {total_steps} (configured effective batch {args.batch_size * args.grad_accum})")
    optimizer = build_optimizer(args.optim, model, args.lr, args.weight_decay)
    warmup_steps = math.ceil(total_steps * args.warmup_ratio)
    scheduler = get_scheduler(
        args.lr_scheduler, optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    resumed = checkpoints.restore(optimizer, scheduler) if checkpoints is not None else {}
    val_loss_before = resumed.get("val_loss_before")
    validation_before_seconds = resumed.get("validation_before_seconds", 0.0)
    validation_history = list(resumed.get("validation_history", []))
    if not resumed and len(val_loader.dataset):
        val_loss_before, validation_before_seconds = timed_validation(model, val_loader, device, autocast_dtype)
        print(f"Validation loss before training: {val_loss_before:.4f}")
        initial = {"step": 0, "epochs_completed": 0.0, "val_loss": val_loss_before}
        if generation_evaluator is not None:
            synchronize(device)
            initial_started = time.perf_counter()
            initial.update(generation_evaluator(0))
            synchronize(device)
            validation_before_seconds += time.perf_counter() - initial_started
        validation_history.append(initial)
    write_history(output_dir, validation_history)

    memory = MemoryTracker(device)
    history = list(resumed.get("history", []))
    step = resumed.get("step", 0)
    micro = resumed.get("micro", 0)
    tokens_seen = resumed.get("tokens_seen", 0)
    response_tokens_seen = resumed.get("response_tokens_seen", 0)
    examples_seen = resumed.get("examples_seen", 0)
    summed_loss = resumed.get("summed_loss", 0.0)
    periodic_validation_seconds = checkpoint_seconds = 0.0
    previous_train_seconds = resumed.get("train_seconds", 0.0)
    checkpoint_seconds_total = resumed.get("checkpoint_seconds", 0.0)
    optimizer.zero_grad(set_to_none=True)
    synchronize(device)
    started = time.perf_counter()
    with (output_dir / "training_history.jsonl").open("w", encoding="utf-8") as history_stream:
        for record in history:
            history_stream.write(json.dumps(record, allow_nan=False) + "\n")
        for epoch_index, batches_this_epoch in enumerate(epoch_limits):
            if step >= total_steps:
                break
            if resumed and epoch_index < resumed["epoch_index"]:
                continue
            skip_batches = 0
            if resumed and epoch_index == resumed["epoch_index"]:
                train_loader.generator.set_state(resumed["epoch_generator_state"])
                skip_batches = resumed["batch_index"]
            epoch_generator_state = train_loader.generator.get_state().clone() if train_loader.generator is not None else None
            iterator = iter(train_loader)
            for _ in islice(iterator, skip_batches):
                pass
            if resumed and epoch_index == resumed["epoch_index"]:
                checkpoints.restore_random_state()
            window_loss = 0.0
            window_tokens = window_examples = micro_in_window = 0
            window_padded_tokens = window_sequence_length = 0
            previous_update_batch = skip_batches
            synchronize(device)
            window_started = time.perf_counter()
            for batch_index, batch in enumerate(islice(iterator, batches_this_epoch - skip_batches), start=skip_batches + 1):
                batch = {key: value.to(device) for key, value in batch.items()}
                n_tokens = int((batch["labels"][:, 1:] != IGNORE_INDEX).sum())
                with autocast(device, autocast_dtype):
                    loss = model(**batch).loss
                loss_value = float(loss.detach())
                if not math.isfinite(loss_value):
                    raise FloatingPointError("Training produced a non-finite loss")
                memory.sample()
                (loss * n_tokens).backward()
                memory.sample()
                window_loss += loss_value * n_tokens
                window_tokens += n_tokens
                window_examples += len(batch["input_ids"])
                summed_loss += loss_value * n_tokens
                tokens_seen += int(batch["attention_mask"].sum())
                response_tokens_seen += n_tokens
                examples_seen += len(batch["input_ids"])
                micro += 1
                micro_in_window += 1
                window_padded_tokens += batch["input_ids"].numel()
                window_sequence_length = max(window_sequence_length, batch["input_ids"].shape[1])
                if micro_in_window < args.grad_accum and batch_index < batches_this_epoch:
                    continue

                normalize_gradients(model, window_tokens)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.max_grad_norm, error_if_nonfinite=True
                )
                learning_rate = optimizer.param_groups[0]["lr"]
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                memory.sample()
                step += 1
                synchronize(device)
                record = {
                    "step": step,
                    "epochs_completed": examples_seen / len(train_loader.dataset),
                    "train_loss": window_loss / window_tokens,
                    "examples": window_examples,
                    "response_tokens": window_tokens,
                    "grad_norm": float(grad_norm),
                    "lr": learning_rate,
                    "step_seconds": time.perf_counter() - window_started,
                    "padded_tokens": window_padded_tokens,
                    "max_sequence_length": window_sequence_length,
                }
                if device.type == "cuda":
                    stats = torch.cuda.memory_stats(device)
                    free, _ = torch.cuda.mem_get_info(device)
                    record.update({
                        "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                        "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                        "gpu_allocated_bytes_after_step": torch.cuda.memory_allocated(device),
                        "gpu_reserved_bytes_after_step": torch.cuda.memory_reserved(device),
                        "gpu_free_bytes_after_step": free,
                        "cuda_allocation_retries_total": stats.get("num_alloc_retries", 0),
                        "cuda_ooms_total": stats.get("num_ooms", 0),
                    })
                final = step >= total_steps
                epoch_end = batch_index == len(train_loader)
                generation_due = generation_evaluator is not None and (
                    validation_due(previous_update_batch, batch_index, len(train_loader), args.generation_checks_per_epoch)
                    or final
                )
                loss_due = len(val_loader.dataset) and (
                    (step % args.eval_every == 0 if args.eval_every else validation_due(
                        previous_update_batch, batch_index, len(train_loader), args.val_checks_per_epoch))
                    or generation_due or final
                )
                candidate = None
                if loss_due:
                    memory.sample()
                    record["val_loss"], seconds = timed_validation(model, val_loader, device, autocast_dtype)
                    periodic_validation_seconds += seconds
                    memory.restart_interval()
                    validation_record = {"step": step, "epochs_completed": record["epochs_completed"],
                                         "train_loss": record["train_loss"], "val_loss": record["val_loss"]}
                    if generation_due:
                        synchronize(device)
                        generation_started = time.perf_counter()
                        metrics = generation_evaluator(step)
                        synchronize(device)
                        periodic_validation_seconds += time.perf_counter() - generation_started
                        validation_record.update(metrics)
                        record.update({key: value for key, value in metrics.items()
                                       if key not in ("predictions_file", "examples")})
                        record["validation_examples"] = metrics["examples"]
                        candidate = {"step": step, "epochs_completed": record["epochs_completed"],
                                     "complete_call_accuracy": metrics["complete_call_accuracy"], "val_loss": record["val_loss"]}
                        best = checkpoints.best if checkpoints is not None else None
                        if not is_better(candidate, best):
                            candidate = None
                        memory.restart_interval()
                    validation_history.append(validation_record)
                    write_history(output_dir, validation_history)
                history.append(record)
                history_stream.write(json.dumps(record, allow_nan=False) + "\n")
                history_stream.flush()
                if step % args.log_every == 0 or step == total_steps:
                    elapsed = previous_train_seconds + time.perf_counter() - started - periodic_validation_seconds - checkpoint_seconds
                    extra = f" val_loss {record['val_loss']:.4f}" if "val_loss" in record else ""
                    print(f"[step {step}/{total_steps}] loss {record['train_loss']:.4f} "
                          f"grad_norm {record['grad_norm']:.2f} lr {learning_rate:.2e}{extra} "
                          f"({elapsed:.0f}s training; {record['step_seconds']:.2f}s/step; max_length {window_sequence_length})", flush=True)
                window_loss = 0.0
                window_tokens = window_examples = micro_in_window = 0
                if checkpoints is not None and (generation_due or checkpoints.should_save(step, epoch_end, final)):
                    synchronize(device)
                    memory.sample()
                    progress = {
                        "step": step, "micro": micro, "tokens_seen": tokens_seen,
                        "response_tokens_seen": response_tokens_seen, "examples_seen": examples_seen,
                        "summed_loss": summed_loss, "history": list(history),
                        "epoch_index": epoch_index, "batch_index": batch_index,
                        "epoch_generator_state": epoch_generator_state,
                        "validation_history": list(validation_history),
                        "val_loss_before": val_loss_before, "validation_before_seconds": validation_before_seconds,
                        "periodic_validation_seconds": resumed.get("periodic_validation_seconds", 0.0) + periodic_validation_seconds,
                        "train_seconds": previous_train_seconds + time.perf_counter() - started - periodic_validation_seconds - checkpoint_seconds,
                        "checkpoint_seconds": checkpoint_seconds_total + checkpoint_seconds,
                        "peak_memory_bytes": max(resumed.get("peak_memory_bytes") or 0, memory.peak_bytes() or 0),
                        "peak_reserved_memory_bytes": max(resumed.get("peak_reserved_memory_bytes") or 0, memory.reserved_bytes() or 0),
                    }
                    checkpoint_seconds += checkpoints.save(model, optimizer, scheduler, progress, best_candidate=candidate)
                memory.restart_interval()
                previous_update_batch = batch_index
                window_padded_tokens = window_sequence_length = 0
                synchronize(device)
                window_started = time.perf_counter()
                if step >= total_steps:
                    break
            if step >= total_steps:
                break
    synchronize(device)
    loop_seconds = time.perf_counter() - started
    train_seconds = previous_train_seconds + loop_seconds - periodic_validation_seconds - checkpoint_seconds
    memory.sample()
    peak_memory = memory.peak_bytes()
    peak_reserved = memory.reserved_bytes()
    if resumed:
        if peak_memory is not None:
            peak_memory = max(peak_memory, resumed.get("peak_memory_bytes") or 0)
        if peak_reserved is not None:
            peak_reserved = max(peak_reserved, resumed.get("peak_reserved_memory_bytes") or 0)

    val_loss_after, validation_after_seconds = None, 0.0
    if len(val_loader.dataset):
        val_loss_after = validation_history[-1]["val_loss"]
    return {
        "optimizer_steps": step,
        "micro_batches": micro,
        "examples_seen": examples_seen,
        "epochs_completed": examples_seen / len(train_loader.dataset),
        "tokens_seen": tokens_seen,
        "response_tokens_seen": response_tokens_seen,
        "train_loss": summed_loss / response_tokens_seen,
        "train_loss_last_step": history[-1]["train_loss"],
        "train_seconds": train_seconds,
        "training_loop_seconds": train_seconds + periodic_validation_seconds + resumed.get("periodic_validation_seconds", 0.0) + checkpoint_seconds_total + checkpoint_seconds,
        "periodic_checkpoint_seconds": checkpoint_seconds_total + checkpoint_seconds,
        "seconds_per_step": train_seconds / step,
        "examples_per_second": examples_seen / max(train_seconds, 1e-9),
        "tokens_per_second": tokens_seen / max(train_seconds, 1e-9),
        "response_tokens_per_second": response_tokens_seen / max(train_seconds, 1e-9),
        "peak_memory_bytes": peak_memory,
        "peak_reserved_memory_bytes": peak_reserved,
        "peak_memory_kind": memory.label(),
        "memory_measurement_scope": {
            "cuda": "training_intervals", "mps": "sampled_training_intervals", "cpu": "process_lifetime",
        }[device.type],
        "val_loss_before": val_loss_before,
        "val_loss_after": val_loss_after,
        "validation_seconds": {
            "before_training": validation_before_seconds,
            "during_training": periodic_validation_seconds + resumed.get("periodic_validation_seconds", 0.0),
            "after_training": validation_after_seconds,
        },
        "history": history,
        "validation_history": validation_history,
        "best_checkpoint": checkpoints.best if checkpoints is not None else None,
        "selected_model_path": "best" if checkpoints is not None and checkpoints.best else None,
        "last_model_path": ".",
        "lr_scheduler": args.lr_scheduler,
        "warmup_steps": warmup_steps,
        "optimizer_settings": {key: list(value) if isinstance(value, tuple) else value
                               for key, value in optimizer.defaults.items()
                               if key in ("lr", "betas", "eps", "weight_decay")},
        "early_stopping": False,
    }


def run_experiment(args, build_model, *, method, splits=None, describe_model=None, method_source=None):
    """Execute one method using a single shared data, validation and reporting protocol."""
    started = time.perf_counter()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    (weight_dtype, autocast_dtype), precision = resolve_precision(args.precision, device)
    checkpoints = CheckpointManager(args) if not args.no_save else None
    state = checkpoints.state if checkpoints is not None else None
    output_dir = prepare_output_directory(args, state)
    environment = environment_info(device)

    preprocessing_started = time.perf_counter()
    revision = (state["base_revision"] or args.revision) if state else args.revision
    prepared = prepare_data(args, splits=splits, revision=revision, code_hashes=source_hashes(method_source))
    if checkpoints is not None:
        checkpoints.prepare(prepared.manifest, args, prepared.tokenizer)
        checkpoints.record_environment(environment)
    write_json(output_dir / "data_splits.json", prepared.manifest)
    preprocessing_seconds = time.perf_counter() - preprocessing_started
    train_loader, val_loader = make_loaders(prepared, args)
    if args.generation_checks_per_epoch and prepared.generation_examples:
        from eval.metrics import validate_examples
        validate_examples(prepared.generation_examples)

    context = ModelContext(device, weight_dtype, autocast_dtype or weight_dtype,
                           state["base_revision"] if state else None)
    synchronize(device)
    load_started = time.perf_counter()
    model = build_model(args, context)
    synchronize(device)
    model_load_seconds = time.perf_counter() - load_started
    model.config.use_cache = False
    model.train()
    trainable_params, total_params = parameter_counts(model)
    print(f"Device: {device}, precision: {precision}, optimizer: {args.optim}")
    print(f"Parameters: {trainable_params:,} trainable / {total_params:,} total "
          f"({100 * trainable_params / total_params:.1f}%)")
    if checkpoints is not None:
        checkpoints.bind_model(model)

    evaluator = None
    if args.generation_checks_per_epoch and prepared.generation_examples:
        def evaluator(step):
            return validation.generation_metrics(model, prepared.tokenizer, prepared.generation_examples, args,
                                                 device, autocast_dtype, output_dir, step)
    training_metrics = run_training(model, train_loader, val_loader, args, device, autocast_dtype,
                                    output_dir, checkpoints, generation_evaluator=evaluator)
    sizes = {"model_weights_bytes": None, "checkpoint_bytes": None}
    save_started = time.perf_counter()
    if not args.no_save:
        model.config.use_cache = True
        model.save_pretrained(output_dir, safe_serialization=True)
        tokenizer_files = prepared.tokenizer.save_pretrained(output_dir)
        sizes = checkpoint_sizes(output_dir, tokenizer_files)
    checkpoint_save_seconds = time.perf_counter() - save_started
    report = build_report(args=args, method=method, model=model, device=device, precision=precision,
                          weight_dtype=weight_dtype, autocast_dtype=autocast_dtype, environment=environment,
                          prepared=prepared, training_metrics=training_metrics, sizes=sizes,
                          model_load_seconds=model_load_seconds, checkpoint_save_seconds=checkpoint_save_seconds,
                          preprocessing_seconds=preprocessing_seconds, run_wall_seconds=time.perf_counter()-started,
                          checkpoints=checkpoints)
    if describe_model is not None:
        report.update(describe_model(model, args, report, context))
    write_json(output_dir / "train_metrics.json", report)
    print_report(report, args)
    return report
