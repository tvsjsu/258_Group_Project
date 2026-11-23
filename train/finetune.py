"""Full fine-tuning of the base model on the xLAM train split.

Every model weight is trainable (no frozen layers, no adapters). Training uses
exactly the prompt that eval/eval.py sends at inference time, and only the
response tokens (the JSON array of calls plus the end-of-turn token) count
toward the loss.

Initial test run (a few optimizer steps on a small slice of the data):

    uv run python -m train.finetune --max-steps 20 --limit-train 400 --limit-val 50

The output directory contains the model/tokenizer, train_metrics.json,
training_history.jsonl, and data_splits.json. Function-calling accuracy is
measured separately by eval/eval.py on the saved checkpoint.
"""

import argparse
import hashlib
from importlib.metadata import PackageNotFoundError, version
from itertools import islice
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from model.prompt import build_messages

try:
    import resource
except ImportError:  # Windows does not provide resource.
    resource = None

IGNORE_INDEX = -100
DATA_SEED = 42
UNSEEN_FRACTION = 0.2


# ---------------------------------------------------------------- tokenization

def response_text(tokenizer, query, tools, answers):
    """Return (prompt, target) strings.

    The prompt is rendered exactly as at inference (generation prompt added).
    The target is whatever the chat template places after it for an assistant
    turn, so training matches the model's native format, and it always ends
    with the EOS token that stops generation.
    """
    messages = build_messages(query, tools)
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    answer = json.dumps(answers, ensure_ascii=False)
    full = tokenizer.apply_chat_template(
        messages + [{"role": "assistant", "content": answer}], tokenize=False
    )
    if full.startswith(prompt):
        target = full[len(prompt):].rstrip()
    else:
        target = answer
    eos = tokenizer.eos_token
    if eos and not target.endswith(eos):
        target += eos
    return prompt, target


def encode_example(tokenizer, example):
    prompt, target = response_text(
        tokenizer, example["query"], example["tools"], example["answers"]
    )
    # Same call as FunctionCallingModel.generate_batch: the template already
    # contains any BOS token, so do not add special tokens again.
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
    return {
        "input_ids": prompt_ids + target_ids,
        "labels": [IGNORE_INDEX] * len(prompt_ids) + target_ids,
    }


def update_fingerprint(digest, example):
    content = json.dumps(example, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    digest.update(content.encode("utf-8") + b"\n")


def prepare_split(tokenizer, examples, max_length):
    """Encode one split and record exactly which examples survive filtering."""
    features, retained_ids, dropped_ids = [], [], []
    digest = hashlib.sha256()
    sequence_tokens = response_tokens = 0
    for example in examples:
        feature = encode_example(tokenizer, example)
        if len(feature["input_ids"]) > max_length:
            dropped_ids.append(example["id"])
            continue
        supervised = sum(label != IGNORE_INDEX for label in feature["labels"][1:])
        if not supervised:
            raise ValueError(f"Example {example['id']!r} has no supervised response tokens")
        features.append(feature)
        retained_ids.append(example["id"])
        update_fingerprint(digest, example)
        sequence_tokens += len(feature["input_ids"])
        response_tokens += supervised
    return features, {
        "selected_count": len(examples),
        "retained_count": len(features),
        "dropped_count": len(dropped_ids),
        "retained_ids": retained_ids,
        "dropped_ids": dropped_ids,
        "retained_sha256": digest.hexdigest(),
        "sequence_tokens": sequence_tokens,
        "response_tokens": response_tokens,
    }


def encode_split(tokenizer, examples, max_length):
    """Keep the original (features, dropped_count) interface for other callers."""
    features, details = prepare_split(tokenizer, examples, max_length)
    return features, details["dropped_count"]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def describe_splits(splits):
    """Fingerprint split contents and order, preserving original dataset IDs."""
    result = {}
    for name, examples in splits.items():
        digest = hashlib.sha256()
        ids = []
        for example in examples:
            ids.append(example["id"])
            update_fingerprint(digest, example)
        result[name] = {"count": len(ids), "ids": ids, "sha256": digest.hexdigest()}
    return result


def environment_info(device):
    """Record hardware and software so measurements can be interpreted later."""
    packages = {}
    for name in ("torch", "transformers", "datasets", "peft", "bitsandbytes", "accelerate"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "packages": packages,
        "device": str(device),
        "device_type": device.type,
        "cpu_threads": torch.get_num_threads(),
        "cuda_version": torch.version.cuda,
    }
    try:
        info["system_physical_memory_bytes"] = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, OSError, ValueError):
        info["system_physical_memory_bytes"] = None
    if sys.platform == "darwin":
        try:
            chip = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=2, check=False,
            )
            if chip.returncode == 0:
                info["processor"] = chip.stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            pass
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        info["gpu_name"] = properties.name
        info["gpu_total_memory_bytes"] = properties.total_memory
    return info


def source_hashes():
    root = Path(__file__).resolve().parents[1]
    return {
        name: hashlib.sha256((root / name).read_text(encoding="utf-8").encode("utf-8")).hexdigest()
        for name in ("data/data.py", "model/prompt.py", "model/model.py", "eval/metrics.py", "eval/eval.py",
                     "train/finetune.py", "train/validation.py", "train/checkpointing.py")
    }


def checkpoint_sizes(output_dir, tokenizer_files):
    """Count model/tokenizer artifacts without counting logs or split records."""
    output_dir = Path(output_dir)
    weights, indexes = [], []
    for filename in ("adapter_model.safetensors", "model.safetensors", "pytorch_model.bin"):
        path = output_dir / filename
        index = output_dir / (filename + ".index.json")
        if path.is_file():
            weights = [path]
            break
        if index.is_file():
            mapping = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
            weights = [output_dir / name for name in sorted(set(mapping.values()))]
            indexes = [index]
            break
    files = set(weights + indexes + [Path(name) for name in tokenizer_files])
    files.update(output_dir / name for name in ("config.json", "generation_config.json", "adapter_config.json"))
    return {
        "model_weights_bytes": sum(path.stat().st_size for path in weights),
        "checkpoint_bytes": sum(path.stat().st_size for path in files if path.is_file()),
    }


def make_collator(pad_id):
    def collate(batch):
        width = max(len(item["input_ids"]) for item in batch)
        input_ids, labels, attention = [], [], []
        for item in batch:
            pad = width - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [pad_id] * pad)
            labels.append(item["labels"] + [IGNORE_INDEX] * pad)
            attention.append([1] * len(item["input_ids"]) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(attention),
        }
    return collate


# ---------------------------------------------------------------- device utils

def pick_device(requested):
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_precision(precision, device):
    """Return (weight dtype, autocast dtype or None)."""
    if precision == "auto":
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            precision = "bf16-mixed"
        else:
            precision = "fp32"
    return {
        "fp32": (torch.float32, None),
        "bf16-mixed": (torch.float32, torch.bfloat16),
        "bf16": (torch.bfloat16, None),
    }[precision], precision


class MemoryTracker:
    """Track training intervals; keep platform-specific measurement labels."""

    def __init__(self, device):
        self.device = device
        self.peak = 0
        self.reserved_peak = 0
        self.restart_interval()

    def restart_interval(self):
        """Exclude a validation interval without losing earlier training peaks."""
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        self.sample()

    def sample(self):
        if self.device.type == "cuda":
            self.peak = max(self.peak, torch.cuda.max_memory_allocated(self.device))
            self.reserved_peak = max(self.reserved_peak, torch.cuda.max_memory_reserved(self.device))
        elif self.device.type == "mps":
            self.peak = max(self.peak, torch.mps.driver_allocated_memory())
        elif resource is not None:
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            self.peak = max(self.peak, rss if sys.platform == "darwin" else rss * 1024)

    def peak_bytes(self):
        return None if self.device.type == "cpu" and resource is None else self.peak

    def reserved_bytes(self):
        return self.reserved_peak if self.device.type == "cuda" else None

    def label(self):
        return {
            "cuda": "Peak GPU memory allocated during training",
            "mps": "Sampled peak MPS driver memory during training",
            "cpu": "Peak process RSS (process lifetime)",
        }[self.device.type]


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


# ---------------------------------------------------------------- training

@torch.no_grad()
def evaluate_loss(model, loader, device, autocast_dtype):
    """Token-weighted mean loss over the response tokens."""
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        tokens = int((batch["labels"][:, 1:] != IGNORE_INDEX).sum())
        with autocast(device, autocast_dtype):
            loss = model(**batch).loss
        total_loss += float(loss) * tokens
        total_tokens += tokens
    model.train()
    return total_loss / max(total_tokens, 1)


def autocast(device, dtype):
    if dtype is None:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def build_optimizer(name, model, lr, weight_decay):
    if name == "adafactor":
        from transformers.optimization import Adafactor
        return Adafactor(
            model.parameters(), lr=lr, weight_decay=weight_decay,
            scale_parameter=False, relative_step=False, warmup_init=False,
        )
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def example_limit(value):
    if value == "all":
        return None
    try:
        limit = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use a nonnegative integer or 'all'") from exc
    if limit < 0:
        raise argparse.ArgumentTypeError("Use a nonnegative integer or 'all'")
    return limit


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


def timed_validation(model, loader, device, autocast_dtype):
    synchronize(device)
    started = time.perf_counter()
    loss = evaluate_loss(model, loader, device, autocast_dtype)
    synchronize(device)
    return loss, time.perf_counter() - started


def parse_args(argv=None, configure_parser=None):
    from model.model import MODEL_NAME
    p = argparse.ArgumentParser(description="Full fine-tuning on the xLAM train split.")
    p.add_argument("--model", default=MODEL_NAME, help="Hugging Face id or local path.")
    p.add_argument("--output-dir", default="checkpoints/full")
    p.add_argument("--limit-train", type=example_limit, default=None, help="Use the first N train examples, or 'all' (default).")
    p.add_argument("--limit-val", type=example_limit, default=None, help="Validation examples for loss AND generation: N, 'all' (default), or 0 to skip.")
    p.add_argument("--epochs", type=float, default=4.0)
    p.add_argument("--max-steps", type=int, default=None, help="Stop after N optimizer steps (overrides --epochs).")
    p.add_argument("--batch-size", type=int, default=1, help="Micro-batch size per forward pass.")
    p.add_argument("--grad-accum", type=int, default=16, help="Micro-batches per optimizer step.")
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--max-length", type=int, default=2048, help="Drop examples longer than this many tokens.")
    p.add_argument("--optim", default="adamw", choices=["adamw", "adafactor"])
    p.add_argument("--precision", default="auto", choices=["auto", "fp32", "bf16-mixed", "bf16"],
                   help="auto = bf16-mixed on CUDA GPUs that support it, else fp32.")
    p.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--device", default=None)
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--eval-every", type=int, default=0, help="Override epoch-based loss checks with every N optimizer steps (0 = automatic epoch cadence).")
    p.add_argument("--val-checks-per-epoch", type=int, default=10)
    p.add_argument("--generation-checks-per-epoch", type=int, default=2, help="0 disables generated validation and best-model selection.")
    p.add_argument("--eval-batch-size", type=int, default=2, help="Independent teacher-forced validation micro-batch.")
    p.add_argument("--generation-batch-size", type=int, default=1)
    p.add_argument("--generation-max-new-tokens", type=int, default=512)
    p.add_argument("--lr-scheduler", choices=["cosine", "linear"], default="cosine")
    p.add_argument("--revision", default=None, help="Optional base model/tokenizer commit ID.")
    p.add_argument("--save-steps", type=int, default=100, help="Resumable checkpoint interval; also save at epoch ends, generation checks and final step.")
    p.add_argument("--resume-from-checkpoint", default=None)
    p.add_argument("--reference-data", default=None)
    p.add_argument("--no-save", action="store_true", help="Skip writing the model checkpoint.")
    p.add_argument("--seed", type=int, default=42)

    # Sample command line for a quick sanity check (20 optimizer steps on 400 train examples, 50 validation examples):
    # uv run python -m train.finetune \
    #   --limit-train 400 \
    #   --limit-val 50 \
    #   --max-steps 20 \
    #   --log-every 1 \
    #   --output-dir checkpoints/full_sanity

    if configure_parser is not None:
        configure_parser(p)
    args = p.parse_args(argv)
    for name in ("batch_size", "grad_accum", "max_length", "eval_batch_size", "generation_batch_size", "generation_max_new_tokens", "save_steps"):
        if getattr(args, name) < 1:
            p.error(f"--{name.replace('_', '-')} must be at least 1")
    if args.max_steps is not None and args.max_steps < 1:
        p.error("--max-steps must be at least 1")
    if not math.isfinite(args.epochs) or args.epochs <= 0:
        p.error("--epochs must be positive and finite")
    if args.log_every < 1 or args.eval_every < 0:
        p.error("--log-every must be positive and --eval-every must be nonnegative")
    if args.val_checks_per_epoch < 0 or args.generation_checks_per_epoch < 0:
        p.error("Validation checks per epoch must be nonnegative")
    if args.no_save and args.resume_from_checkpoint:
        p.error("--resume-from-checkpoint requires saving")
    if not math.isfinite(args.lr) or args.lr <= 0:
        p.error("--lr must be positive and finite")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        p.error("--weight-decay must be nonnegative and finite")
    if not 0 <= args.warmup_ratio <= 1:
        p.error("--warmup-ratio must be between 0 and 1")
    if not math.isfinite(args.max_grad_norm) or args.max_grad_norm <= 0:
        p.error("--max-grad-norm must be positive and finite")
    return args


def run_training(model, train_loader, val_loader, args, device, autocast_dtype, output_dir, checkpoints=None,
                 generation_evaluator=None):
    """Train, persist step history, and measure training separately from validation."""
    from transformers import get_scheduler
    from train.validation import is_better, validation_due, write_history

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
                        record.update({key: value for key, value in metrics.items() if key != "predictions_file"})
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


def main(argv=None, splits=None, *, args=None, model_loader=None, checkpoints=None, method="full", extra_report=None):
    """Run full fine-tuning and save the data, model, and report artifacts."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    run_started = time.perf_counter()
    args = args if args is not None else parse_args(argv)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    (weight_dtype, autocast_dtype), precision = resolve_precision(args.precision, device)
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.resume_from_checkpoint:
        raise FileExistsError("Output directory is not empty. Choose a new run directory, or resume its latest checkpoint.")
    output_dir.mkdir(parents=True, exist_ok=True)
    if checkpoints is None and not args.no_save:
        from train.checkpointing import AdapterCheckpoints
        checkpoints = AdapterCheckpoints(args)  # save_pretrained supports both full models and adapters.
    if checkpoints is not None and checkpoints.state:
        saved_step = checkpoints.state["progress"]["step"]
        if any(int(path.name.removeprefix("checkpoint-")) > saved_step for path in output_dir.glob("checkpoint-*")
               if path.name.removeprefix("checkpoint-").isdigit()):
            raise ValueError("Later checkpoints exist. Resume the latest one, or use a new output directory to branch.")
    environment = environment_info(device)
    code_hashes = source_hashes()

    # Data ------------------------------------------------------------------
    preprocessing_started = time.perf_counter()
    supplied_splits = splits is not None
    dataset_name = "provided_splits"
    if splits is None:
        from data.data import FunctionCallingDataset
        dataset_name = FunctionCallingDataset.DATASET_NAME
        splits = FunctionCallingDataset.load_splits(seed=DATA_SEED, unseen_fraction=UNSEEN_FRACTION)
    train_examples = list(splits["train"])[:args.limit_train]
    val_examples = list(splits["validation"])[:args.limit_val]
    tokenizer_revision_to_load = getattr(args, "revision", None)
    if checkpoints is not None and checkpoints.state:
        tokenizer_revision_to_load = checkpoints.state["base_revision"] or tokenizer_revision_to_load
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=tokenizer_revision_to_load)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_features, train_data = prepare_split(tokenizer, train_examples, args.max_length)
    val_features, validation_data = prepare_split(tokenizer, val_examples, args.max_length)
    if not train_features:
        raise SystemExit("No training examples fit within --max-length.")
    print(f"Train examples: {len(train_features)} (dropped {train_data['dropped_count']} over {args.max_length} tokens)")
    print(f"Validation examples: {len(val_features)} (dropped {validation_data['dropped_count']})")
    data_settings = {
        "dataset": dataset_name,
        "source_split": None if supplied_splits else "train",
        "split_seed": None if supplied_splits else DATA_SEED,
        "unseen_fraction": None if supplied_splits else UNSEEN_FRACTION,
        "limit_train": args.limit_train,
        "limit_val": args.limit_val,
        "max_length": args.max_length,
        "length_rule": "drop if prompt plus response (including ending token) exceeds max_length",
    }
    tokenizer_revision = getattr(tokenizer, "init_kwargs", {}).get("_commit_hash")
    data_manifest = {
        "settings": data_settings,
        "raw_splits": describe_splits(splits),
        "used_splits": {"train": train_data, "validation": validation_data},
        "tokenizer": args.model,
        "tokenizer_revision": tokenizer_revision,
        "chat_template_sha256": hashlib.sha256(
            json.dumps(tokenizer.chat_template, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest(),
        "code_sha256": code_hashes,
    }
    if method != "full":
        root = Path(__file__).resolve().parents[1]
        for name in ("train/QLoRA.py", "train/checkpointing.py"):
            code_hashes[name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
    if checkpoints is not None:
        checkpoints.prepare(data_manifest, args, tokenizer)
        checkpoints.record_environment(environment)
    write_json(output_dir / "data_splits.json", data_manifest)
    preprocessing_seconds = time.perf_counter() - preprocessing_started

    collate = make_collator(tokenizer.pad_token_id)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_features, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate, generator=generator)
    val_loader = DataLoader(val_features, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate)
    retained_validation_ids = set(validation_data["retained_ids"])
    generation_examples = [row for row in val_examples if row["id"] in retained_validation_ids]
    if args.generation_checks_per_epoch and generation_examples:
        from eval.metrics import validate_examples
        validate_examples(generation_examples)

    # Model -----------------------------------------------------------------
    synchronize(device)
    load_started = time.perf_counter()
    if model_loader is not None:
        model = model_loader(args, device, autocast_dtype or weight_dtype)
    else:
        source = args.resume_from_checkpoint or args.model
        options = {"dtype": weight_dtype}
        if not args.resume_from_checkpoint:
            options["revision"] = args.revision
        model = AutoModelForCausalLM.from_pretrained(source, **options).to(device)
        if checkpoints is not None and checkpoints.state:
            model.config._commit_hash = checkpoints.state["base_revision"]
    synchronize(device)
    model_load_seconds = time.perf_counter() - load_started
    model.config.use_cache = False
    if args.gradient_checkpointing and model_loader is None:
        model.gradient_checkpointing_enable()
    model.train()
    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if hasattr(model, "get_nb_trainable_parameters"):
        trainable_params, total_params = model.get_nb_trainable_parameters()
    print(f"Device: {device}, precision: {precision}, optimizer: {args.optim}")
    print(f"Parameters: {trainable_params:,} trainable / {total_params:,} total "
          f"({100 * trainable_params / total_params:.1f}%)")

    if checkpoints is not None:
        checkpoints.bind_model(model)
    generation_evaluator = None
    if args.generation_checks_per_epoch and generation_examples:
        from train.validation import generation_metrics

        def generation_evaluator(step):
            return generation_metrics(model, tokenizer, generation_examples, args, device, autocast_dtype, output_dir, step)

    training_metrics = run_training(model, train_loader, val_loader, args, device, autocast_dtype, output_dir, checkpoints,
                                    generation_evaluator=generation_evaluator)

    # Save the checkpoint before measuring its model/tokenizer artifact sizes.
    sizes = {"model_weights_bytes": None, "checkpoint_bytes": None}
    save_started = time.perf_counter()
    if not args.no_save:
        model.config.use_cache = True
        model.save_pretrained(output_dir, safe_serialization=True)
        tokenizer_files = tokenizer.save_pretrained(output_dir)
        sizes = checkpoint_sizes(output_dir, tokenizer_files)
    checkpoint_save_seconds = time.perf_counter() - save_started
    report = {
        "method": method,
        "model": args.model,
        "model_revision": getattr(model.config, "_commit_hash", None),
        "tokenizer_revision": tokenizer_revision,
        "device": str(device),
        "precision": precision,
        "weight_dtype": str(weight_dtype),
        "autocast_dtype": str(autocast_dtype) if autocast_dtype is not None else None,
        "settings": {key: value for key, value in vars(args).items() if key not in ("model", "device")},
        "environment": environment,
        "code_sha256": code_hashes,
        "data_settings": data_settings,
        "split_counts": {name: details["count"] for name, details in data_manifest["raw_splits"].items()},
        "train_examples_selected": len(train_examples),
        "train_examples": len(train_features),
        "train_examples_dropped": train_data["dropped_count"],
        "validation_examples_selected": len(val_examples),
        "validation_examples": len(val_features),
        "validation_examples_dropped": validation_data["dropped_count"],
        "trainable_parameters": trainable_params,
        "total_parameters": total_params,
        "trainable_parameter_percentage": 100 * trainable_params / total_params,
        "effective_batch_size": args.batch_size * args.grad_accum,
        "loss_objective": "mean over supervised response tokens in each accumulation window",
        "training_time_scope": "training loop including logging, excluding validation, preprocessing, model loading, and checkpoint saving",
        "preprocessing_seconds": preprocessing_seconds,
        "model_load_seconds": model_load_seconds,
        "checkpoint_save_seconds": checkpoint_save_seconds,
        "run_wall_seconds": time.perf_counter() - run_started,
        "data_splits_file": "data_splits.json",
        "training_history_file": "training_history.jsonl",
        "validation_history_file": "validation_history.jsonl",
        "validation_history_csv": "validation_history.csv",
        "validation_protocol": {
            "metric_protocol": "strict-json-v1", "split": "validation",
            "examples": len(generation_examples), "uses_retained_after_length_filter": True,
            "val_checks_per_epoch": args.val_checks_per_epoch, "eval_every_override": args.eval_every,
            "generation_checks_per_epoch": args.generation_checks_per_epoch,
            "generation_batch_size": args.generation_batch_size,
            "max_new_tokens": args.generation_max_new_tokens, "do_sample": False, "num_beams": 1,
            "generation_weights": "in-memory training weights; QLoRA retains NF4 base and FP32 adapters",
            "precision": precision, "warmup_rounding": "ceil",
            "selection_includes_initial_baseline": False,
        },
        "periodic_checkpoints_storage_bytes": checkpoints.storage_bytes() if checkpoints is not None else 0,
        "checkpoint_storage_scope": "logical file bytes; hard links may reduce occupied disk space",
        "memory_measurement_note": "Training tensor peaks exclude validation operations; reserved memory includes cached allocations retained from earlier phases. CUDA counters do not measure Windows shared-memory residency.",
        **training_metrics,
        **sizes,
    }
    if extra_report is not None:
        report.update(extra_report(model, report))
    write_json(output_dir / "train_metrics.json", report)

    gib = 1024 ** 3
    print(f"\n=== {method} fine-tuning summary ===")
    print(f"Optimizer steps:      {report['optimizer_steps']}")
    print(f"Epochs completed:     {report['epochs_completed']:.3f}")
    print(f"Training time:        {report['train_seconds']:.1f}s ({report['seconds_per_step']:.2f}s/step)")
    peak = report["peak_memory_bytes"]
    memory_text = f"{peak / gib:.2f} GiB" if peak is not None else "unavailable"
    print(f"{report['peak_memory_kind']}: {memory_text}")
    print(f"Trainable parameters: {trainable_params:,}")
    if report["val_loss_before"] is not None:
        print(f"Validation loss:      {report['val_loss_before']:.4f} -> {report['val_loss_after']:.4f}")
    if report["checkpoint_bytes"] is not None:
        print(f"Checkpoint:           {output_dir} ({report['checkpoint_bytes'] / gib:.2f} GiB)")
    print(f"Metrics saved to {output_dir / 'train_metrics.json'}")
    return report


if __name__ == "__main__":
    main()
