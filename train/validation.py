"""Measure validation loss and score generated function calls during training."""

import gc
import json
import time
from contextlib import nullcontext

import torch

from train.checkpointing import restore_rng, rng_state
from train.preparation import IGNORE_INDEX
from train.runtime import autocast, synchronize


def release_validation_cache(device):
    """Release unused allocations at a phase boundary, never live model tensors."""
    if device.type == "cuda" and torch.cuda.is_available():
        with torch.cuda.device(device):
            gc.collect()
            torch.cuda.empty_cache()


def generation_oom_message(step, batch_size):
    """Explain a generation memory failure and suggest an explicit smaller batch."""
    smaller = max(1, batch_size // 2)
    if batch_size == 1:
        advice = "Generation already uses batch size 1; inspect live GPU memory and other GPU processes."
    else:
        advice = f"Restart with --generation-batch-size {smaller} or smaller; keep the chosen value fixed across comparison runs."
    return (
        f"Out of memory during validation generation at optimizer step {step}, generation batch size {batch_size}. "
        "The training-prepared model and its memory footprint differ from a separately loaded inference model. "
        f"{advice} Training --batch-size and --grad-accum are separate settings. "
        "No automatic batch-size change was made."
    )


def validation_due(previous_batch, current_batch, batches_per_epoch, checks):
    """Cross evenly spaced epoch fractions, rounded to optimizer boundaries."""
    if checks == 0:
        return False
    before = previous_batch * checks // batches_per_epoch
    after = current_batch * checks // batches_per_epoch
    return after > before


def is_better(candidate, best):
    """Select only trained checkpoints: exact-call accuracy, then loss, then earlier step."""
    if best is None:
        return True
    accuracy = candidate["complete_call_accuracy"]
    other = best["complete_call_accuracy"]
    return accuracy > other or (
        accuracy == other and candidate["val_loss"] < best["val_loss"]
    )


@torch.no_grad()
def generation_metrics(model, tokenizer, examples, args, device, autocast_dtype, output_dir, step):
    """Same prompt, strict scoring, greedy decoding and EOS handling as eval/eval.py.

    Reuse training weights (including NF4 for QLoRA); do not load a second model.
    Preserve training RNG and tokenizer padding across the validation operation.
    """
    from eval.eval import METRIC_PROTOCOL
    from eval.metrics import parse_calls, score, summarize
    from model.model import FunctionCallingModel

    saved_rng = rng_state()
    was_training, padding = model.training, tokenizer.padding_side
    wrapper = FunctionCallingModel.__new__(FunctionCallingModel)
    wrapper.model, wrapper.tokenizer = model, tokenizer
    model.eval()
    tokenizer.padding_side = "left"
    release_validation_cache(device)
    scores = []
    directory = output_dir / "validation_predictions"
    directory.mkdir(exist_ok=True)
    destination = directory / f"step-{step}.jsonl"
    temporary = destination.with_suffix(".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            for offset in range(0, len(examples), args.generation_batch_size):
                batch = examples[offset:offset + args.generation_batch_size]
                context = (torch.autocast(device_type=device.type, dtype=autocast_dtype)
                           if autocast_dtype is not None else nullcontext())
                with context:
                    outputs = wrapper.generate_batch(
                        [row["query"] for row in batch], [row["tools"] for row in batch],
                        max_new_tokens=args.generation_max_new_tokens,
                    )
                if len(outputs) != len(batch):
                    raise RuntimeError("Generation returned a different number of outputs than inputs")
                for row, output in zip(batch, outputs):
                    predicted = parse_calls(output)
                    result = score(predicted, row["answers"], row["tools"])
                    scores.append(result)
                    stream.write(json.dumps({
                        "metric_protocol": METRIC_PROTOCOL, "split": "validation",
                        "id": row["id"], "expected": row["answers"],
                        "predicted": predicted, "raw_output": output, **result,
                    }, ensure_ascii=False, allow_nan=False) + "\n")
                print(f"[validation generation step {step}] {offset + len(batch)}/{len(examples)}", flush=True)
        temporary.replace(destination)
        return {**summarize(scores), "predictions_file": str(destination.relative_to(output_dir)),
                "generation_batch_size": args.generation_batch_size}
    except torch.OutOfMemoryError as error:
        diagnosis = {
            "phase": "validation_generation", "optimizer_step": step,
            "generation_batch_size": args.generation_batch_size,
            "max_new_tokens": args.generation_max_new_tokens,
            "successfully_scored_examples": len(scores),
            "batch_size_automatically_changed": False,
            "embedding_dtype": str(model.get_input_embeddings().weight.dtype)
            if hasattr(model, "get_input_embeddings") else None,
            "original_error": str(error),
            "message": generation_oom_message(step, args.generation_batch_size),
        }
        (output_dir / "validation_oom.json").write_text(
            json.dumps(diagnosis, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        raise RuntimeError(diagnosis["message"]) from error
    finally:
        temporary.unlink(missing_ok=True)
        tokenizer.padding_side = padding
        model.train(was_training)
        restore_rng(saved_rng)
        release_validation_cache(device)


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


def timed_validation(model, loader, device, autocast_dtype):
    """Measure synchronized teacher-forced validation loss and elapsed time."""
    synchronize(device)
    started = time.perf_counter()
    loss = evaluate_loss(model, loader, device, autocast_dtype)
    synchronize(device)
    return loss, time.perf_counter() - started
