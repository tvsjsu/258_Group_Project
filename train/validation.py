"""Validation cadence and generation scoring on the in-memory training model."""

import csv
import json
from contextlib import nullcontext

import torch

from train.checkpointing import restore_rng, rng_state


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


def write_history(output_dir, records):
    """Rewrite from recovered state so resumed curves have no duplicate future rows."""
    with (output_dir / "validation_history.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
    if records:
        fields = list(dict.fromkeys(key for row in records for key in row))
        with (output_dir / "validation_history.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)


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
                if offset == 0 or offset + len(batch) >= len(examples) or (offset // args.generation_batch_size + 1) % 100 == 0:
                    print(f"[validation generation step {step}] {offset + len(batch)}/{len(examples)}", flush=True)
        temporary.replace(destination)
        return {**summarize(scores), "predictions_file": str(destination.relative_to(output_dir))}
    finally:
        temporary.unlink(missing_ok=True)
        tokenizer.padding_side = padding
        model.train(was_training)
        restore_rng(saved_rng)
