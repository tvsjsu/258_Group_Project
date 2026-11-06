"""Full fine-tuning of the base model on the xLAM train split.

Every model weight is trainable (no frozen layers, no adapters). Training uses
exactly the prompt that eval/baseline.py sends at inference time, and only the
response tokens (the JSON array of calls plus the end-of-turn token) count
toward the loss.

Initial test run (a few optimizer steps on a small slice of the data):

    uv run python -m train.finetune --max-steps 20 --limit-train 400 --limit-val 50

The script reports training time, peak memory, trainable parameter count,
validation loss before and after training, and checkpoint size, and saves them
to <output-dir>/train_metrics.json.
"""

import argparse
import json
import math
import random
import resource
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from model.prompt import build_messages

IGNORE_INDEX = -100


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


def encode_split(tokenizer, examples, max_length):
    """Encode examples; drop (not truncate) any longer than max_length."""
    features, dropped = [], 0
    for example in examples:
        feature = encode_example(tokenizer, example)
        if len(feature["input_ids"]) > max_length:
            dropped += 1
            continue
        features.append(feature)
    return features, dropped


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
    """Peak memory for the training process on cuda, mps, or cpu."""

    def __init__(self, device):
        self.device = device
        self.peak = 0
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

    def sample(self):
        if self.device.type == "mps":
            self.peak = max(self.peak, torch.mps.driver_allocated_memory())

    def peak_bytes(self):
        if self.device.type == "cuda":
            return torch.cuda.max_memory_allocated(self.device)
        if self.device.type == "mps":
            return self.peak
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return rss if sys.platform == "darwin" else rss * 1024

    def label(self):
        return {
            "cuda": "Peak GPU memory allocated",
            "mps": "Peak MPS driver memory",
            "cpu": "Peak process RSS",
        }[self.device.type]


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def directory_size(path):
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())


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


def parse_args(argv=None):
    from model.model import MODEL_NAME
    p = argparse.ArgumentParser(description="Full fine-tuning on the xLAM train split.")
    p.add_argument("--model", default=MODEL_NAME, help="Hugging Face id or local path.")
    p.add_argument("--output-dir", default="checkpoints/full")
    p.add_argument("--limit-train", type=int, default=None, help="Use only the first N train examples.")
    p.add_argument("--limit-val", type=int, default=100, help="Validation examples for loss (0 to skip).")
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=None, help="Stop after N optimizer steps (overrides --epochs).")
    p.add_argument("--batch-size", type=int, default=1, help="Micro-batch size per forward pass.")
    p.add_argument("--grad-accum", type=int, default=8, help="Micro-batches per optimizer step.")
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--max-length", type=int, default=2048, help="Drop examples longer than this many tokens.")
    p.add_argument("--optim", default="adamw", choices=["adamw", "adafactor"])
    p.add_argument("--precision", default="auto", choices=["auto", "fp32", "bf16-mixed", "bf16"],
                   help="auto = bf16-mixed on CUDA GPUs that support it, else fp32.")
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--device", default=None)
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--eval-every", type=int, default=0, help="Validation loss every N steps (0 = only before/after).")
    p.add_argument("--no-save", action="store_true", help="Skip writing the model checkpoint.")
    p.add_argument("--seed", type=int, default=42)

    # Sample command line for a quick sanity check (20 optimizer steps on 400 train examples, 50 validation examples):
    # uv run python -m train.finetune \
    #   --limit-train 400 \
    #   --limit-val 50 \
    #   --max-steps 20 \
    #   --log-every 1 \
    #   --output-dir checkpoints/full_sanity

    args = p.parse_args(argv)
    for name in ("batch_size", "grad_accum", "max_length"):
        if getattr(args, name) < 1:
            p.error(f"--{name.replace('_', '-')} must be at least 1")
    if args.max_steps is not None and args.max_steps < 1:
        p.error("--max-steps must be at least 1")
    return args


def main(argv=None, splits=None):
    """Run training. `splits` lets tests pass in-memory data instead of xLAM."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

    args = parse_args(argv)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    (weight_dtype, autocast_dtype), precision = resolve_precision(args.precision, device)

    # Data ------------------------------------------------------------------
    if splits is None:
        from data.data import FunctionCallingDataset
        splits = FunctionCallingDataset.load_splits(seed=42, unseen_fraction=0.2)
    train_examples = list(splits["train"])[:args.limit_train]
    val_examples = list(splits["validation"])[:args.limit_val] if args.limit_val else []

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_features, train_dropped = encode_split(tokenizer, train_examples, args.max_length)
    val_features, val_dropped = encode_split(tokenizer, val_examples, args.max_length)
    if not train_features:
        raise SystemExit("No training examples fit within --max-length.")
    print(f"Train examples: {len(train_features)} (dropped {train_dropped} over {args.max_length} tokens)")
    print(f"Validation examples: {len(val_features)} (dropped {val_dropped})")

    collate = make_collator(tokenizer.pad_token_id)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_features, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate, generator=generator)
    val_loader = DataLoader(val_features, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate)

    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_steps = args.max_steps or max(1, math.ceil(steps_per_epoch * args.epochs))

    # Model -----------------------------------------------------------------
    memory = MemoryTracker(device)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=weight_dtype).to(device)
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.train()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Device: {device}, precision: {precision}, optimizer: {args.optim}")
    print(f"Parameters: {trainable_params:,} trainable / {total_params:,} total "
          f"({100 * trainable_params / total_params:.1f}%)")
    print(f"Optimizer steps: {total_steps} (effective batch {args.batch_size * args.grad_accum})")

    optimizer = build_optimizer(args.optim, model, args.lr, args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * args.warmup_ratio), total_steps
    )

    val_loss_before = evaluate_loss(model, val_loader, device, autocast_dtype) if val_features else None
    if val_loss_before is not None:
        print(f"Validation loss before training: {val_loss_before:.4f}")

    # Loop ------------------------------------------------------------------
    history = []
    step, micro, tokens_seen = 0, 0, 0
    window_loss, window_tokens = 0.0, 0
    optimizer.zero_grad(set_to_none=True)
    synchronize(device)
    started = time.perf_counter()
    done = False
    while not done:
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            n_tokens = int((batch["labels"][:, 1:] != IGNORE_INDEX).sum())
            with autocast(device, autocast_dtype):
                loss = model(**batch).loss
            (loss / args.grad_accum).backward()
            window_loss += float(loss.detach()) * n_tokens
            window_tokens += n_tokens
            tokens_seen += int(batch["attention_mask"].sum())
            micro += 1
            memory.sample()
            if micro % args.grad_accum:
                continue

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            memory.sample()
            step += 1

            train_loss = window_loss / max(window_tokens, 1)
            window_loss, window_tokens = 0.0, 0
            record = {"step": step, "train_loss": train_loss,
                      "grad_norm": float(grad_norm), "lr": scheduler.get_last_lr()[0]}
            if args.eval_every and step % args.eval_every == 0 and val_features:
                record["val_loss"] = evaluate_loss(model, val_loader, device, autocast_dtype)
            history.append(record)
            if step % args.log_every == 0 or step == total_steps:
                elapsed = time.perf_counter() - started
                extra = f" val_loss {record['val_loss']:.4f}" if "val_loss" in record else ""
                print(f"[step {step}/{total_steps}] loss {train_loss:.4f} "
                      f"grad_norm {record['grad_norm']:.2f} lr {record['lr']:.2e}{extra} "
                      f"({elapsed:.0f}s)", flush=True)
            if step >= total_steps:
                done = True
                break
    synchronize(device)
    train_seconds = time.perf_counter() - started

    val_loss_after = evaluate_loss(model, val_loader, device, autocast_dtype) if val_features else None
    memory.sample()

    # Report ----------------------------------------------------------------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_bytes = None
    if not args.no_save:
        model.config.use_cache = True
        model.save_pretrained(output_dir, safe_serialization=True)
        tokenizer.save_pretrained(output_dir)
        checkpoint_bytes = directory_size(output_dir)

    report = {
        "method": "full",
        "model": args.model,
        "device": str(device),
        "precision": precision,
        "settings": {k: v for k, v in vars(args).items() if k not in ("model", "device")},
        "train_examples": len(train_features),
        "train_examples_dropped": train_dropped,
        "validation_examples": len(val_features),
        "optimizer_steps": step,
        "tokens_seen": tokens_seen,
        "trainable_parameters": trainable_params,
        "total_parameters": total_params,
        "train_seconds": train_seconds,
        "seconds_per_step": train_seconds / max(step, 1),
        "peak_memory_bytes": memory.peak_bytes(),
        "peak_memory_kind": memory.label(),
        "val_loss_before": val_loss_before,
        "val_loss_after": val_loss_after,
        "checkpoint_bytes": checkpoint_bytes,
        "history": history,
    }
    (output_dir / "train_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    gib = 1024 ** 3
    print("\n=== Full fine-tuning summary ===")
    print(f"Optimizer steps:      {step}")
    print(f"Training time:        {train_seconds:.1f}s ({train_seconds / max(step, 1):.2f}s/step)")
    print(f"{memory.label() + ':':<22}{memory.peak_bytes() / gib:.2f} GiB")
    print(f"Trainable parameters: {trainable_params:,}")
    if val_loss_before is not None:
        print(f"Validation loss:      {val_loss_before:.4f} -> {val_loss_after:.4f}")
    if checkpoint_bytes is not None:
        print(f"Checkpoint:           {output_dir} ({checkpoint_bytes / gib:.2f} GiB)")
    print(f"Metrics saved to {output_dir / 'train_metrics.json'}")
    return report


if __name__ == "__main__":
    main()
    