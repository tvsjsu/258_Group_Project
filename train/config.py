"""Define shared training options and method-specific adapter settings."""

import argparse
import math


def example_limit(value):
    """Parse a nonnegative example count or an unrestricted all-examples limit."""
    if value == "all":
        return None
    try:
        limit = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use a nonnegative integer or 'all'") from exc
    if limit < 0:
        raise argparse.ArgumentTypeError("Use a nonnegative integer or 'all'")
    return limit


def parse_args(argv=None, configure_parser=None):
    """Parse common training options and apply optional method-specific defaults."""
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
    p.add_argument("--optim", default="adamw8bit", choices=["adamw8bit", "adamw", "adafactor"],
                   help="Shared default: bitsandbytes 8-bit AdamW on CUDA. Use adamw for standard PyTorch AdamW.")
    p.add_argument("--precision", default="auto", choices=["auto", "fp32", "bf16-mixed", "bf16"],
                   help="auto = bf16-mixed on CUDA GPUs that support it, else fp32.")
    p.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--device", default=None)
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--eval-every", type=int, default=0, help="Override epoch-based loss checks with every N optimizer steps (0 = automatic epoch cadence).")
    p.add_argument("--val-checks-per-epoch", type=int, default=10)
    p.add_argument("--generation-checks-per-epoch", type=int, default=1, help="Generate once per epoch by default; 0 disables generated validation and best-model selection.")
    p.add_argument("--generation-limit", type=example_limit, default=256,
                   help="Fixed seeded subset of retained validation rows for generation only (default: 256; 'all' for all rows). Does not reduce loss validation.")
    p.add_argument("--generation-subset-seed", type=int, default=42,
                   help="Subset seed shared across methods, independent of --seed (default: 42).")
    p.add_argument("--eval-batch-size", type=int, default=2, help="Independent teacher-forced validation micro-batch.")
    p.add_argument("--generation-batch-size", type=int, default=4)
    p.add_argument("--generation-max-new-tokens", type=int, default=512)
    p.add_argument("--lr-scheduler", choices=["cosine", "linear"], default="cosine")
    p.add_argument("--revision", default=None, help="Optional base model/tokenizer commit ID.")
    p.add_argument("--save-steps", type=int, default=100, help="Resumable checkpoint interval; also save at epoch ends, generation checks and final step.")
    p.add_argument("--resume-from-checkpoint", default=None)
    p.add_argument("--reference-data", default=None)
    p.add_argument("--no-save", action="store_true", help="Skip writing the model checkpoint.")
    p.add_argument("--seed", type=int, default=42)

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


def add_adapter_arguments(parser):
    """Register the rank, scaling, dropout, and target-layer options for adapters."""
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", nargs="+", default=["all-linear"])


def validate_adapter_args(args):
    """Validate adapter settings and require checkpoint saving."""
    if args.rank < 1 or args.lora_alpha < 1:
        raise ValueError("--rank and --lora-alpha must be positive")
    if not 0 <= args.lora_dropout < 1:
        raise ValueError("--lora-dropout must be in [0, 1)")
    if "all-linear" in args.target_modules and args.target_modules != ["all-linear"]:
        raise ValueError("Use 'all-linear' alone or supply individual target module names")
    if args.no_save:
        raise ValueError("Adapter methods save their adapter and resumable checkpoints; omit --no-save")
    return args
