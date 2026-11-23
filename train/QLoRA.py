"""QLoRA on the shared baseline splits; CUDA, NF4, double quantization, PEFT LoRA.

    uv sync --extra qlora
    uv run python -m train.QLoRA --limit-val all --output-dir checkpoints/qlora

Training and report generation reuse train.finetune. Adapter checkpoints retain
optimizer/scheduler, RNG, history, and sampler position for interrupted runs.
"""

from pathlib import Path

import torch

from train import finetune
from train.checkpointing import AdapterCheckpoints


def configure_parser(parser):
    parser.description = "Train QLoRA with the baseline data/prompt and comparable report artifacts."
    parser.set_defaults(output_dir="checkpoints/qlora", lr=2e-4, gradient_checkpointing=True)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", nargs="+", default=["all-linear"])
    parser.add_argument("--revision", default=None, help="Optional base model/tokenizer commit ID.")
    parser.add_argument("--save-steps", type=int, default=100, help="Checkpoint every N optimizer steps, plus epoch ends and final step.")
    parser.add_argument("--resume-from-checkpoint", default=None, help="Path to a checkpoint-N folder, not the final adapter folder.")
    parser.add_argument("--reference-data", default=None, help="Optional full fine-tuning data_splits.json; stop if split/filtering identities differ.")


def parse_args(argv=None):
    args = finetune.parse_args(argv, configure_parser)
    if args.rank < 1 or args.lora_alpha < 1 or args.save_steps < 1:
        raise ValueError("--rank, --lora-alpha, and --save-steps must be positive")
    if not 0 <= args.lora_dropout < 1:
        raise ValueError("--lora-dropout must be in [0, 1)")
    if "all-linear" in args.target_modules and args.target_modules != ["all-linear"]:
        raise ValueError("Use 'all-linear' alone or supply individual target module names")
    if args.no_save:
        raise ValueError("QLoRA always saves its final adapter and resumable checkpoints; omit --no-save")
    return args


def build_model(args, device, compute_dtype):
    from peft import LoraConfig, PeftConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    revision = args.revision
    if args.resume_from_checkpoint:
        revision = PeftConfig.from_pretrained(args.resume_from_checkpoint).revision or revision
    base = AutoModelForCausalLM.from_pretrained(
        args.model, revision=revision, dtype=compute_dtype,
        quantization_config=quantization, device_map={"": device.index if device.index is not None else torch.cuda.current_device()},
    )
    base = prepare_model_for_kbit_training(
        base, use_gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    if args.resume_from_checkpoint:
        model = PeftModel.from_pretrained(base, args.resume_from_checkpoint, is_trainable=True)
    else:
        config = LoraConfig(
            task_type="CAUSAL_LM", r=args.rank, lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout, bias="none",
            revision=getattr(base.config, "_commit_hash", None) or args.revision,
            target_modules="all-linear" if args.target_modules == ["all-linear"] else args.target_modules,
        )
        model = get_peft_model(base, config)
    if not getattr(model, "is_loaded_in_4bit", False):
        raise RuntimeError("The base model was not loaded in four-bit precision")
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable or any("lora_" not in name for name in trainable):
        raise RuntimeError("Expected only LoRA adapter parameters to be trainable")
    model.config.use_cache = False
    return model


def main(argv=None):
    args = parse_args(argv)
    device = finetune.pick_device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This QLoRA training configuration requires an NVIDIA CUDA GPU. Install dependencies with: uv sync --extra qlora")
    device = torch.device("cuda", device.index if device.index is not None else torch.cuda.current_device())
    args.device = str(device)
    torch.cuda.set_device(device)
    if args.precision.startswith("bf16") and not torch.cuda.is_bf16_supported():
        raise ValueError("This GPU does not support BF16; use --precision fp32")
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.resume_from_checkpoint:
        raise FileExistsError("Output directory is not empty. Use a new run directory or --resume-from-checkpoint.")
    checkpoints = AdapterCheckpoints(args)
    if checkpoints.state:
        saved_step = checkpoints.state["progress"]["step"]
        if any(path.name.removeprefix("checkpoint-").isdigit() and int(path.name.removeprefix("checkpoint-")) > saved_step
               for path in output_dir.glob("checkpoint-*")):
            raise FileExistsError("Later checkpoints exist in this output directory. Resume the latest checkpoint, or branch into a new output directory.")

    def additional_report(model, report):
        dtype = finetune.resolve_precision(args.precision, device)[0][1]
        compute = dtype or finetune.resolve_precision(args.precision, device)[0][0]
        return {
            "weight_dtype": "mixed: frozen 4-bit NF4 base; higher-precision adapters and unquantized modules",
            "quantization": {"bits": 4, "type": "nf4", "double_quantization": True, "compute_dtype": str(compute)},
            "lora": {"rank": args.rank, "alpha": args.lora_alpha, "dropout": args.lora_dropout, "target_modules": args.target_modules},
            "adapter_parameter_dtypes": sorted({str(parameter.dtype) for parameter in model.parameters() if parameter.requires_grad}),
            "trainable_parameter_names": [name for name, parameter in model.named_parameters() if parameter.requires_grad],
            "checkpoint_artifact": "adapter and tokenizer; base model required separately",
            "adapter_weights_bytes": report["model_weights_bytes"],
            "base_model_required": args.model,
            "loaded_model_memory_footprint_bytes": model.get_memory_footprint(),
            "periodic_checkpoints_storage_bytes": checkpoints.storage_bytes(),
            "resume_from_checkpoint": args.resume_from_checkpoint,
            "training_sessions": checkpoints.sessions(),
            "run_wall_time_scope": "current invocation only",
            "resume_accounting": "training totals include recovered checkpoint progress plus this invocation; discarded work since the last checkpoint is excluded",
        }

    return finetune.main(
        args=args, model_loader=build_model, checkpoints=checkpoints,
        method="qlora", extra_report=additional_report,
    )


if __name__ == "__main__":
    main()
