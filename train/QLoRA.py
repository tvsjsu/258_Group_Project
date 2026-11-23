"""Configure and build NF4-quantized LoRA training runs for the shared trainer."""

from train.config import add_adapter_arguments, parse_args as parse_config, validate_adapter_args
from train.reporting import adapter_report
from train.trainer import run_experiment
from train.runtime import require_cuda


def configure_parser(parser):
    """Set QLoRA defaults and register shared adapter options."""
    parser.description = "Train QLoRA with the baseline data/prompt and comparable report artifacts."
    parser.set_defaults(output_dir="checkpoints/qlora", lr=1e-4, batch_size=2, grad_accum=8, gradient_checkpointing=True)
    add_adapter_arguments(parser)


def parse_args(argv=None):
    """Parse and validate the QLoRA training configuration."""
    return validate_adapter_args(parse_config(argv, configure_parser))


def build_model(args, context):
    """Load and prepare an NF4 base with newly created or restored LoRA adapters."""
    device, compute_dtype = context.device, context.compute_dtype
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


def describe_model(model, args, report, context):
    """Describe adapter settings, quantization, and saved QLoRA artifacts."""
    return {
        **adapter_report(model, args, report),
        "weight_dtype": "mixed: frozen 4-bit NF4 base; higher-precision adapters and unquantized modules",
        "quantization": {"bits": 4, "type": "nf4", "double_quantization": True,
                         "compute_dtype": str(context.compute_dtype)},
    }


def main(argv=None):
    """Resolve CUDA requirements and launch the shared QLoRA experiment."""
    args = parse_args(argv)
    args.device = str(require_cuda(args.device, args.precision))
    return run_experiment(args, build_model, method="qlora", describe_model=describe_model,
                          method_source=__file__)


if __name__ == "__main__":
    main()
