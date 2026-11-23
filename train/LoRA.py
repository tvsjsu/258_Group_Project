"""Configure and build LoRA adapter training runs for the shared trainer."""

from train.config import add_adapter_arguments, parse_args as parse_config, validate_adapter_args
from train.reporting import adapter_report
from train.trainer import run_experiment


def configure_parser(parser):
    """Set LoRA defaults and register shared adapter options."""
    parser.description = "LoRA on the shared function-calling data and evaluation protocol."
    parser.set_defaults(output_dir="checkpoints/lora", lr=1e-4, batch_size=1, grad_accum=16)
    add_adapter_arguments(parser)


def parse_args(argv=None):
    """Parse and validate the LoRA training configuration."""
    return validate_adapter_args(parse_config(argv, configure_parser))


def build_model(args, context):
    """Load an unquantized base and create or restore trainable LoRA adapters."""
    device, compute_dtype = context.device, context.compute_dtype
    from peft import LoraConfig, PeftConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM

    revision = (PeftConfig.from_pretrained(args.resume_from_checkpoint).revision
                if args.resume_from_checkpoint else args.revision)
    revision = revision or args.revision
    weight_dtype = context.weight_dtype
    base = AutoModelForCausalLM.from_pretrained(args.model, revision=revision, dtype=weight_dtype).to(device)
    if args.resume_from_checkpoint:
        model = PeftModel.from_pretrained(base, args.resume_from_checkpoint, is_trainable=True)
    else:
        model = get_peft_model(base, LoraConfig(
            task_type="CAUSAL_LM", r=args.rank, lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout, bias="none",
            revision=getattr(base.config, "_commit_hash", None) or args.revision,
            target_modules="all-linear" if args.target_modules == ["all-linear"] else args.target_modules,
        ))
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable or any("lora_" not in name for name in trainable):
        raise RuntimeError("Expected only LoRA adapter parameters to be trainable")
    return model


def describe_model(model, args, report, context):
    """Describe the adapter settings and saved artifacts for a LoRA run."""
    return {**adapter_report(model, args, report), "quantization": None}


def main(argv=None):
    """Launch LoRA training through the shared experiment runner."""
    args = parse_args(argv)
    return run_experiment(args, build_model, method="lora", describe_model=describe_model,
                          method_source=__file__)


if __name__ == "__main__":
    main()
