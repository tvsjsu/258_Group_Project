"""Configure and build full fine-tuning runs for the shared trainer."""

from train.config import parse_args
from train.trainer import run_experiment


def build_model(args, context):
    """Load trainable full-model weights and configure gradient checkpointing."""
    from transformers import AutoModelForCausalLM

    source = args.resume_from_checkpoint or args.model
    options = {"dtype": context.weight_dtype}
    if not args.resume_from_checkpoint:
        options["revision"] = args.revision
    model = AutoModelForCausalLM.from_pretrained(source, **options).to(context.device)
    if args.resume_from_checkpoint:
        model.config._commit_hash = context.base_revision
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    return model


def main(argv=None, splits=None):
    """Launch full fine-tuning through the shared experiment runner."""
    return run_experiment(parse_args(argv), build_model, method="full", splits=splits, method_source=__file__)


if __name__ == "__main__":
    main()
