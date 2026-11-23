"""Full fine-tuning entry point; shared experiment mechanics live in trainer.py.

    uv run python -m train.finetune --output-dir checkpoints/full_final
"""
from train.config import parse_args
from train.trainer import run_experiment


def build_model(args, context):
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
    return run_experiment(parse_args(argv), build_model, method="full", splits=splits, method_source=__file__)


if __name__ == "__main__":
    main()
