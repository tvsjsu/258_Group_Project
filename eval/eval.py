"""Evaluate full models and adapters with the shared function-calling protocol."""

import argparse
import json
from contextlib import nullcontext
from pathlib import Path
from time import perf_counter

from eval.metrics import parse_calls, print_summary, score, summarize, validate_examples


METRIC_PROTOCOL = "strict-json-v1"


def parse_args(argv=None):
    """Parse and validate evaluation options."""
    parser = argparse.ArgumentParser(
        description="Evaluate a Hugging Face causal language model or local full-model checkpoint with five function-calling metrics."
    )
    parser.add_argument(
        "--model", default=None,
        help="Hugging Face model ID or local full-model checkpoint directory (default: deepseek-ai/deepseek-coder-1.3b-instruct).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Maximum examples per split (default: all).")
    parser.add_argument("--adapter", default=None, help="PEFT adapter directory; infer its base model when --model is omitted.")
    parser.add_argument("--load-in-4bit", action="store_true", help="Evaluate with an NF4/double-quantized CUDA base; default uses the requested inference dtype.")
    parser.add_argument("--split", default="test_seen", choices=["test_seen", "test_unseen", "validation", "both"])
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--check-data-only", action="store_true", help="Validate selected examples and schemas without loading model weights.")
    output_options = parser.add_mutually_exclusive_group()
    output_options.add_argument("--output", default="results/eval_predictions.jsonl")
    output_options.add_argument("--no-output", action="store_true", help="Print metrics without saving predictions or a metrics JSON file.")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be nonnegative")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be at least 1")
    return args


def prepare_splits(split, limit=None):
    """Load, select, and validate examples before loading the model."""
    from data.data import FunctionCallingDataset

    splits = FunctionCallingDataset.load_splits()
    selected_splits = ["test_seen", "test_unseen"] if split == "both" else [split]
    selected = {
        name: FunctionCallingDataset(splits[name][:limit])
        for name in selected_splits
    }
    for name, examples in selected.items():
        validate_examples(examples)
        print(f"Schema preflight passed: {name}, {len(examples)} examples", flush=True)
    return selected


def load_model(model_name=None, *, device=None, dtype="auto", adapter=None, load_in_4bit=False):
    """Lazily load the requested model or adapter and return its resolved base-model name."""

    from model.model import MODEL_NAME, FunctionCallingModel

    if adapter is not None:
        from peft import PeftConfig
        base_name = PeftConfig.from_pretrained(adapter).base_model_name_or_path
        if model_name is not None and model_name != base_name:
            raise ValueError("--model must match the base model recorded in the adapter config; omit it to infer the base")
        model_name = base_name
    model_name = model_name if model_name is not None else MODEL_NAME
    options = {}
    if adapter is not None:
        options["adapter_name"] = adapter
    if load_in_4bit:
        options["load_in_4bit"] = True
    model = FunctionCallingModel(model_name=model_name, device=device, dtype=dtype, **options)
    return model, model_name


def evaluate_split(model, name, examples, *, batch_size=1, max_new_tokens=512, prediction_stream=None):
    """Generate and score one split, optionally writing prediction records."""
    loader = examples.make_loader(
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    scores = []
    done = 0
    started = perf_counter()
    for batch in loader:
        outputs = model.generate_batch(
            [example["query"] for example in batch],
            [example["tools"] for example in batch],
            max_new_tokens=max_new_tokens,
        )
        if len(outputs) != len(batch):
            raise RuntimeError("Model returned a different number of outputs than inputs")
        for example, output in zip(batch, outputs):
            predicted = parse_calls(output)
            result = score(predicted, example["answers"], example["tools"])
            scores.append(result)
            if prediction_stream is not None:
                prediction_stream.write(json.dumps({
                    "metric_protocol": METRIC_PROTOCOL,
                    "split": name,
                    "id": example["id"],
                    "query": example["query"],
                    "tools": example["tools"],
                    "expected": example["answers"],
                    "predicted": predicted,
                    "raw_output": output,
                    **result,
                }, ensure_ascii=False, allow_nan=False) + "\n")
        done += len(batch)
        elapsed = perf_counter() - started
        print(f"[{name} {done}/{len(examples)}] {done / max(elapsed, 1e-9):.2f} examples/s", flush=True)
    print_summary(name, scores)
    return scores


def evaluate_model(model, model_name, splits, *, batch_size=1, max_new_tokens=512, output_path=None):
    """Evaluate selected splits and aggregate their scores into one report."""
    report = {
        "metric_protocol": METRIC_PROTOCOL,
        "settings": {
            "model": model_name,
            "dtype": str(model.model.dtype),
            "device": str(model.model.device),
            "seed": 42,
            "unseen_fraction": 0.2,
            "batch_size": batch_size,
            "max_new_tokens": max_new_tokens,
            "group_by_length": False,
            "quantization": getattr(model, "quantization_settings", None),
            "adapter_parameter_dtypes": getattr(model, "adapter_parameter_dtypes", []),
        },
        "splits": {},
    }
    all_scores = []
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    output_context = nullcontext() if output_path is None else open(output_path, "w", encoding="utf-8")
    with output_context as stream:
        for name, examples in splits.items():
            scores = evaluate_split(
                model, name, examples,
                batch_size=batch_size,
                max_new_tokens=max_new_tokens,
                prediction_stream=stream,
            )
            report["splits"][name] = summarize(scores)
            all_scores.extend(scores)

    report["combined"] = summarize(all_scores)
    if len(splits) > 1:
        print_summary("combined", all_scores)
    return report


def save_report(report, output_path):
    """Save aggregate metrics alongside the predictions file."""
    summary_path = Path(output_path).with_suffix(".metrics.json")
    summary_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Predictions saved to {output_path}")
    print(f"Metrics saved to {summary_path}")


def main(argv=None):
    """Run schema preflight, generate predictions, and save aggregate evaluation metrics."""
    args = parse_args(argv)
    splits = prepare_splits(args.split, args.limit)
    if args.check_data_only:
        return

    model, model_name = load_model(
        args.model, device=args.device, dtype=args.dtype,
        adapter=args.adapter, load_in_4bit=args.load_in_4bit,
    )
    output_path = None if args.no_output else args.output
    report = evaluate_model(
        model, model_name, splits,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        output_path=output_path,
    )
    report["settings"]["adapter"] = args.adapter
    report["settings"]["base_quantization"] = "nf4-double-quantized" if args.load_in_4bit else "unquantized"
    if output_path is not None:
        save_report(report, output_path)
    return report


if __name__ == "__main__":
    main()
