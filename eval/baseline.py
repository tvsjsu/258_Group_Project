import argparse
import json
import os
from contextlib import nullcontext
from pathlib import Path
from time import perf_counter

from eval.metrics import parse_calls, print_summary, score, summarize, validate_examples


def main():
    parser = argparse.ArgumentParser(description="Evaluate the base model with five function-calling metrics.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum examples per split (default: all).")
    parser.add_argument("--split", default="test_seen", choices=["test_seen", "test_unseen", "validation", "both"])
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--check-data-only", action="store_true", help="Validate selected examples and schemas without loading model weights.")
    output_options = parser.add_mutually_exclusive_group()
    output_options.add_argument("--output", default="results/baseline_predictions.jsonl")
    output_options.add_argument("--no-output", action="store_true", help="Print metrics without saving predictions or a metrics JSON file.")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be nonnegative")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be at least 1")

    from data.data import FunctionCallingDataset

    splits = FunctionCallingDataset.load_splits()
    selected_splits = ["test_seen", "test_unseen"] if args.split == "both" else [args.split]
    selected = {
        name: FunctionCallingDataset(splits[name][:args.limit])
        for name in selected_splits
    }
    for name, examples in selected.items():
        validate_examples(examples)
        print(f"Schema preflight passed: {name}, {len(examples)} examples", flush=True)
    if args.check_data_only:
        return

    # Model imports stay here so metric tests do not load PyTorch/model weights.
    from model.model import MODEL_NAME, FunctionCallingModel

    model = FunctionCallingModel(device=args.device, dtype=args.dtype)
    report = {
        "metric_protocol": "strict-json-v1",
        "settings": {
            "model": MODEL_NAME,
            "dtype": str(model.model.dtype),
            "device": str(model.model.device),
            "seed": 42,
            "unseen_fraction": 0.2,
            "batch_size": args.batch_size,
            "max_new_tokens": args.max_new_tokens,
            "group_by_length": False,
        },
        "splits": {},
    }
    all_scores = []
    if not args.no_output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    output_context = nullcontext() if args.no_output else open(args.output, "w", encoding="utf-8")
    with output_context as stream:
        for name, examples in selected.items():
            loader = examples.make_loader(
                batch_size=args.batch_size,
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
                    max_new_tokens=args.max_new_tokens,
                )
                if len(outputs) != len(batch):
                    raise RuntimeError("Model returned a different number of outputs than inputs")
                for example, output in zip(batch, outputs):
                    predicted = parse_calls(output)
                    result = score(predicted, example["answers"], example["tools"])
                    scores.append(result)
                    if stream is not None:
                        stream.write(json.dumps({
                            "metric_protocol": "strict-json-v1",
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
            report["splits"][name] = summarize(scores)
            all_scores.extend(scores)

    report["combined"] = summarize(all_scores)
    if args.split == "both":
        print_summary("combined", all_scores)
    if not args.no_output:
        summary_path = Path(args.output).with_suffix(".metrics.json")
        summary_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        print(f"Predictions saved to {args.output}")
        print(f"Metrics saved to {summary_path}")


if __name__ == "__main__":
    main()
