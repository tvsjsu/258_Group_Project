import argparse
import json
import os
from collections import Counter
from contextlib import nullcontext
from time import perf_counter

def parse_calls(text):
    """Parse model output into a list of {"name", "arguments"} calls, or None if invalid."""
    # Tolerate a Markdown code fence (```json ... ```), even after a short preamble.
    if "```" in text:
        fenced = text.split("```", 2)[1]
        text = fenced.split("\n", 1)[1] if "\n" in fenced else ""
    text = text.strip()
    try:
        calls = json.loads(text)
    except json.JSONDecodeError:
        return None

    if not isinstance(calls, list):
        return None
    for call in calls:
        if not isinstance(call, dict) or set(call) != {"name", "arguments"}:
            return None
        if not isinstance(call["name"], str) or not isinstance(call["arguments"], dict):
            return None
    return calls


def _call_key(call):
    return json.dumps(call, sort_keys=True)


def score(predicted, expected):
    """Score one example. Call order is ignored; duplicates are counted."""
    if predicted is None:
        return {"valid": False, "name_match": False, "exact_match": False}
    return {
        "valid": True,
        "name_match": Counter(c["name"] for c in predicted) == Counter(c["name"] for c in expected),
        "exact_match": Counter(map(_call_key, predicted)) == Counter(map(_call_key, expected)),
    }


def summarize(scores):
    n = len(scores)
    return {
        "examples": n,
        "valid_rate": sum(s["valid"] for s in scores) / n if n else 0.0,
        "name_accuracy": sum(s["name_match"] for s in scores) / n if n else 0.0,
        "exact_match": sum(s["exact_match"] for s in scores) / n if n else 0.0,
    }


def print_summary(label, scores):
    metrics = summarize(scores)
    print(f"\nSplit: {label}  Examples: {metrics['examples']}")
    print(f"Valid output rate:      {metrics['valid_rate']:.3f}")
    print(f"Function-name accuracy: {metrics['name_accuracy']:.3f}")
    print(f"Exact-match accuracy:   {metrics['exact_match']:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Baseline eval of the base model.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Maximum examples per split (default: all).")
    parser.add_argument("--split", default="test_seen",
                        choices=["test_seen", "test_unseen", "validation", "both"],
                        help="Split to evaluate; 'both' evaluates test_seen and test_unseen.")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Examples per generation call (default: 1; increase on GPU).")
    parser.add_argument("--dtype", default="auto",
                        choices=["auto", "float16", "bfloat16", "float32"],
                        help="Weight precision (auto: checkpoint dtype on CUDA, float32 elsewhere).")
    parser.add_argument("--device", default=None, help="e.g. cuda, mps, cpu (default: cuda if available)")
    output_options = parser.add_mutually_exclusive_group()
    output_options.add_argument("--output", default="results/baseline_predictions.jsonl",
                                help="Prediction JSONL path (default: results/baseline_predictions.jsonl).")
    output_options.add_argument("--no-output", action="store_true",
                                help="Calculate metrics without serializing or saving prediction records.")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be nonnegative")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be at least 1")

    # Imported here so the parsing/metric functions can be used without loading torch.
    from data.data import FunctionCallingDataset
    from model.model import FunctionCallingModel

    splits = FunctionCallingDataset.load_splits()
    selected_splits = ["test_seen", "test_unseen"] if args.split == "both" else [args.split]
    model = FunctionCallingModel(device=args.device, dtype=args.dtype)

    all_scores = []
    if not args.no_output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with (nullcontext() if args.no_output else open(args.output, "w")) as f:
        for split in selected_splits:
            examples = FunctionCallingDataset(
            splits[split][:args.limit]
            )
            scores = []
            done = 0

            # Include length calculation in the evaluation timing.
            started = perf_counter()

            loader = examples.make_loader(
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=0,
                tokenizer=model.tokenizer,
                group_by_length=True,
            )
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
                    result = score(predicted, example["answers"])
                    scores.append(result)
                    if f is not None:
                        f.write(json.dumps({
                            "split": split,
                            "id": example["id"],
                            "query": example["query"],
                            "expected": example["answers"],
                            "predicted": predicted,
                            "raw_output": output,
                            **result,
                        }, ensure_ascii=False) + "\n")
                elapsed = perf_counter() - started
                done += len(batch)
                print(f"[{split} {done}/{len(examples)}] "
                      f"{done / max(elapsed, 1e-9):.2f} examples/s", flush=True)
            print_summary(split, scores)
            all_scores.extend(scores)

    if args.split == "both":
        print_summary("combined", all_scores)
    if not args.no_output:
        print(f"Predictions saved to {args.output}")


if __name__ == "__main__":
    main()
