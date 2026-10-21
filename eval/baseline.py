import argparse
import json
import os
from collections import Counter

from data.data import load_data, preprocess, split_examples


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


def main():
    parser = argparse.ArgumentParser(description="Baseline eval of the base model.")
    parser.add_argument("--limit", type=int, default=100, help="Number of examples to evaluate.")
    parser.add_argument("--split", default="test_seen", choices=["test_seen", "test_unseen", "validation"])
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--device", default=None, help="e.g. cuda, mps, cpu (default: cuda if available)")
    parser.add_argument("--output", default="results/baseline_predictions.jsonl")
    args = parser.parse_args()

    # Imported here so the parsing/metric functions can be used without loading torch.
    from model.model import FunctionCallingModel

    examples = split_examples(preprocess(load_data()))[args.split][:args.limit]
    model = FunctionCallingModel(device=args.device)

    scores = []
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        for i, example in enumerate(examples, 1):
            output = model.generate(example["query"], example["tools"], max_new_tokens=args.max_new_tokens)
            predicted = parse_calls(output)
            result = score(predicted, example["answers"])
            scores.append(result)
            f.write(json.dumps({
                "id": example["id"],
                "query": example["query"],
                "expected": example["answers"],
                "predicted": predicted,
                "raw_output": output,
                **result,
            }, ensure_ascii=False) + "\n")
            print(f"[{i}/{len(examples)}] valid={result['valid']} exact={result['exact_match']}")

    metrics = summarize(scores)
    print(f"\nSplit: {args.split}  Examples: {metrics['examples']}")
    print(f"Valid output rate:      {metrics['valid_rate']:.3f}")
    print(f"Function-name accuracy: {metrics['name_accuracy']:.3f}")
    print(f"Exact-match accuracy:   {metrics['exact_match']:.3f}")
    print(f"Predictions saved to {args.output}")


if __name__ == "__main__":
    main()
