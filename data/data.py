import json
import random

from datasets import load_dataset

DATASET_NAME = "Salesforce/xlam-function-calling-60k"


def load_data():
    return load_dataset(DATASET_NAME, split="train")


def preprocess(rows):
    examples = []
    for row in rows:
        tools = json.loads(row["tools"])
        answers = json.loads(row["answers"])
        examples.append({
            "id": row["id"],
            "query": row["query"],
            "tools": tools,
            "answers": answers,
        })
    return examples


def tool_names(example):
    return {tool["name"] for tool in example["tools"]}


def split_examples(examples, seed=42, unseen_fraction=0.2):
    rng = random.Random(seed)
    names = sorted({name for example in examples for name in tool_names(example)})
    rng.shuffle(names)
    unseen_names = set(names[:int(len(names) * unseen_fraction)])

    seen, unseen = [], []
    for example in examples:
        names = tool_names(example)
        if names <= unseen_names:
            unseen.append(example)
        elif names.isdisjoint(unseen_names):
            seen.append(example)
        # Leave out examples that mix seen and unseen API definitions.

    rng.shuffle(seen)
    train_end = int(len(seen) * 0.8)
    validation_end = int(len(seen) * 0.9)
    splits = {
        "train": seen[:train_end],
        "validation": seen[train_end:validation_end],
        "test_seen": seen[validation_end:],
        "test_unseen": unseen,
    }

    # Seen evaluation APIs must appear in training.
    train_names = {name for example in splits["train"] for name in tool_names(example)}
    for split in ("validation", "test_seen"):
        kept = []
        for example in splits[split]:
            names = tool_names(example)
            if names <= train_names:
                kept.append(example)
            else:
                splits["train"].append(example)
                train_names.update(names)
        splits[split] = kept
    return splits
