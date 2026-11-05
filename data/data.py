import json
import random

from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

class FunctionCallingDataset(Dataset):
    """One prepared split of function-calling examples."""

    DATASET_NAME = "Salesforce/xlam-function-calling-60k"

    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]

    @classmethod
    def load_splits(cls, seed=42, unseen_fraction=0.2):
        """Prepare the data once and return a Dataset for each split."""
        rows = cls._load_data()
        examples = cls._preprocess(rows)
        groups = cls._split_examples(
            examples,
            seed=seed,
            unseen_fraction=unseen_fraction,
        )

        return {
            name: cls(items)
            for name, items in groups.items()
        }

    @classmethod
    def _load_data(cls):
        return load_dataset(cls.DATASET_NAME, split="train")

    @staticmethod
    def _preprocess(rows):
        examples = []

        for batch in rows.iter(batch_size=1000, drop_last_batch=False):
            for row_id, query, tools, answers in zip(
                batch["id"],
                batch["query"],
                batch["tools"],
                batch["answers"],
            ):
                examples.append({
                    "id": row_id,
                    "query": query,
                    "tools": json.loads(tools),
                    "answers": json.loads(answers),
                })

        return examples

    @staticmethod
    def _tool_names(example):
        return {tool["name"] for tool in example["tools"]}

    @classmethod
    def _split_examples(cls, examples, seed=42, unseen_fraction=0.2):
        rng = random.Random(seed)

        names = sorted({
            name
            for example in examples
            for name in cls._tool_names(example)
        })
        rng.shuffle(names)

        unseen_names = set(
            names[:int(len(names) * unseen_fraction)]
        )

        seen, unseen = [], []

        for example in examples:
            names = cls._tool_names(example)

            if names <= unseen_names:
                unseen.append(example)
            elif names.isdisjoint(unseen_names):
                seen.append(example)

            # Exclude examples that mix seen and unseen tools.

        rng.shuffle(seen)

        train_end = int(len(seen) * 0.8)
        validation_end = int(len(seen) * 0.9)

        splits = {
            "train": seen[:train_end],
            "validation": seen[train_end:validation_end],
            "test_seen": seen[validation_end:],
            "test_unseen": unseen,
        }

        # Every seen evaluation tool must also appear in training.
        train_names = {
            name
            for example in splits["train"]
            for name in cls._tool_names(example)
        }

        for split in ("validation", "test_seen"):
            kept = []

            for example in splits[split]:
                names = cls._tool_names(example)

                if names <= train_names:
                    kept.append(example)
                else:
                    splits["train"].append(example)
                    train_names.update(names)

            splits[split] = kept

        return splits

    def make_loader(self, batch_size=1, shuffle=False, num_workers=0):
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self._collate_examples,
            drop_last=False,
        )

    @staticmethod
    def _collate_examples(batch):
        """Keep each example's nested tools and answers intact."""
        return batch
