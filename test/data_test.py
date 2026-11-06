import json

import pytest

from data.data import FunctionCallingDataset


@pytest.fixture(scope="module")
def raw_dataset():
    return FunctionCallingDataset._load_data()


@pytest.fixture(scope="module")
def examples(raw_dataset):
    return FunctionCallingDataset._preprocess(raw_dataset)


def test_load_data(raw_dataset):
    assert len(raw_dataset) > 0
    assert {
        "id", "query", "tools", "answers"
    } <= set(raw_dataset.column_names)


def test_preprocess(raw_dataset, examples):
    wrapped = FunctionCallingDataset(examples)

    assert len(wrapped) == len(raw_dataset)

    row = raw_dataset[0]
    result = wrapped[0]

    assert set(result) == {"id", "query", "tools", "answers"}
    assert result["id"] == row["id"]
    assert result["query"] == row["query"]
    assert result["tools"] == json.loads(row["tools"])
    assert result["answers"] == json.loads(row["answers"])


def test_split_examples(examples):
    groups = FunctionCallingDataset._split_examples(examples)

    assert groups == FunctionCallingDataset._split_examples(examples)

    splits = {
        name: FunctionCallingDataset(items)
        for name, items in groups.items()
    }

    assert set(splits) == {
        "train", "validation", "test_seen", "test_unseen"
    }
    assert all(len(dataset) > 0 for dataset in splits.values())

    all_rows = [
        row
        for dataset in splits.values()
        for row in dataset
    ]

    assert len({row["id"] for row in all_rows}) == len(all_rows)
    assert len(all_rows) <= len(examples)

    names = {
        split: {
            name
            for row in dataset
            for name in FunctionCallingDataset._tool_names(row)
        }
        for split, dataset in splits.items()
    }

    for split in ("train", "validation", "test_seen"):
        assert names[split].isdisjoint(names["test_unseen"])

    assert names["test_seen"] <= names["train"]
    assert names["validation"] <= names["train"]

def test_loader_preserves_examples(examples):
    expected = examples[:3]
    dataset = FunctionCallingDataset(expected)

    batches = list(dataset.make_loader(batch_size=2))

    # Keep the final batch even though it contains only one example.
    assert [len(batch) for batch in batches] == [2, 1]

    # Preserve order and all nested example contents.
    actual = [example for batch in batches for example in batch]
    assert actual == expected

def test_length_grouping_preserves_examples(examples, monkeypatch):
    expected = examples[:3]
    dataset = FunctionCallingDataset(expected)

    monkeypatch.setattr(
        dataset,
        "prompt_lengths",
        lambda tokenizer: [30, 10, 20],
    )

    batches = list(dataset.make_loader(
        batch_size=2,
        tokenizer=object(),
        group_by_length=True,
    ))

    assert [len(batch) for batch in batches] == [2, 1]

    actual = [example for batch in batches for example in batch]
    assert actual == [expected[1], expected[2], expected[0]]