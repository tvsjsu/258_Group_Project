import json

import pytest

from data.data import load_data, preprocess, split_examples, tool_names


@pytest.fixture(scope="module")
def dataset():
    return load_data()


@pytest.fixture(scope="module")
def examples(dataset):
    return preprocess(dataset)


def test_load_data(dataset):
    assert len(dataset) > 0
    assert {"id", "query", "tools", "answers"} <= set(dataset.column_names)


def test_preprocess(dataset, examples):
    assert len(examples) == len(dataset)
    row = dataset[0]
    result = examples[0]
    assert set(result) == {"id", "query", "tools", "answers"}
    assert result["id"] == row["id"]
    assert result["query"] == row["query"]
    assert result["tools"] == json.loads(row["tools"])
    assert result["answers"] == json.loads(row["answers"])


def test_split_examples(examples):
    splits = split_examples(examples)
    assert splits == split_examples(examples)
    all_rows = [row for group in splits.values() for row in group]
    assert len({row["id"] for row in all_rows}) == len(all_rows)
    assert len(all_rows) <= len(examples)

    names = {split: {name for row in group for name in tool_names(row)}
             for split, group in splits.items()}
    for split in ("train", "validation", "test_seen"):
        assert names[split].isdisjoint(names["test_unseen"])
    assert names["test_seen"] <= names["train"]
    assert names["validation"] <= names["train"]
    assert all(splits.values())
