"""Check training budgets, loss weighting, data identity, and report artifacts."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from safetensors.torch import save_file
import transformers

from train.preparation import (
    IGNORE_INDEX,
    describe_splits,
    make_collator,
    prepare_split,
)
from train.finetune import main
from train.config import parse_args
from train.trainer import normalize_gradients, training_plan


class TinyTokenizer:
    """Provide a deterministic tokenizer substitute for local training checks."""

    pad_token_id = 0
    eos_token = "<eos>"
    chat_template = "test-only role/content template"
    init_kwargs = {"_commit_hash": "test-tokenizer-revision"}

    def apply_chat_template(self, messages, *, tokenize=False, add_generation_prompt=False):
        """Render role and content fields with an optional assistant generation prefix."""
        assert tokenize is False
        text = "\n".join(f"{message['role']}: {message['content']}" for message in messages)
        return text + ("\nassistant: " if add_generation_prompt else "")

    def __call__(self, text, *, add_special_tokens=False):
        """Encode text deterministically without adding special tokens."""
        assert add_special_tokens is False
        return {"input_ids": [2 + sum(word.encode("utf-8")) % 14 for word in text.split()]}

    def save_pretrained(self, output_dir):
        """Write the minimal model or tokenizer artifacts required by the test."""
        path = Path(output_dir, "tokenizer_config.json")
        path.write_text(json.dumps({"chat_template": self.chat_template}), encoding="utf-8")
        return (str(path),)


class TinyModel(torch.nn.Module):
    """Provide a small trainable model with observable batch sizes for training checks."""

    def __init__(self):
        """Initialize the local test double and its recorded state."""
        super().__init__()
        self.embedding = torch.nn.Embedding(16, 4)
        self.projection = torch.nn.Linear(4, 16)
        self.config = SimpleNamespace(use_cache=True, _commit_hash="test-model-revision")
        self.training_batch_sizes = []

    def forward(self, input_ids, labels, attention_mask):
        """Calculate causal response loss and record training batch sizes."""
        if self.training:
            self.training_batch_sizes.append(len(input_ids))
        logits = self.projection(self.embedding(input_ids))
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, 16), labels[:, 1:].reshape(-1), ignore_index=IGNORE_INDEX
        )
        return SimpleNamespace(loss=loss)

    def save_pretrained(self, output_dir, *, safe_serialization=True):
        """Write the minimal model or tokenizer artifacts required by the test."""
        assert safe_serialization is True
        save_file(self.state_dict(), str(Path(output_dir, "model.safetensors")))
        Path(output_dir, "config.json").write_text(json.dumps(vars(self.config)), encoding="utf-8")


@pytest.fixture(scope="module", autouse=True)
def one_cpu_thread():
    """Temporarily limit CPU execution to one thread for repeatable lightweight checks."""
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def splits():
    """Provide controlled train, validation, and test examples with length-filter cases."""
    def row(name, index, *, long=False):
        """Construct a labeled example with an optional overlength query."""
        return {
            "id": f"{name}-{index}",
            "query": "long " * 4096 if long else f"Echo value {index}.",
            "tools": [{"name": "echo", "parameters": {"value": {"type": "str", "required": True}}}],
            "answers": [{"name": "echo", "arguments": {"value": f"value {index}"}}],
        }
    return {
        "train": [row("train", i) for i in range(4)] + [row("train", 4, long=True)],
        "validation": [row("validation", i) for i in range(2)] + [row("validation", 2, long=True)],
        "test_seen": [row("test_seen", 0)],
        "test_unseen": [row("test_unseen", 0)],
    }


@pytest.fixture
def model_loaders(monkeypatch):
    """Replace pretrained loaders with local tokenizer and model test doubles."""
    models = []

    def load_model(*args, **kwargs):
        """Create and record a small model for the mocked pretrained loader."""
        model = TinyModel()
        models.append(model)
        return model

    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", load_model)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: TinyTokenizer())
    return models


def training_args(output_dir):
    """Build a small CPU training configuration for an isolated output directory."""
    return [
        "--device", "cpu", "--precision", "fp32", "--batch-size", "1",
        "--grad-accum", "3", "--epochs", "1", "--limit-val", "all",
        "--max-length", "1024", "--output-dir", str(output_dir),
        "--generation-checks-per-epoch", "0", "--no-gradient-checkpointing",
    ]


@pytest.mark.parametrize("epochs,max_steps,limits,steps", [
    (1, None, [10], 2),
    (2, None, [10, 10], 4),
    (1.5, None, [10, 5], 3),
    (0.5, None, [5], 1),
    (1, 3, [10, 10], 3),
])
def test_training_budget(epochs, max_steps, limits, steps):
    """Verify training budget."""
    assert training_plan(10, 8, epochs, max_steps) == (limits, steps)


def test_response_token_weighting_matches_one_combined_batch():
    """Verify response token weighting matches one combined batch."""
    torch.manual_seed(7)
    combined_model = TinyModel()
    accumulated_model = copy.deepcopy(combined_model)
    features = [
        {"input_ids": [2, 3, 4, 5], "labels": [IGNORE_INDEX, 3, 4, 5]},
        {"input_ids": [2, 6], "labels": [IGNORE_INDEX, 6]},
    ]
    collate = make_collator(0)
    combined_model(**collate(features)).loss.backward()
    response_tokens = 0
    for feature in features:
        batch = collate([feature])
        tokens = int((batch["labels"][:, 1:] != IGNORE_INDEX).sum())
        (accumulated_model(**batch).loss * tokens).backward()
        response_tokens += tokens
    normalize_gradients(accumulated_model, response_tokens)
    for combined, accumulated in zip(combined_model.parameters(), accumulated_model.parameters()):
        torch.testing.assert_close(combined.grad, accumulated.grad)


def test_filtering_records_ids_and_masks_prompt(splits):
    """Verify filtering records ids and masks prompt."""
    features, details = prepare_split(TinyTokenizer(), splits["train"], 1024)
    assert len(features) == 4
    assert details["selected_count"] == 5
    assert details["retained_ids"] == [f"train-{i}" for i in range(4)]
    assert details["dropped_ids"] == ["train-4"]
    assert details["response_tokens"] > 0
    assert details["sequence_tokens"] > details["response_tokens"]
    assert features[0]["labels"][0] == IGNORE_INDEX
    assert features[0]["labels"][-1] != IGNORE_INDEX


def test_split_fingerprint_detects_changed_content_with_same_ids(splits):
    """Verify split fingerprint detects changed content with same ids."""
    original = describe_splits(splits)
    changed = copy.deepcopy(splits)
    changed["train"][0]["query"] = "Changed query"
    current = describe_splits(changed)
    assert original["train"]["ids"] == current["train"]["ids"]
    assert original["train"]["sha256"] != current["train"]["sha256"]


def test_shared_baseline_splits_and_report_artifacts(tmp_path, splits, model_loaders, monkeypatch):
    """Verify shared baseline splits and report artifacts."""
    from data.data import FunctionCallingDataset

    load_splits = Mock(return_value=splits)
    monkeypatch.setattr(FunctionCallingDataset, "load_splits", load_splits)
    report = main(training_args(tmp_path) + ["--eval-every", "1"])
    load_splits.assert_called_once_with(seed=42, unseen_fraction=0.2)
    saved = json.loads((tmp_path / "train_metrics.json").read_text())
    manifest = json.loads((tmp_path / "data_splits.json").read_text())
    history = [json.loads(line) for line in (tmp_path / "training_history.jsonl").read_text().splitlines()]
    assert saved == report
    assert history == report["history"]
    assert sum(model_loaders[0].training_batch_sizes) == 4
    assert report["optimizer_steps"] == 2
    assert report["micro_batches"] == report["examples_seen"] == 4
    assert report["epochs_completed"] == 1
    assert [record["examples"] for record in history] == [3, 1]
    assert report["train_examples_selected"] == 5
    assert report["train_examples"] == 4
    assert report["train_examples_dropped"] == 1
    assert report["validation_examples_selected"] == 3
    assert report["validation_examples"] == 2
    assert report["validation_examples_dropped"] == 1
    assert report["response_tokens_seen"] == manifest["used_splits"]["train"]["response_tokens"]
    assert report["tokens_seen"] > report["response_tokens_seen"]
    assert report["trainable_parameters"] == report["total_parameters"] > 0
    assert report["trainable_parameter_percentage"] == 100
    assert report["environment"]["device"] == "cpu"
    assert report["environment"]["packages"]["torch"]
    assert report["val_loss_before"] is not None and report["val_loss_after"] is not None
    assert report["validation_seconds"]["during_training"] > 0
    assert report["train_seconds"] == pytest.approx(
        report["training_loop_seconds"] - report["validation_seconds"]["during_training"]
        - report["periodic_checkpoint_seconds"]
    )
    assert manifest["settings"]["split_seed"] == 42
    assert manifest["raw_splits"]["test_seen"]["ids"] == ["test_seen-0"]
    assert manifest["raw_splits"]["test_unseen"]["ids"] == ["test_unseen-0"]
    assert manifest["used_splits"]["train"]["retained_ids"] == [f"train-{i}" for i in range(4)]
    assert report["model_weights_bytes"] == (tmp_path / "model.safetensors").stat().st_size
    checkpoint_bytes = sum((tmp_path / name).stat().st_size for name in (
        "model.safetensors", "config.json", "tokenizer_config.json"
    ))
    assert report["checkpoint_bytes"] == checkpoint_bytes


def test_no_save_still_writes_data_and_training_metrics(tmp_path, splits, model_loaders):
    """Verify no save still writes data and training metrics."""
    report = main(training_args(tmp_path) + ["--limit-val", "0", "--no-save"], splits=splits)
    assert report["checkpoint_bytes"] is None and report["model_weights_bytes"] is None
    assert report["validation_examples"] == 0
    assert report["val_loss_before"] is None and report["val_loss_after"] is None
    for name in ("train_metrics.json", "training_history.jsonl", "data_splits.json"):
        assert (tmp_path / name).is_file()
    assert not (tmp_path / "model.safetensors").exists()


def test_all_validation_option():
    """Verify all validation option."""
    assert parse_args(["--limit-val", "all"]).limit_val is None
    assert parse_args([]).limit_val is None
    with pytest.raises(SystemExit):
        parse_args(["--limit-val", "-1"])
