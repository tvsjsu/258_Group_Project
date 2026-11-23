"""Cadence, held-out validation, best selection, retention, and deterministic recovery."""

import copy
import json
import random
import shutil

import pytest
import torch
from torch.utils.data import DataLoader
from transformers import LlamaConfig, LlamaForCausalLM

from finetune_test import TinyTokenizer, model_loaders, splits, training_args
from train import finetune
from train.checkpointing import AdapterCheckpoints
from train.validation import generation_metrics, is_better, validation_due


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_epoch_cadence_rounds_to_optimizer_boundaries():
    boundaries = list(range(0, 103, 8)) + [103]
    events = [end for start, end in zip(boundaries, boundaries[1:])
              if validation_due(start, end, 103, 10)]
    assert len(events) == 10 and events[-1] == 103
    assert [end for start, end in zip(boundaries, boundaries[1:])
            if validation_due(start, end, 103, 2)] == [56, 103]
    assert not validation_due(0, 103, 103, 0)


def test_selection_ties_and_baseline_exclusion():
    best = {"complete_call_accuracy": 0.8, "val_loss": 0.1, "step": 2}
    assert not is_better({**best, "step": 3}, best)
    assert is_better({**best, "val_loss": 0.09}, best)
    assert not is_better({**best, "complete_call_accuracy": 0.7, "val_loss": 0.01}, best)
    assert is_better({**best, "complete_call_accuracy": 0.9, "val_loss": 0.2}, best)


def test_main_uses_filtered_validation_only(tmp_path, splits, model_loaders, monkeypatch):
    from train import validation
    calls = []

    def generate(model, tokenizer, examples, args, device, dtype, output, step):
        calls.append((step, [row["id"] for row in examples]))
        return {"examples": len(examples), "valid_call_rate": 1.0, "function_accuracy": 1.0,
                "argument_accuracy": 1.0, "schema_compliance": 1.0,
                "complete_call_accuracy": 1.0 if step == 0 else 0.5}

    monkeypatch.setattr(validation, "generation_metrics", generate)
    report = finetune.main(training_args(tmp_path) + ["--generation-checks-per-epoch", "2"], splits=splits)
    assert all(ids == ["validation-0", "validation-1"] for _, ids in calls)
    assert report["validation_protocol"]["examples"] == 2
    assert report["best_checkpoint"]["step"] > 0  # A perfect initial baseline is not a trained checkpoint.
    assert (tmp_path / "best" / "model.safetensors").is_file()
    assert (tmp_path / "validation_history.csv").is_file()
    assert report["early_stopping"] is False


@pytest.mark.parametrize("adapter", [False, True])
def test_four_epochs_best_retention_and_resume(tmp_path, adapter):
    peft = pytest.importorskip("peft") if adapter else None
    torch.manual_seed(71)
    base = LlamaForCausalLM(LlamaConfig(
        vocab_size=16, hidden_size=8, intermediate_size=16, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=2, attention_dropout=0.1,
        attn_implementation="eager", use_cache=False,
    ))
    original = (peft.get_peft_model(base, peft.LoraConfig(
        task_type="CAUSAL_LM", r=2, lora_alpha=4, lora_dropout=0.2,
        target_modules=["q_proj", "v_proj"],
    )) if adapter else base)
    features = [{"input_ids": [2, 3, 4, 5], "labels": [-100, -100, 4, 5]} for _ in range(8)]
    captured = tmp_path / "recovery-point"

    def run(folder, resume=None):
        folder.mkdir()
        args = finetune.parse_args([
            "--device", "cpu", "--precision", "fp32", "--epochs", "4",
            "--batch-size", "2", "--grad-accum", "2", "--lr", "0.001",
            "--save-steps", "1", "--no-gradient-checkpointing", "--output-dir", str(folder),
        ] + (["--resume-from-checkpoint", str(resume)] if resume else []))
        torch.manual_seed(19)
        model = copy.deepcopy(original)
        manager = AdapterCheckpoints(args)
        if resume:
            if adapter:
                from safetensors.torch import load_file
                peft.set_peft_model_state_dict(model, load_file(str(resume / "adapter_model.safetensors")))
            else:
                model.load_state_dict(LlamaForCausalLM.from_pretrained(resume).state_dict())
        manager.prepare({"raw_splits": {}, "used_splits": {}, "chat_template_sha256": "same",
                         "tokenizer_revision": None, "code_sha256": {}}, args, TinyTokenizer())
        manager.bind_model(model)
        save = manager.save

        def capture(*values, **keywords):
            seconds = save(*values, **keywords)
            point = folder / "checkpoint-4"
            if point.is_dir() and not captured.exists():
                shutil.copytree(point, captured)
            return seconds

        manager.save = capture
        loader = DataLoader(features, batch_size=2, shuffle=True,
                            generator=torch.Generator().manual_seed(42), collate_fn=finetune.make_collator(0))
        val_loader = DataLoader(features[:2], batch_size=2, collate_fn=finetune.make_collator(0))

        def generate(step):
            return {"examples": 2, "valid_call_rate": 1.0, "function_accuracy": 1.0,
                    "argument_accuracy": 1.0, "schema_compliance": 1.0,
                    "complete_call_accuracy": 1.0 if step == 0 else (0.9 if step == 3 else 0.2)}

        report = finetune.run_training(model, loader, val_loader, args, torch.device("cpu"), None,
                                       folder, manager, generation_evaluator=generate)
        return model, report

    model, report = run(tmp_path / "continuous")
    assert report["optimizer_steps"] == 8 and report["epochs_completed"] == 4
    assert report["best_checkpoint"]["step"] == 3
    assert report["selected_model_path"] == "best"
    assert {p.name for p in (tmp_path / "continuous").glob("checkpoint-*")} == {
        "checkpoint-3", "checkpoint-7", "checkpoint-8"}
    resumed, recovered = run(tmp_path / "resumed", captured)
    assert recovered["best_checkpoint"] == report["best_checkpoint"]
    assert recovered["validation_history"] == report["validation_history"]
    assert recovered["optimizer_steps"] == 8 and recovered["epochs_completed"] == 4
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, resumed.state_dict()[name], rtol=0, atol=0)
    from safetensors.torch import load_file
    filename = "adapter_model.safetensors" if adapter else "model.safetensors"
    selected = load_file(str(tmp_path / "continuous" / "best" / filename))
    recovered_best = load_file(str(tmp_path / "resumed" / "best" / filename))
    for name, value in selected.items():
        torch.testing.assert_close(value, recovered_best[name], rtol=0, atol=0)
    assert len((tmp_path / "resumed" / "validation_history.jsonl").read_text().splitlines()) == 9


@pytest.mark.parametrize("fail", [False, True])
def test_generation_reuses_weights_and_restores_training_state(tmp_path, monkeypatch, fail):
    from model.model import FunctionCallingModel
    model = torch.nn.Linear(2, 2)
    model.train()
    tokenizer = TinyTokenizer()
    tokenizer.padding_side = "right"
    row = {"id": 9, "query": "Echo", "tools": [{"name": "echo", "parameters": {}}],
           "answers": [{"name": "echo", "arguments": {}}]}

    def generate(wrapper, queries, tools, max_new_tokens):
        assert wrapper.model is model and not model.training and tokenizer.padding_side == "left"
        assert max_new_tokens == 512
        random.random()
        torch.rand(3)
        if fail:
            raise RuntimeError("test interruption")
        return ['[{"name":"echo","arguments":{}}]']

    monkeypatch.setattr(FunctionCallingModel, "generate_batch", generate)
    args = finetune.parse_args([])
    python_rng, torch_rng = random.getstate(), torch.get_rng_state().clone()
    if fail:
        with pytest.raises(RuntimeError, match="test interruption"):
            generation_metrics(model, tokenizer, [row], args, torch.device("cpu"), None, tmp_path, 1)
    else:
        metrics = generation_metrics(model, tokenizer, [row], args, torch.device("cpu"), None, tmp_path, 1)
        assert metrics["complete_call_accuracy"] == 1.0
        prediction = json.loads((tmp_path / metrics["predictions_file"]).read_text())
        assert prediction["id"] == 9 and prediction["split"] == "validation"
    assert model.training and tokenizer.padding_side == "right"
    assert random.getstate() == python_rng
    assert torch.equal(torch.get_rng_state(), torch_rng)


def test_full_main_real_generation_export_and_resume(tmp_path, monkeypatch):
    """Exercise real tokenizer, generation, full-weight reload and prediction recovery."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    base = tmp_path / "base"
    torch.manual_seed(12)
    LlamaForCausalLM(LlamaConfig(
        vocab_size=16, hidden_size=8, intermediate_size=16, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=2, attn_implementation="eager",
    )).save_pretrained(base)
    raw = Tokenizer(models.WordLevel({"[UNK]": 0, "[PAD]": 1, "[EOS]": 2, "word": 3}, unk_token="[UNK]"))
    raw.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=raw, unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]",
        model_input_names=["input_ids", "attention_mask"],
        chat_template="{% for m in messages %}{{m['role'] + ': ' + m['content'] + '\\n'}}{% endfor %}{% if add_generation_prompt %}{{'assistant: '}}{% endif %}",
    ).save_pretrained(base)
    rows = [{"id": i, "query": "Echo word", "tools": [{"name": "echo", "parameters": {}}],
             "answers": [{"name": "echo", "arguments": {}}]} for i in range(6)]
    splits = {"train": rows[:4], "validation": rows[4:], "test_seen": [], "test_unseen": []}
    captured = tmp_path / "captured"
    save = AdapterCheckpoints.save

    def capture(manager, *values, **keywords):
        result = save(manager, *values, **keywords)
        point = manager.output_dir / "checkpoint-4"
        if point.is_dir() and not captured.exists():
            shutil.copytree(point, captured)
        return result

    monkeypatch.setattr(AdapterCheckpoints, "save", capture)
    common = ["--model", str(base), "--device", "cpu", "--precision", "fp32",
              "--epochs", "4", "--batch-size", "2", "--grad-accum", "1",
              "--generation-batch-size", "2", "--generation-max-new-tokens", "2",
              "--save-steps", "2", "--no-gradient-checkpointing"]
    original = finetune.main(common + ["--output-dir", str(tmp_path / "original")], splits=splits)
    recovered = finetune.main(common + ["--output-dir", str(tmp_path / "recovered"),
                                      "--resume-from-checkpoint", str(captured)], splits=splits)
    assert original["epochs_completed"] == recovered["epochs_completed"] == 4
    assert original["best_checkpoint"] == recovered["best_checkpoint"]
    assert original["validation_history"] == recovered["validation_history"]
    a = LlamaForCausalLM.from_pretrained(tmp_path / "original")
    b = LlamaForCausalLM.from_pretrained(tmp_path / "recovered")
    for name, parameter in a.state_dict().items():
        torch.testing.assert_close(parameter, b.state_dict()[name], rtol=0, atol=0)
    from eval.eval import load_model
    best, _ = load_model(str(tmp_path / "recovered" / "best"), device="cpu", dtype="float32")
    assert len(best.generate_batch(["Echo"], [rows[0]["tools"]], max_new_tokens=2)) == 1
    for row in recovered["validation_history"]:
        assert (tmp_path / "recovered" / row["predictions_file"]).is_file()
