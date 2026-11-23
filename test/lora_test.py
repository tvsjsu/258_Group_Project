"""One shared protocol and actual adapter training without quantization or downloads."""

import json
import shutil

import pytest
import torch
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

from finetune_test import TinyTokenizer
from train.config import parse_args as full_args
from train.LoRA import main, parse_args
from train.QLoRA import parse_args as qlora_args


def test_three_final_recipes_share_monitoring_and_effective_batch():
    full, lora, qlora = full_args([]), parse_args([]), qlora_args([])
    assert [args.lr for args in (full, lora, qlora)] == [5e-6, 1e-4, 1e-4]
    for args in (full, lora, qlora):
        assert args.epochs == 4 and args.batch_size * args.grad_accum == 16
        assert args.limit_train is None and args.limit_val is None
        assert args.generation_limit == 256 and args.generation_subset_seed == 42
        assert args.val_checks_per_epoch == 10 and args.generation_checks_per_epoch == 1
        assert args.generation_batch_size == 4 and args.generation_max_new_tokens == 512
        assert args.lr_scheduler == "cosine" and args.gradient_checkpointing
    for args in (lora, qlora):
        assert args.rank == 16 and args.lora_alpha == 32 and args.lora_dropout == 0.05
        assert args.target_modules == ["all-linear"]


def test_native_lora_trains_only_adapters_with_shared_checkpoint_selection(tmp_path, monkeypatch):
    pytest.importorskip("peft")
    from data.data import FunctionCallingDataset
    from train import validation
    from train.checkpointing import CheckpointManager
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        base = tmp_path / "base"
        LlamaForCausalLM(LlamaConfig(
            vocab_size=16, hidden_size=8, intermediate_size=16, num_hidden_layers=1,
            num_attention_heads=2, num_key_value_heads=2,
        )).save_pretrained(base)
        rows = [{"id": i, "query": "Echo", "tools": [{"name": "echo", "parameters": {}}],
                 "answers": [{"name": "echo", "arguments": {}}]} for i in range(6)]
        splits = {"train": rows[:4], "validation": rows[4:], "test_seen": [], "test_unseen": []}
        monkeypatch.setattr(FunctionCallingDataset, "load_splits", lambda **kwargs: splits)
        monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *args, **kwargs: TinyTokenizer())
        calls = []

        def generate(model, tokenizer, examples, args, device, dtype, output, step):
            calls.append((step, [row["id"] for row in examples]))
            return {"examples": len(examples), "complete_call_accuracy": 1.0 if step in (0, 2) else 0.0}

        monkeypatch.setattr(validation, "generation_metrics", generate)
        captured = tmp_path / "recovery-point"
        save = CheckpointManager.save

        def capture(manager, *values, **keywords):
            result = save(manager, *values, **keywords)
            point = manager.output_dir / "checkpoint-4"
            if point.is_dir() and not captured.exists():
                shutil.copytree(point, captured)
            return result

        monkeypatch.setattr(CheckpointManager, "save", capture)
        out = tmp_path / "run"
        common = ["--model", str(base), "--device", "cpu", "--precision", "fp32",
                  "--batch-size", "1", "--grad-accum", "2", "--rank", "2", "--lora-alpha", "4"]
        report = main(common + ["--output-dir", str(out)])
        assert report["method"] == "lora" and report["quantization"] is None
        assert report["epochs_completed"] == 4 and report["optimizer_steps"] == 8
        assert [step for step, _ in calls] == [0, 2, 4, 6, 8]
        assert all(ids == [4, 5] for _, ids in calls)
        assert report["best_checkpoint"]["step"] == 2
        assert report["trainable_parameter_names"]
        assert all("lora_" in name for name in report["trainable_parameter_names"])
        assert report["trainable_parameters"] < report["total_parameters"]
        assert report["adapter_parameter_dtypes"] == ["torch.float32"]
        assert (out / "best" / "adapter_model.safetensors").is_file()
        manifest = json.loads((out / "data_splits.json").read_text())
        assert manifest["generation_validation"]["ids"] == [4, 5]
        assert "train/LoRA.py" in manifest["code_sha256"]
        assert "train/QLoRA.py" not in manifest["code_sha256"]
        resumed_dir = tmp_path / "resumed"
        recovered = main(common + ["--output-dir", str(resumed_dir), "--resume-from-checkpoint", str(captured)])
        assert recovered["best_checkpoint"] == report["best_checkpoint"]
        assert recovered["validation_history"] == report["validation_history"]
        from safetensors.torch import load_file
        a = load_file(str(out / "adapter_model.safetensors"))
        b = load_file(str(resumed_dir / "adapter_model.safetensors"))
        for name, parameter in a.items():
            torch.testing.assert_close(parameter, b[name], rtol=0, atol=0)
    finally:
        torch.set_num_threads(previous_threads)
