"""Check QLoRA configuration and adapter recovery with CPU models and loader mocks."""

import copy
import json
import shutil
from unittest.mock import Mock

import pytest
import torch
from torch.utils.data import DataLoader
import transformers

from train import trainer
from train.preparation import make_collator
from train.runtime import ModelContext
from train.QLoRA import build_model, parse_args
from train.checkpointing import CheckpointManager
from finetune_test import TinyTokenizer


@pytest.fixture
def peft_library():
    """Provide PEFT when installed or skip adapter-dependent checks."""
    return pytest.importorskip("peft")


@pytest.fixture(autouse=True)
def one_thread():
    """Temporarily limit CPU execution to one thread for local model checks."""
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def tiny_base():
    """Create a local tiny Llama model without downloading weights."""
    config = transformers.LlamaConfig(
        vocab_size=16, hidden_size=8, intermediate_size=16, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=2048,
        attention_dropout=0.1, attn_implementation="eager",
    )
    return transformers.LlamaForCausalLM(config)


def adapter(base, peft):
    """Attach small LoRA adapters to a local base model."""
    return peft.get_peft_model(base, peft.LoraConfig(
        task_type="CAUSAL_LM", r=2, lora_alpha=4, lora_dropout=0.2,
        target_modules=["q_proj", "v_proj"],
    ))


def test_qlora_defaults_and_invalid_options():
    """Verify QLoRA defaults and invalid options."""
    args = parse_args([])
    assert args.rank == 16 and args.lora_alpha == 32
    assert args.max_length == 2048 and args.grad_accum == 8
    assert args.gradient_checkpointing is True
    assert args.batch_size == 2 and args.grad_accum == 8
    assert args.epochs == 4 and args.limit_val is None and args.lr == 1e-4
    assert args.lr_scheduler == "cosine"
    assert args.generation_limit == 256 and args.generation_subset_seed == 42
    assert args.generation_checks_per_epoch == 1 and args.val_checks_per_epoch == 10
    assert args.generation_batch_size == 4
    assert args.output_dir == "checkpoints/qlora"
    with pytest.raises(SystemExit):
        parse_args(["--save-steps", "0"])
    with pytest.raises(ValueError):
        parse_args(["--no-save"])


def test_nf4_loading_and_only_adapter_parameters_train(peft_library, monkeypatch):
    """Check NF4 loader options and adapter-only training using a mocked model loader."""
    pytest.importorskip("bitsandbytes")
    base = tiny_base()
    base.is_loaded_in_4bit = True
    loader = Mock(return_value=base)
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", loader)
    args = parse_args(["--no-gradient-checkpointing", "--rank", "2", "--lora-alpha", "4"])
    model = build_model(args, ModelContext(torch.device("cuda:0"), torch.float32, torch.bfloat16))
    config = loader.call_args.kwargs["quantization_config"]
    assert config.load_in_4bit and config.bnb_4bit_use_double_quant
    assert config.bnb_4bit_quant_type == "nf4"
    assert config.bnb_4bit_compute_dtype == torch.bfloat16
    assert loader.call_args.kwargs["device_map"] == {"": 0}
    assert all("lora_" in name for name, parameter in model.named_parameters() if parameter.requires_grad)
    assert model.get_nb_trainable_parameters()[0] > 0


@pytest.mark.parametrize("checkpoint_step", [1, 2])
def test_resume_matches_uninterrupted_training(tmp_path, peft_library, checkpoint_step):
    """Verify resume matches uninterrupted training."""
    torch.manual_seed(17)
    original = tiny_base()
    original.save_pretrained(tmp_path / "base")
    original.name_or_path = str(tmp_path / "base")
    features = [
        {"input_ids": [2, 3, 4, 5], "labels": [-100, 3, 4, 5]},
        {"input_ids": [2, 6], "labels": [-100, 6]},
        {"input_ids": [3, 6, 7], "labels": [-100, 6, 7]},
        {"input_ids": [2, 7, 8], "labels": [-100, 7, 8]},
    ]
    manifest = {
        "raw_splits": {"train": {"ids": [0, 1, 2, 3], "sha256": "test-raw"}},
        "used_splits": {"train": {"retained_ids": [0, 1, 2, 3]}},
        "chat_template_sha256": "test-template", "tokenizer_revision": None,
        "code_sha256": {"test": "same-code"},
    }

    captured = tmp_path / "captured"

    def run(folder, resume=None):
        """Execute a small training job with optional checkpoint recovery."""
        folder.mkdir()
        options = ["--device", "cpu", "--precision", "fp32", "--epochs", "2",
                   "--grad-accum", "3", "--save-steps", "1", "--eval-every", "1",
                   "--output-dir", str(folder), "--no-gradient-checkpointing"]
        if resume:
            options += ["--resume-from-checkpoint", str(resume)]
        args = parse_args(options)
        manager = CheckpointManager(args)
        original_save = manager.save

        def save(*save_args, **save_kwargs):
            """Save a checkpoint and preserve the requested recovery point for comparison."""
            result = original_save(*save_args, **save_kwargs)
            checkpoint = folder / f"checkpoint-{checkpoint_step}"
            if checkpoint.exists() and not captured.exists():
                shutil.copytree(checkpoint, captured)
            return result

        manager.save = save
        manager.prepare(manifest, args, TinyTokenizer())
        torch.manual_seed(23)
        base = copy.deepcopy(original)
        model = (peft_library.PeftModel.from_pretrained(base, resume, is_trainable=True)
                 if resume else adapter(base, peft_library))
        model.config.use_cache = False
        model.train()
        manager.bind_model(model)
        loader = DataLoader(features, batch_size=1, shuffle=True,
                            generator=torch.Generator().manual_seed(42), collate_fn=make_collator(0))
        validation = DataLoader(features[:2], batch_size=1, collate_fn=make_collator(0))
        report = trainer.run_training(model, loader, validation, args, torch.device("cpu"), None, folder, manager)
        return model, report, manager

    continuous, metrics, _ = run(tmp_path / "continuous")
    checkpoint = captured
    assert (checkpoint / "training_state.pt").is_file()
    assert (checkpoint / "adapter_model.safetensors").is_file()
    resumed, resumed_metrics, manager = run(tmp_path / "resumed", checkpoint)
    def deterministic_history(report):
        """Exclude elapsed times when comparing training histories."""
        return [{key: value for key, value in row.items() if key != "step_seconds"} for row in report["history"]]
    assert deterministic_history(resumed_metrics) == deterministic_history(metrics)
    assert resumed_metrics["examples_seen"] == metrics["examples_seen"] == 8
    assert resumed_metrics["epochs_completed"] == metrics["epochs_completed"] == 2
    for key, value in continuous.state_dict().items():
        torch.testing.assert_close(value, resumed.state_dict()[key], rtol=0, atol=0)
    altered = copy.deepcopy(manifest)
    altered["used_splits"]["train"]["retained_ids"] = [0, 1]
    with pytest.raises(ValueError, match="differs"):
        manager.prepare(altered, manager.args, TinyTokenizer())


def test_saved_adapter_can_be_loaded_by_shared_evaluator(tmp_path, peft_library, monkeypatch):
    """Verify saved adapter can be loaded by shared evaluator."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from eval.eval import load_model

    base_dir, adapter_dir = tmp_path / "base", tmp_path / "adapter"
    raw = Tokenizer(models.WordLevel({"[UNK]": 0, "[PAD]": 1, "[EOS]": 2, "word": 3}, unk_token="[UNK]"))
    raw.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=raw, unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]",
        model_input_names=["input_ids", "attention_mask"],
        chat_template="{% for m in messages %}{{m['role'] + ': ' + m['content'] + '\\n'}}{% endfor %}{% if add_generation_prompt %}{{'assistant: '}}{% endif %}",
    )
    tiny_base().save_pretrained(base_dir)
    tokenizer.save_pretrained(base_dir)
    model = adapter(transformers.AutoModelForCausalLM.from_pretrained(base_dir), peft_library)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    loaded, name = load_model(adapter=str(adapter_dir), device="cpu", dtype="float32")
    assert name == str(base_dir)
    assert loaded.model.peft_config
    assert not loaded.model.training
    reduced_precision, _ = load_model(adapter=str(adapter_dir), device="cpu", dtype="bfloat16")
    assert reduced_precision.adapter_parameter_dtypes == ["torch.bfloat16"]
    assert len(loaded.generate_batch(["Echo word"], [[{"name": "echo", "parameters": {}}]], max_new_tokens=2)) == 1
    from data.data import FunctionCallingDataset
    from eval.eval import main as evaluate
    rows = [{"id": 1, "query": "Echo word", "tools": [{"name": "echo", "parameters": {}}],
             "answers": [{"name": "echo", "arguments": {}}]}]
    monkeypatch.setattr(FunctionCallingDataset, "load_splits", lambda: {"test_seen": rows, "test_unseen": rows})
    output = tmp_path / "predictions.jsonl"
    evaluate(["--adapter", str(adapter_dir), "--split", "both", "--device", "cpu",
              "--dtype", "float32", "--max-new-tokens", "2", "--output", str(output)])
    report = json.loads(output.with_suffix(".metrics.json").read_text())
    assert report["settings"]["model"] == str(base_dir)
    assert report["settings"]["adapter"] == str(adapter_dir)
    assert report["settings"]["base_quantization"] == "unquantized"
    assert report["combined"]["examples"] == 2
    assert "complete_call_accuracy" in report["combined"]


def test_reference_data_mismatch_is_rejected(tmp_path):
    """Verify reference data mismatch is rejected."""
    manifest = {
        "raw_splits": {"train": {"ids": [1, 2], "sha256": "content"}},
        "used_splits": {"train": {"retained_ids": [1, 2]}},
        "chat_template_sha256": "template", "tokenizer_revision": None, "code_sha256": {},
    }
    reference = tmp_path / "data_splits.json"
    reference.write_text(json.dumps(manifest))
    args = parse_args(["--reference-data", str(reference)])
    manager = CheckpointManager(args)
    manager.prepare(manifest, args, TinyTokenizer())
    changed = copy.deepcopy(manifest)
    changed["used_splits"]["train"]["retained_ids"] = [2, 1]
    with pytest.raises(ValueError, match="reference"):
        manager.prepare(changed, args, TinyTokenizer())
