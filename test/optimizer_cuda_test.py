"""Exercise real 8-bit AdamW updates and checkpoint recovery for each training method."""

import importlib
import json
import os

import pytest
import torch


@pytest.mark.skipif(
    os.environ.get("RUN_8BIT_ADAMW_CUDA_TEST") != "1" or not torch.cuda.is_available(),
    reason="Set RUN_8BIT_ADAMW_CUDA_TEST=1 on a CUDA machine with bitsandbytes installed",
)
@pytest.mark.parametrize("method", ["finetune", "LoRA", "QLoRA"])
def test_8bit_training_and_resume(tmp_path, monkeypatch, method):
    """Train with real uint8 moment states and recover them through the shared checkpoint manager."""
    pytest.importorskip("bitsandbytes")
    pytest.importorskip("peft")
    from safetensors.torch import load_file
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
    from data.data import FunctionCallingDataset

    base, output, resumed = tmp_path / "base", tmp_path / "run", tmp_path / "resumed"
    torch.manual_seed(12)
    LlamaForCausalLM(LlamaConfig(
        vocab_size=32, hidden_size=128, intermediate_size=512, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=2048,
    )).save_pretrained(base)
    raw = Tokenizer(models.WordLevel({"[UNK]": 0, "[PAD]": 1, "[EOS]": 2, "word": 3}, unk_token="[UNK]"))
    raw.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw, unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]",
        model_input_names=["input_ids", "attention_mask"],
        chat_template="{% for m in messages %}{{m['role'] + ': ' + m['content'] + '\\n'}}{% endfor %}{% if add_generation_prompt %}{{'assistant: '}}{% endif %}",
    )
    tokenizer.save_pretrained(base)
    rows = [{"id": i, "query": "Echo word", "tools": [{"name": "echo", "parameters": {}}],
             "answers": [{"name": "echo", "arguments": {}}]} for i in range(4)]
    monkeypatch.setattr(FunctionCallingDataset, "load_splits", lambda **kwargs: {
        "train": rows[:3], "validation": rows[3:], "test_seen": [], "test_unseen": [],
    })
    common = ["--model", str(base), "--device", "cuda:0", "--precision", "fp32",
              "--batch-size", "1", "--grad-accum", "2", "--max-steps", "2",
              "--save-steps", "1", "--generation-max-new-tokens", "2"]
    main = importlib.import_module(f"train.{method}").main
    report = main(common + ["--output-dir", str(output)])
    assert report["optimizer_steps"] == 2 and report["optimizer_settings"]["name"] == "adamw8bit"
    assert report["optimizer_settings"]["state_tensor_bytes_by_dtype"]["torch.uint8"] > 0
    filename = "model.safetensors" if method == "finetune" else "adapter_model.safetensors"
    before = load_file(str(output / "checkpoint-1" / filename))
    after = load_file(str(output / "checkpoint-2" / filename))
    assert any(not torch.equal(value, after[name]) for name, value in before.items())

    recovered = main(common + ["--output-dir", str(resumed), "--resume-from-checkpoint", str(output / "checkpoint-1")])
    assert recovered["optimizer_steps"] == 2 and recovered["examples_seen"] == 3
    assert recovered["history"][0] == report["history"][0]
    assert recovered["optimizer_settings"]["state_tensor_bytes_by_dtype"] == report["optimizer_settings"]["state_tensor_bytes_by_dtype"]
    state = torch.load(resumed / "checkpoint-2" / "training_state.pt", map_location="cpu", weights_only=True)
    assert state["contract"]["settings"]["optim"] == "adamw8bit"
    saved_report = json.loads((resumed / "train_metrics.json").read_text())
    assert saved_report["optimizer_settings"] == recovered["optimizer_settings"]
    assert (resumed / "best" / filename).is_file()
