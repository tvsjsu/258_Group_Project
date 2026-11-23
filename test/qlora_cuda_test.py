"""Opt-in real NF4 training smoke test; no model or dataset downloads required."""

import os
import pytest
import torch


@pytest.mark.skipif(
    os.environ.get("RUN_QLORA_CUDA_TEST") != "1" or not torch.cuda.is_available(),
    reason="Set RUN_QLORA_CUDA_TEST=1 on a CUDA machine with the qlora extra installed",
)
def test_real_nf4_training_and_adapter_reload(tmp_path, monkeypatch):
    pytest.importorskip("peft")
    pytest.importorskip("bitsandbytes")
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
    from data.data import FunctionCallingDataset
    from train.QLoRA import main
    from eval.eval import load_model

    base_dir, output_dir = tmp_path / "base", tmp_path / "run"
    config = LlamaConfig(vocab_size=16, hidden_size=64, intermediate_size=128,
                         num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                         max_position_embeddings=2048)
    LlamaForCausalLM(config).save_pretrained(base_dir)
    raw = Tokenizer(models.WordLevel({"[UNK]": 0, "[PAD]": 1, "[EOS]": 2, "word": 3}, unk_token="[UNK]"))
    raw.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw, unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]",
        model_input_names=["input_ids", "attention_mask"],
        chat_template="{% for m in messages %}{{m['role'] + ': ' + m['content'] + '\\n'}}{% endfor %}{% if add_generation_prompt %}{{'assistant: '}}{% endif %}",
    )
    tokenizer.save_pretrained(base_dir)
    rows = [{"id": i, "query": "Echo word", "tools": [{"name": "echo", "parameters": {}}],
             "answers": [{"name": "echo", "arguments": {}}]} for i in range(4)]
    monkeypatch.setattr(FunctionCallingDataset, "load_splits", lambda **kwargs: {
        "train": rows[:2], "validation": rows[2:3], "test_seen": [], "test_unseen": rows[3:],
    })
    report = main(["--model", str(base_dir), "--output-dir", str(output_dir),
                   "--grad-accum", "1", "--max-steps", "2", "--save-steps", "1", "--limit-val", "all",
                   "--generation-max-new-tokens", "8"])
    assert report["method"] == "qlora" and report["optimizer_steps"] == 2
    assert report["quantization"]["bits"] == 4
    assert 0 < report["trainable_parameters"] < report["total_parameters"]
    assert (output_dir / "checkpoint-1" / "training_state.pt").is_file()
    assert (output_dir / "checkpoint-2" / "training_state.pt").is_file()
    assert (output_dir / "best" / "adapter_model.safetensors").is_file()
    assert report["best_checkpoint"]["step"] in (1, 2)
    assert len(report["validation_history"]) == 3
    dtype = "bfloat16" if torch.cuda.is_bf16_supported() else "float32"
    loaded, _ = load_model(adapter=str(output_dir), device="cuda:0", dtype=dtype, load_in_4bit=True)
    assert loaded.model.is_loaded_in_4bit
    assert len(loaded.generate_batch(["Echo word"], [rows[0]["tools"]], max_new_tokens=2)) == 1
