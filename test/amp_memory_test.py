"""Check mixed-precision cache policy and checkpointed gradient equivalence."""

import copy

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from train.runtime import autocast


def test_uncached_autocast_preserves_checkpointed_loss_and_gradients():
    """Compare real BF16 checkpointed training with and without weight caching."""
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(23)
        original = LlamaForCausalLM(LlamaConfig(
            vocab_size=32, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
            num_attention_heads=2, num_key_value_heads=2, use_cache=False,
        ))
        original.gradient_checkpointing_enable()
        uncached = copy.deepcopy(original)
        batch = {"input_ids": torch.tensor([[2, 3, 4, 5, 6]]),
                 "labels": torch.tensor([[-100, -100, 4, 5, 6]])}
        with torch.autocast("cpu", dtype=torch.bfloat16, cache_enabled=True):
            baseline_loss = original(**batch).loss
        baseline_loss.backward()
        flags = []
        uncached.model.layers[0].register_forward_pre_hook(
            lambda *args: flags.append(torch.is_autocast_cache_enabled())
        )
        previous_cache = torch.is_autocast_cache_enabled()
        with autocast(torch.device("cpu"), torch.bfloat16):
            assert not torch.is_autocast_cache_enabled()
            loss = uncached(**batch).loss
        loss.backward()
        assert torch.is_autocast_cache_enabled() == previous_cache
        assert len(flags) == 2 and not any(flags)
        torch.testing.assert_close(loss, baseline_loss, rtol=0, atol=0)
        for parameter, baseline in zip(uncached.parameters(), original.parameters()):
            assert parameter.dtype == torch.float32 and parameter.grad is not None
            torch.testing.assert_close(parameter.grad, baseline.grad, rtol=0, atol=0)
        with autocast(torch.device("cpu"), None):
            assert not torch.is_autocast_enabled("cpu")
    finally:
        torch.set_num_threads(previous_threads)
