"""Check shared optimizer selection, state reporting, and recovery serialization."""

import io

import pytest
import torch

from train import trainer
from train.config import parse_args
from train.reporting import optimizer_report


def test_adamw_excludes_frozen_parameters_and_updates_trainable_weights():
    """Keep frozen weights out of optimizer groups and update the remaining weights."""
    model = torch.nn.Linear(4, 2)
    model.bias.requires_grad_(False)
    before = model.weight.detach().clone()
    optimizer = trainer.build_optimizer("adamw", model, 0.001, 0.02)
    assert optimizer.param_groups[0]["params"] == [model.weight]
    model(torch.ones(1, 4)).sum().backward()
    optimizer.step()
    assert not torch.equal(before, model.weight)
    report = optimizer_report(optimizer, "adamw")
    assert report["betas"] == [0.9, 0.999] and report["weight_decay"] == 0.02
    assert report["state_tensor_bytes_by_dtype"]["torch.float32"] > 0


def test_8bit_requires_cuda_before_loading_data(tmp_path, monkeypatch):
    """Fail early on unsupported devices rather than silently selecting another optimizer."""
    def forbidden(*args, **kwargs):
        """Fail if preprocessing or model loading starts before device validation."""
        pytest.fail("Data or model loading should not run")

    monkeypatch.setattr(trainer, "prepare_data", forbidden)
    args = parse_args(["--device", "cpu", "--output-dir", str(tmp_path / "run")])
    with pytest.raises(ValueError, match="adamw8bit requires CUDA"):
        trainer.run_experiment(args, forbidden, method="full")
    assert not (tmp_path / "run").exists()
    with pytest.raises(ValueError, match="adamw8bit requires CUDA"):
        trainer.build_optimizer("adamw8bit", torch.nn.Linear(2, 2), 0.001, 0.0)
    with pytest.raises(ValueError, match="Unknown optimizer"):
        trainer.build_optimizer("typo", torch.nn.Linear(2, 2), 0.001, 0.0)


def test_8bit_constructor_and_uint8_state_reload(monkeypatch):
    """Check real bitsandbytes construction and serialization without running CUDA kernels."""
    pytest.importorskip("bitsandbytes")
    from bitsandbytes.optim import AdamW8bit

    monkeypatch.setattr(trainer, "validate_optimizer_device", lambda *args: None)
    model = torch.nn.Linear(128, 64)
    model.bias.requires_grad_(False)
    optimizer = trainer.build_optimizer("adamw8bit", model, 0.0001, 0.0)
    assert isinstance(optimizer, AdamW8bit)
    assert optimizer.param_groups[0]["params"] == [model.weight]
    assert optimizer.args.optim_bits == 8 and optimizer.args.min_8bit_size == 4096
    assert optimizer.args.percentile_clipping == 100 and optimizer.args.block_wise
    assert not optimizer.is_paged

    optimizer.state[model.weight] = {
        "step": 2,
        "state1": torch.arange(model.weight.numel(), dtype=torch.int64).to(torch.uint8).reshape_as(model.weight),
        "state2": torch.ones_like(model.weight, dtype=torch.uint8),
        "absmax1": torch.ones(32, dtype=torch.float32),
        "absmax2": torch.ones(32, dtype=torch.float32),
    }
    stream = io.BytesIO()
    torch.save({"optimizer": optimizer.state_dict()}, stream)
    stream.seek(0)
    saved = torch.load(stream, map_location="cpu", weights_only=True)
    restored = trainer.build_optimizer("adamw8bit", model, 0.0001, 0.0)
    restored.load_state_dict(saved["optimizer"])
    for key, value in optimizer.state[model.weight].items():
        recovered = restored.state[model.weight][key]
        if isinstance(value, torch.Tensor):
            assert recovered.dtype == value.dtype
            torch.testing.assert_close(recovered, value, rtol=0, atol=0)
        else:
            assert recovered == value
    report = optimizer_report(restored, "adamw8bit")
    assert report["name"] == "adamw8bit" and report["configured_state_bits"] == 8
    assert report["implementation"].endswith(".AdamW8bit")
    assert report["min_8bit_size"] == 4096 and report["small_tensor_state_bits"] == 32
    assert report["state_tensor_bytes_by_dtype"] == {
        "torch.uint8": model.weight.numel() * 2, "torch.float32": 32 * 4 * 2,
    }
    assert report["state_tensor_bytes"] == sum(report["state_tensor_bytes_by_dtype"].values())
