"""Resolve device and precision settings shared by model builders and training."""

from contextlib import nullcontext
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ModelContext:
    """Provide immutable device, precision, and revision settings to model builders."""

    device: torch.device
    weight_dtype: torch.dtype
    compute_dtype: torch.dtype
    base_revision: str | None = None


def pick_device(requested):
    """Resolve the requested device or select CUDA, MPS, then CPU."""
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_precision(precision, device):
    """Return (weight dtype, autocast dtype or None)."""
    if precision == "auto":
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            precision = "bf16-mixed"
        else:
            precision = "fp32"
    return {
        "fp32": (torch.float32, None),
        "bf16-mixed": (torch.float32, torch.bfloat16),
        "bf16": (torch.bfloat16, None),
    }[precision], precision


def autocast(device, dtype):
    """Create an autocast context when mixed-precision computation is enabled."""
    if dtype is None:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def synchronize(device):
    """Wait for queued accelerator operations before measuring elapsed time."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def require_cuda(device, precision):
    """Resolve a concrete CUDA device for methods that require NVIDIA hardware."""
    resolved = pick_device(device)
    if resolved.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("QLoRA requires an NVIDIA CUDA GPU. Install dependencies with: uv sync --extra qlora")
    resolved = torch.device("cuda", resolved.index if resolved.index is not None else torch.cuda.current_device())
    torch.cuda.set_device(resolved)
    if precision.startswith("bf16") and not torch.cuda.is_bf16_supported():
        raise ValueError("This GPU does not support BF16; use --precision fp32")
    return resolved
