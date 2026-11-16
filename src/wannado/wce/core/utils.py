"""Shared reproducibility, device, tensor, and statistical aggregation utilities."""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch


def set_global_seed(seed: int) -> np.random.Generator:
    """Seed Python, NumPy, and PyTorch RNGs deterministically."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return np.random.default_rng(seed)


def resolve_device(device: str | torch.device = "auto", *, allow_mps: bool = False) -> torch.device:
    """Resolve 'auto' to CUDA -> CPU (avoiding MPS complex-op gaps unless allow_mps=True)."""
    if isinstance(device, torch.device):
        return device
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if allow_mps and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def make_torch_generator(seed: int, device: torch.device | str = "cpu") -> torch.Generator:
    """Create a deterministic PyTorch Generator on CPU (or CUDA if supported)."""
    dev = resolve_device(device)
    if dev.type == "cuda":
        gen = torch.Generator(device="cuda")
    else:
        gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    return gen


def complex_normal(
    shape: tuple[int, ...],
    *,
    variance: float = 1.0,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.complex64,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample circularly symmetric complex Gaussian tensor CN(0, variance)."""
    real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
    dev = resolve_device(device)
    std_each = float(np.sqrt(max(variance, 0.0) / 2.0))
    re = torch.randn(shape, dtype=real_dtype, device=dev, generator=generator) * std_each
    im = torch.randn(shape, dtype=real_dtype, device=dev, generator=generator) * std_each
    return torch.complex(re, im).to(dtype)


def compute_iqm(values: Sequence[float]) -> float:
    """Compute Interquartile Mean (IQM) across seeds/trials per Agarwal et al. (NeurIPS 2021)."""
    arr = np.sort(np.asarray(values, dtype=np.float64))
    n = len(arr)
    if n == 0:
        return 0.0
    if n <= 2:
        return float(np.mean(arr))
    low = 0.25 * n
    high = 0.75 * n
    total = 0.0
    weight_sum = 0.0
    for idx, val in enumerate(arr):
        left = max(float(idx), low)
        right = min(float(idx + 1), high)
        if right > left:
            w = right - left
            total += w * float(val)
            weight_sum += w
    return float(total / max(weight_sum, 1e-12))


def aggregate_seed_stats(values: Sequence[float]) -> dict[str, float]:
    """Compute multi-seed mean, std, median, IQM, min, and max."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "median": 0.0, "iqm": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=0)),
        "median": float(np.median(arr)),
        "iqm": compute_iqm(arr.tolist()),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def to_python_scalar(val: Any) -> Any:
    """Convert NumPy/PyTorch scalar to JSON-safe Python primitive."""
    if isinstance(val, (np.floating, np.integer)):
        return val.item()
    if isinstance(val, torch.Tensor) and val.numel() == 1:
        return val.item()
    return val
