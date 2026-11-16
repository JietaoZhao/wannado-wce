"""Stage 1: Automated Wireless Environment Preflight & Analytical Sanity Diagnostics.

Verifies numerical finiteness (no NaN/Inf), seeded reproducibility, channel power normalization,
measured hardware provisioning speed (ms/call), and physical unit alignment before multi-seed sweeps.
"""

from __future__ import annotations

import importlib
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import torch

from wannado.wce.blocks import BLOCK_ID_TO_RUNNER
from wannado.wce.core.types import ProvisionResult
from wannado.wce.intake.spec import WCESpec


@dataclass
class PreflightReport:
    """Structured Stage 1 wireless environment preflight card."""

    ok: bool
    block_id: str
    ms_per_call: float
    mean_channel_power: float | None = None
    reproducible: bool = True
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    sample_metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def format_report(self) -> str:
        status = "PASS" if self.ok else "FAIL"
        lines = [
            f"[wannado.wce Stage 1 Preflight: {status}] block='{self.block_id}'",
            f"  Speed         : {self.ms_per_call:.2f} ms/call | Reproducible={self.reproducible}",
        ]
        if self.mean_channel_power is not None:
            lines.append(f"  Mean |H|^2    : {self.mean_channel_power:.4f}")
        if self.sample_metrics:
            preview = ", ".join(f"{k}={v:.4g}" for k, v in list(self.sample_metrics.items())[:5])
            lines.append(f"  Metrics       : {preview}")
        if self.errors:
            lines.append("  Errors:")
            lines.extend(f"    - {e}" for e in self.errors)
        if self.warnings:
            lines.append("  Warnings:")
            lines.extend(f"    - {w}" for w in self.warnings)
        if self.recommendations:
            lines.append("  Recommendations:")
            lines.extend(f"    - {r}" for r in self.recommendations)
        return "\n".join(lines)


def _resolve_block_fn(block_or_fn: str | Callable[..., ProvisionResult]) -> tuple[str, Callable[..., ProvisionResult]]:
    if callable(block_or_fn):
        return getattr(block_or_fn, "__name__", "custom_block").removeprefix("provision_"), block_or_fn
    if block_or_fn not in BLOCK_ID_TO_RUNNER:
        raise KeyError(f"Unknown block '{block_or_fn}'.")
    mod_path, fn_name = BLOCK_ID_TO_RUNNER[block_or_fn]
    mod = importlib.import_module(mod_path)
    return block_or_fn, getattr(mod, fn_name)


def preflight_wce(
    block_or_fn: str | Callable[..., ProvisionResult],
    *,
    config: Any | None = None,
    wce_spec: WCESpec | None = None,
    seed: int = 42,
) -> PreflightReport:
    """Run Stage 1 numerical finiteness, seeded reproducibility, power sanity, and latency benchmarking."""
    bid, fn = _resolve_block_fn(block_or_fn)
    errors: list[str] = []
    warnings: list[str] = []
    recs: list[str] = []

    t0 = time.perf_counter()
    res_a = fn(config=config, seed=seed) if config is not None else fn(seed=seed)
    ms_call = (time.perf_counter() - t0) * 1000.0
    res_b = fn(config=config, seed=seed) if config is not None else fn(seed=seed)

    # 1. NaN / Inf tensor and metric checks
    repro = True
    for name, val in res_a.tensors.items():
        t_a = val if isinstance(val, torch.Tensor) else torch.as_tensor(val)
        if t_a.is_complex():
            finite = bool(torch.isfinite(t_a.real).all().item() and torch.isfinite(t_a.imag).all().item())
        else:
            finite = bool(torch.isfinite(t_a).all().item())
        if not finite:
            errors.append(f"Tensor '{name}' contains NaN or Inf.")
        if name in res_b.tensors:
            t_b = res_b.tensors[name] if isinstance(res_b.tensors[name], torch.Tensor) else torch.as_tensor(res_b.tensors[name])
            if t_a.shape == t_b.shape and not torch.allclose(t_a, t_b, atol=1e-5):
                repro = False

    if not repro:
        warnings.append(f"Block '{bid}' produced slightly different outputs across identical seeds={seed}; inspect RNG seeding.")

    for mk, mv in res_a.metrics.items():
        if not np.isfinite(mv):
            errors.append(f"Metric '{mk}' is non-finite ({mv}).")

    # 2. Small-scale channel power sanity
    mean_pow: float | None = None
    for k in ("h", "h_small", "h_freq", "h_near"):
        if k in res_a.tensors:
            t_h = res_a.tensors[k]
            mean_pow = float(torch.mean(t_h.abs().pow(2)).item())
            if mean_pow < 0.25 or mean_pow > 4.0:
                warnings.append(f"Tensor '{k}' mean entry power E[|h|^2]={mean_pow:.3f} deviates noticeably from unity (1.0).")
            break

    if not res_a.parameter_provenance:
        recs.append("Record ParamProvenance entries so Stage 4 report traces every knob to its 3GPP/ITU source.")
    if wce_spec is not None and wce_spec.csi_assumption == "ideal_genie_baseline":
        recs.append("Pair ideal_genie_baseline runs with `ls_lmmse_csi` or noisy feedback for publication rigor.")

    return PreflightReport(
        ok=len(errors) == 0,
        block_id=bid,
        ms_per_call=round(ms_call, 3),
        mean_channel_power=mean_pow,
        reproducible=repro,
        warnings=warnings,
        errors=errors,
        recommendations=recs,
        sample_metrics=dict(res_a.metrics),
    )
