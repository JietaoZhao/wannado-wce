"""KKT Water-Filling Power Allocation over Parallel MIMO Eigenmodes / Subchannels.

[Lineage & Intuition]
- Family / Era / Tier : baselines / classic / S (Optimal Parallel-Channel Power Baseline)
- Parent              : `awgn_link`
- Key Diff            : Computes exact water-level mu via bisection solving sum_i max(0, mu - 1/lambda_i) = P_tot
                        and benchmarks capacity gain over naive Equal Power Allocation.
- Extension Hook [AI-Prompt Hook]:
    1. Extend single-user SVD water-filling to multi-user interfering precoding -> `wmmse_precoder`.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch

from wannado.wce.core.provenance import MECHANISM_PROVENANCE
from wannado.wce.core.types import ParamProvenance, ProvisionResult, WCEMeta
from wannado.wce.core.units import db_to_linear
from wannado.wce.core.utils import complex_normal, resolve_device, set_global_seed

META = WCEMeta(
    id="water_filling",
    name="Optimal KKT Water-Filling Power Allocation Baseline",
    family="baselines",
    era="classic",
    tier="S",
    output_kind="baseline_policy_solution",
    supports_torch_complex=True,
    supports_autograd=False,
    standard_ids=(),
    parent_id="awgn_link",
    diff_from_parent="Solves exact KKT water-filling over parallel channel eigenmodes vs uniform power.",
    summary_plain="Optimal MIMO/OFDM water-filling power allocation baseline guaranteeing no DRL/heuristic claims trivial wins.",
    when_to_use="Baseline power allocation benchmark over parallel OFDM subcarriers or SVD MIMO modes.",
    when_not_to_use="Multi-user co-channel interference where WMMSE is needed (`wmmse_precoder`).",
    hyperparam_intuition={
        "snr_db": "Total power-to-noise budget P_tot / sigma^2 in dB; water-filling shines most at low/medium SNR.",
        "num_subchannels": "Number of parallel eigenmodes/subcarriers.",
    },
    extension_hooks=("Upgrade to multi-user interfering WMMSE -> `wmmse_precoder`",),
    module_path="wannado.wce.blocks.baselines.water_filling_shannon",
    default_mechanisms=("shannon_awgn",),
    citation=MECHANISM_PROVENANCE["shannon_awgn"].citation,
)


@dataclass
class WaterFillingConfig:
    batch_size: int = 128
    num_tx: int = 8
    num_rx: int = 4
    snr_db: float = 5.0
    device: str = "auto"
    seed: int = 42


def _solve_water_filling_batch(gains: np.ndarray, total_power: float) -> np.ndarray:
    """Solve KKT water-filling for each row in gains (B, K) with sum_k p_k = total_power."""
    b = gains.shape[0]
    inv_g = 1.0 / np.maximum(gains, 1e-12)
    alloc = np.zeros_like(gains)
    for i in range(b):
        lo, hi = float(np.min(inv_g[i])), float(np.max(inv_g[i]) + total_power)
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            p = np.maximum(0.0, mid - inv_g[i])
            if np.sum(p) > total_power:
                hi = mid
            else:
                lo = mid
        alloc[i] = np.maximum(0.0, 0.5 * (lo + hi) - inv_g[i])
    return alloc


def provision_water_filling(config: WaterFillingConfig | None = None, **overrides: Any) -> ProvisionResult:
    """Compare KKT Water-Filling vs Equal-Power Allocation over MIMO SVD eigenmodes."""
    cfg = config or WaterFillingConfig()
    if overrides:
        cfg = WaterFillingConfig(**{**asdict(cfg), **overrides})
    t0 = time.perf_counter()
    set_global_seed(cfg.seed)
    dev = resolve_device(cfg.device)

    h = complex_normal((cfg.batch_size, cfg.num_rx, cfg.num_tx), device=dev)
    svals = torch.linalg.svdvals(h).cpu().numpy()
    eigen_gains = svals ** 2  # (B, min(Nr, Nt))

    p_tot = float(db_to_linear(cfg.snr_db))
    p_wf = _solve_water_filling_batch(eigen_gains, p_tot)
    p_eq = np.full_like(eigen_gains, p_tot / max(1, eigen_gains.shape[1]))

    rate_wf = np.sum(np.log2(1.0 + p_wf * eigen_gains), axis=-1)
    rate_eq = np.sum(np.log2(1.0 + p_eq * eigen_gains), axis=-1)

    prov = [
        ParamProvenance("snr_db", cfg.snr_db, "dB", "Total transmit SNR budget", "shannon_awgn", "BSTJ 1948", "shannon1948mathematical", "Sets total water volume P_tot for KKT water-filling."),
    ]
    return ProvisionResult(
        block_id=META.id,
        seed=cfg.seed,
        elapsed_seconds=time.perf_counter() - t0,
        config=asdict(cfg),
        tensors={"power_wf": torch.from_numpy(p_wf).to(torch.float32), "eigen_gains": torch.from_numpy(eigen_gains).to(torch.float32)},
        metrics={
            "water_filling_rate_bps_hz": float(np.mean(rate_wf)),
            "equal_power_rate_bps_hz": float(np.mean(rate_eq)),
            "wf_gain_over_equal_pct": float(((np.mean(rate_wf) - np.mean(rate_eq)) / max(np.mean(rate_eq), 1e-12)) * 100.0),
        },
        diagnostics={"device": str(dev), "normalized_channel_power": 1.0},
        parameter_provenance=prov,
        mechanisms_used=list(META.default_mechanisms),
        standard_ids_used=[],
    )
