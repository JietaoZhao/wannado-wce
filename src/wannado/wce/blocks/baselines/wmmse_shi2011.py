"""Iteratively Weighted MMSE (WMMSE) Multi-User MISO Precoding Baseline (Shi et al., IEEE TSP 2011).

[Lineage & Intuition]
- Family / Era / Tier : baselines / standard / S (Gold-Standard Beamforming Benchmark)
- Parent              : `water_filling`
- Key Diff            : Iteratively optimizes MSE weights (u_k, w_k) and transmit beamformers V_k via
                        WMMSE equivalence to maximize weighted sum-rate under total BS power P_max.
- Extension Hook [AI-Prompt Hook]:
    1. Use WMMSE solutions as teacher/baseline floor for Deep Unfolding / DRL beamforming papers!
"""

from __future__ import annotations

import math
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
    id="wmmse_precoder",
    name="Iteratively Weighted MMSE (WMMSE) Multi-User Precoding Baseline",
    family="baselines",
    era="standard",
    tier="S",
    output_kind="baseline_policy_solution",
    supports_torch_complex=True,
    supports_autograd=False,
    standard_ids=("gpp_tr38901",),
    parent_id="water_filling",
    diff_from_parent="Solves NP-hard multi-user MISO interfering sum-rate maximization via iterative WMMSE updates.",
    summary_plain="Gold-standard WMMSE beamforming baseline vs Zero-Forcing (ZF) and Maximum-Ratio Transmission (MRT).",
    when_to_use="Mandatory benchmark whenever publishing AI/DRL/GNN precoding or power control papers.",
    when_not_to_use="Single-user orthogonal channels (`water_filling`).",
    hyperparam_intuition={
        "num_tx": "BS transmit antennas M.",
        "num_users": "Simultaneously served single-antenna users K.",
        "num_iters": "Number of WMMSE alternating coordinate updates (typically 15-30 iterations).",
    },
    extension_hooks=("Compare WMMSE sum-rate against DRL / Deep Unfolding models",),
    module_path="wannado.wce.blocks.baselines.wmmse_shi2011",
    default_mechanisms=("shi_wmmse_duality",),
    citation=MECHANISM_PROVENANCE["shi_wmmse_duality"].citation,
)


@dataclass
class WMMSEConfig:
    batch_size: int = 32
    num_tx: int = 8
    num_users: int = 4
    snr_db: float = 15.0
    num_iters: int = 25
    device: str = "auto"
    seed: int = 42


def _sum_rate_miso(h: np.ndarray, v: np.ndarray, sigma2: float) -> np.ndarray:
    """Compute sum-rate log2(1 + |h_k^H v_k|^2 / (sum_{j!=k}|h_k^H v_j|^2 + sigma2)) across batch."""
    # h: (B, K, M), v: (B, M, K) -> hv: (B, K, K)
    hv = np.einsum("bkm,bmj->bkj", np.conj(h), v)
    pow_mat = np.abs(hv) ** 2
    sig = np.diagonal(pow_mat, axis1=1, axis2=2)
    tot = np.sum(pow_mat, axis=-1)
    interf = np.maximum(tot - sig, 0.0)
    return np.sum(np.log2(1.0 + sig / np.maximum(interf + sigma2, 1e-12)), axis=-1)


def provision_wmmse_precoder(config: WMMSEConfig | None = None, **overrides: Any) -> ProvisionResult:
    """Run Shi et al. (2011) WMMSE beamforming vs Zero-Forcing (ZF) & MRT baselines."""
    cfg = config or WMMSEConfig()
    if overrides:
        cfg = WMMSEConfig(**{**asdict(cfg), **overrides})
    t0 = time.perf_counter()
    set_global_seed(cfg.seed)
    dev = resolve_device(cfg.device)

    h_t = complex_normal((cfg.batch_size, cfg.num_users, cfg.num_tx), device=dev)
    h_np = h_t.cpu().numpy().astype(np.complex128)  # (B, K, M)
    p_tot = 1.0
    sigma2 = p_tot / float(db_to_linear(cfg.snr_db))

    # 1. MRT baseline
    v_mrt = h_np.transpose(0, 2, 1)  # (B, M, K)
    v_mrt = v_mrt * np.sqrt(p_tot / np.maximum(np.sum(np.abs(v_mrt) ** 2, axis=(1, 2), keepdims=True), 1e-12))

    # 2. Zero-Forcing (ZF) baseline: V_zf = H^T (H^* H^T + reg I)^-1
    v_zf = np.zeros_like(v_mrt)
    v_wmmse = v_mrt.copy()
    eye_m = np.eye(cfg.num_tx, dtype=np.complex128)

    for b in range(cfg.batch_size):
        hb = h_np[b]  # (K, M), where y_k = conj(hb[k]) @ v_k
        # ZF under y = conj(hb) @ v: v_zf = hb^T (conj(hb) @ hb^T)^-1
        gram_k = np.conj(hb) @ hb.T + 1e-9 * np.eye(cfg.num_users)
        vz = hb.T @ np.linalg.inv(gram_k)
        vz *= math.sqrt(p_tot / max(float(np.sum(np.abs(vz) ** 2)), 1e-12))
        v_zf[b] = vz

        # WMMSE alternating updates (Shi et al., 2011)
        vb = v_mrt[b].copy()
        for _ in range(cfg.num_iters):
            hv = np.conj(hb) @ vb  # (K, K)
            denom = np.sum(np.abs(hv) ** 2, axis=1) + sigma2
            u = np.diag(hv) / np.maximum(denom, 1e-12)  # (K,)
            e_k = 1.0 - np.real(np.conj(u) * np.diag(hv))
            w = 1.0 / np.maximum(e_k, 1e-8)

            # Update V via bisection on Lagrange multiplier mu >= 0
            weights = w * (np.abs(u) ** 2)
            a_mat = np.zeros((cfg.num_tx, cfg.num_tx), dtype=np.complex128)
            b_rhs = np.zeros((cfg.num_tx, cfg.num_users), dtype=np.complex128)
            for k in range(cfg.num_users):
                hk = hb[k]
                a_mat += weights[k] * np.outer(hk, np.conj(hk))
                b_rhs[:, k] = w[k] * u[k] * hk
            # Bisection for mu >= 0 so trace(V V^H) <= p_tot
            vb_0 = np.linalg.solve(a_mat + 1e-10 * eye_m, b_rhs)
            if np.sum(np.abs(vb_0) ** 2) <= p_tot:
                vb = vb_0
            else:
                lo, hi = 0.0, 1.0
                while np.sum(np.abs(np.linalg.solve(a_mat + hi * eye_m, b_rhs)) ** 2) > p_tot and hi < 1e8:
                    hi *= 4.0
                for _ in range(30):
                    mid = 0.5 * (lo + hi)
                    v_try = np.linalg.solve(a_mat + mid * eye_m, b_rhs)
                    if np.sum(np.abs(v_try) ** 2) > p_tot:
                        lo = mid
                    else:
                        hi = mid
                vb = np.linalg.solve(a_mat + hi * eye_m, b_rhs)
        v_wmmse[b] = vb

    r_mrt = _sum_rate_miso(h_np, v_mrt, sigma2)
    r_zf = _sum_rate_miso(h_np, v_zf, sigma2)
    r_wmmse = _sum_rate_miso(h_np, v_wmmse, sigma2)

    prov = [
        ParamProvenance("num_iters", cfg.num_iters, "iterations", "WMMSE coordinate iterations", "shi_wmmse_duality", "IEEE TSP 2011", "shi2011iteratively", "Monotonically reaches a stationary KKT sum-rate solution."),
    ]
    return ProvisionResult(
        block_id=META.id,
        seed=cfg.seed,
        elapsed_seconds=time.perf_counter() - t0,
        config=asdict(cfg),
        tensors={"v_wmmse": torch.from_numpy(v_wmmse).to(torch.complex64), "v_zf": torch.from_numpy(v_zf).to(torch.complex64)},
        metrics={
            "wmmse_sum_rate_bps_hz": float(np.mean(r_wmmse)),
            "zf_sum_rate_bps_hz": float(np.mean(r_zf)),
            "mrt_sum_rate_bps_hz": float(np.mean(r_mrt)),
            "wmmse_gain_over_zf_pct": float(((np.mean(r_wmmse) - np.mean(r_zf)) / max(np.mean(r_zf), 1e-12)) * 100.0),
        },
        diagnostics={"device": str(dev), "normalized_channel_power": 1.0},
        parameter_provenance=prov,
        mechanisms_used=list(META.default_mechanisms),
        standard_ids_used=list(META.standard_ids),
    )
