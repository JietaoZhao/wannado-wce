"""Stochastic Geometry Homogeneous PPP Cellular Network (Andrews, Baccelli & Ganti, IEEE TCOM 2011).

[Lineage & Intuition]
- Family / Era / Tier : channel_classic / classic / S
- Parent              : `rayleigh_clarke`
- Key Diff            : Replaces single-cell fixed geometry with spatial Poisson Point Process BS
                        topologies and validates empirical SINR coverage against Andrews' closed-form integral.
- Extension Hook [AI-Prompt Hook]:
    1. Replace simple alpha pathloss with full 3GPP TR 38.901 UMa/UMi dual-slope -> `tr38901_uma_umi_rma`.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from scipy import integrate

from wannado.wce.core.geometry import generate_ppp_positions
from wannado.wce.core.provenance import MECHANISM_PROVENANCE
from wannado.wce.core.types import ParamProvenance, ProvisionResult, WCEMeta
from wannado.wce.core.units import db_to_linear, linear_to_db
from wannado.wce.core.utils import set_global_seed

META = WCEMeta(
    id="ppp_stochastic_geo",
    name="Stochastic Geometry PPP Cellular Interference Environment",
    family="channel_classic",
    era="classic",
    tier="S",
    output_kind="complex_channel_tensor",
    supports_torch_complex=False,
    supports_autograd=False,
    standard_ids=("itur_m2160",),
    parent_id="rayleigh_clarke",
    diff_from_parent="Distributes base stations via homogeneous PPP(lambda) and computes SINR coverage vs Andrews 2011 exact bound.",
    summary_plain="Tractable random-topology multi-BS cellular environment with closed-form SINR coverage cross-verification.",
    when_to_use="Large-scale cellular coverage, interference statistics, and dense IoT/BS topology studies.",
    when_not_to_use="Hexagonal wrap-around deterministic site layouts (`multicell_interference_mac`).",
    hyperparam_intuition={
        "bs_density_per_km2": "Average base stations per square km (PPP intensity lambda); cancels out in interference-limited coverage!",
        "pathloss_exponent": "Distance decay slope alpha (> 2, typically 3.5-4.0 for urban propagation).",
        "sinr_threshold_db": "Minimum SINR T required for successful link coverage.",
    },
    extension_hooks=("Switch BS association to 3GPP dual-slope LoS/NLoS pathloss",),
    module_path="wannado.wce.blocks.channel_classic.ppp_cellular_andrews2011",
    default_mechanisms=("ppp_stochastic_geometry", "shannon_awgn"),
    citation=MECHANISM_PROVENANCE["ppp_stochastic_geometry"].citation,
)


@dataclass
class PPPCellularConfig:
    num_realizations: int = 800
    area_radius_m: float = 2000.0
    bs_density_per_km2: float = 4.0
    pathloss_exponent: float = 3.8
    sinr_threshold_db: float = 0.0
    noise_to_signal_ratio: float = 0.0
    seed: int = 42


def _andrews_interference_limited_coverage(tau_lin: float, alpha: float) -> float:
    """Exact interference-limited PPP coverage probability: 1 / (1 + rho(tau, alpha))."""
    lower = tau_lin ** (-2.0 / alpha)
    integral, _ = integrate.quad(lambda u: 1.0 / (1.0 + (u ** (alpha / 2.0))), lower, float("inf"))
    rho = (tau_lin ** (2.0 / alpha)) * integral
    return float(1.0 / (1.0 + rho))


def provision_ppp_stochastic_geo(config: PPPCellularConfig | None = None, **overrides: Any) -> ProvisionResult:
    """Simulate homogeneous PPP cellular topologies around a typical origin user and compute SINR coverage."""
    cfg = config or PPPCellularConfig()
    if overrides:
        cfg = PPPCellularConfig(**{**asdict(cfg), **overrides})
    if cfg.pathloss_exponent <= 2.05:
        raise ValueError("pathloss_exponent must be > 2.05 for finite aggregate PPP interference.")
    t0 = time.perf_counter()
    rng = set_global_seed(cfg.seed)

    lam_m2 = cfg.bs_density_per_km2 / 1e6
    sinrs_lin: list[float] = []
    bs_counts: list[int] = []
    tau_lin = float(db_to_linear(cfg.sinr_threshold_db))

    for _ in range(cfg.num_realizations):
        bs_xyz = generate_ppp_positions(
            None,
            area_radius_m=cfg.area_radius_m,
            density_per_m2=lam_m2,
            z_height_m=0.0,
            min_distance_m=1.0,
            rng=rng,
        )
        bs_counts.append(bs_xyz.shape[0])
        dists = np.linalg.norm(bs_xyz[:, :2], axis=-1)
        order = np.argsort(dists)
        dists = dists[order]
        # Independent Exp(1) Rayleigh channel power gains |h|^2
        fading = rng.exponential(scale=1.0, size=dists.shape[0])
        rx_pow = fading * np.power(dists, -cfg.pathloss_exponent)
        sig = float(rx_pow[0])
        interf = float(np.sum(rx_pow[1:])) if rx_pow.size > 1 else 0.0
        noise = cfg.noise_to_signal_ratio * float(dists[0] ** (-cfg.pathloss_exponent))
        sinrs_lin.append(sig / max(interf + noise, 1e-30))

    sinr_arr = np.asarray(sinrs_lin, dtype=np.float64)
    emp_cov = float(np.mean(sinr_arr >= tau_lin))
    theo_cov = _andrews_interference_limited_coverage(tau_lin, cfg.pathloss_exponent)

    prov = [
        ParamProvenance("bs_density_per_km2", cfg.bs_density_per_km2, "BS/km^2", "Homogeneous PPP BS spatial density lambda", "ppp_stochastic_geometry", "IEEE TCOM 2011", "andrews2011tractable", "Determines Poisson BS counts per realization."),
        ParamProvenance("pathloss_exponent", cfg.pathloss_exponent, "dimensionless", "Distance attenuation slope alpha", "ppp_stochastic_geometry", "IEEE TCOM 2011", "andrews2011tractable", "Governs signal vs aggregate co-channel interference decay."),
    ]
    return ProvisionResult(
        block_id=META.id,
        seed=cfg.seed,
        elapsed_seconds=time.perf_counter() - t0,
        config=asdict(cfg),
        tensors={
            "sinr_linear": torch.from_numpy(sinr_arr).to(torch.float32),
            "sinr_db": torch.from_numpy(linear_to_db(sinr_arr)).to(torch.float32),
        },
        metrics={
            "empirical_coverage_prob": emp_cov,
            "andrews_theoretical_coverage": theo_cov,
            "coverage_theory_gap": abs(emp_cov - theo_cov),
            "mean_bs_count": float(np.mean(bs_counts)),
            "median_sinr_db": float(np.median(linear_to_db(sinr_arr))),
            "mean_rate_bps_hz": float(np.mean(np.log2(1.0 + sinr_arr))),
        },
        diagnostics={"normalized_channel_power": 1.0},
        parameter_provenance=prov,
        mechanisms_used=list(META.default_mechanisms),
        standard_ids_used=list(META.standard_ids),
    )
