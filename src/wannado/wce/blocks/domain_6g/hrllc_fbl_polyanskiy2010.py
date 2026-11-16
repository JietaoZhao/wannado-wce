"""Hyper-URLLC Finite Blocklength (FBL) Channel Dispersion Capacity (Polyanskiy, Poor & Verdu 2010; 3GPP TR 38.824).

[Lineage & Intuition]
- Family / Era / Tier : domain_6g / standard / S (IMT-2030 HRLLC Core Theory)
- Parent              : `awgn_link`
- Key Diff            : Replaces asymptotic Shannon capacity (n -> inf) with exact normal-approximation
                        finite blocklength rate R*(n, eps) = C - sqrt(V/n) * Q^{-1}(eps).
- Extension Hook [AI-Prompt Hook]:
    1. Apply to fading channels from `tr38901_inf` or `v2x_vehicular_sl` for short-packet URLLC BLER!
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
from wannado.wce.core.units import db_to_linear, qfunc, qfuncinv
from wannado.wce.core.utils import set_global_seed

META = WCEMeta(
    id="hrllc_fbl_urllc",
    name="Hyper-URLLC Finite Blocklength (FBL) Dispersion Capacity & BLER",
    family="domain_6g",
    era="standard",
    tier="S",
    output_kind="link_capacity_metric",
    supports_torch_complex=False,
    supports_autograd=False,
    standard_ids=("gpp_tr38824", "itur_m2160"),
    parent_id="awgn_link",
    diff_from_parent="Subtracts finite-blocklength dispersion penalty sqrt(V/n)*Q^{-1}(eps) from Shannon capacity.",
    summary_plain="Exact Polyanskiy-Poor-Verdu finite-blocklength capacity & decoding error rate calculator for short-packet URLLC/HRLLC.",
    when_to_use="Any URLLC/HRLLC, short control packet (n=100..500 channel uses), or ultra-low BLER (10^-5 to 10^-7) study.",
    when_not_to_use="Long eMBB file blocks where n >> 10,000 makes Shannon capacity tight.",
    hyperparam_intuition={
        "blocklength_n": "Number of complex/real channel uses (e.g., 100-500 for URLLC short packets).",
        "target_bler": "Maximum tolerable packet error probability (1e-5 for 5G URLLC; 1e-7 for 6G HRLLC).",
        "snr_db": "Operating link SNR in dB.",
    },
    extension_hooks=("Evaluate FBL outage over fading SNR tensors from `tr38901_inf`",),
    module_path="wannado.wce.blocks.domain_6g.hrllc_fbl_polyanskiy2010",
    default_mechanisms=("polyanskiy_fbl_dispersion", "shannon_awgn"),
    citation=MECHANISM_PROVENANCE["polyanskiy_fbl_dispersion"].citation,
)


@dataclass
class HRLLCFBLConfig:
    snr_db: float = 10.0
    blocklength_n: int = 200
    target_bler: float = 1e-5
    payload_bits: int = 256
    complex_channel: bool = True
    seed: int = 42


def provision_hrllc_fbl_urllc(config: HRLLCFBLConfig | None = None, **overrides: Any) -> ProvisionResult:
    """Compute exact finite-blocklength capacity R*(n, eps), Shannon gap, and achieved BLER for payload_bits."""
    cfg = config or HRLLCFBLConfig()
    if overrides:
        cfg = HRLLCFBLConfig(**{**asdict(cfg), **overrides})
    t0 = time.perf_counter()
    set_global_seed(cfg.seed)

    snr_lin = float(db_to_linear(cfg.snr_db))
    log2e = math.log2(math.e)
    if cfg.complex_channel:
        c_shannon = math.log2(1.0 + snr_lin)
        # Complex Gaussian channel dispersion V = (1 - (1 + snr)^-2) * (log2 e)^2
        v_disp = (1.0 - (1.0 + snr_lin) ** -2) * (log2e ** 2)
    else:
        c_shannon = 0.5 * math.log2(1.0 + snr_lin)
        v_disp = (snr_lin * (2.0 + snr_lin) / (2.0 * ((1.0 + snr_lin) ** 2))) * (log2e ** 2)

    q_inv = float(qfuncinv(cfg.target_bler))
    r_fbl = max(0.0, c_shannon - math.sqrt(v_disp / max(1, cfg.blocklength_n)) * q_inv)

    # Achieved BLER at requested payload_bits / blocklength_n coding rate
    req_rate = cfg.payload_bits / max(1, cfg.blocklength_n)
    z_arg = (c_shannon - req_rate) / max(math.sqrt(v_disp / max(1, cfg.blocklength_n)), 1e-12)
    achieved_bler = float(qfunc(z_arg))
    shannon_overest_pct = ((c_shannon - r_fbl) / max(c_shannon, 1e-12)) * 100.0

    prov = [
        ParamProvenance("blocklength_n", cfg.blocklength_n, "channel uses", "Short-packet blocklength n", "polyanskiy_fbl_dispersion", "IEEE TIT 2010", "polyanskiy2010channel", f"Causes {shannon_overest_pct:.1f}% capacity penalty vs asymptotic Shannon limit."),
        ParamProvenance("target_bler", cfg.target_bler, "prob", "Target packet error probability epsilon", "itur_m2160", "HRLLC 10^-5..10^-7", "itur_m2160_2023", "IMT-2030 Hyper-Reliable Low-Latency target."),
    ]
    return ProvisionResult(
        block_id=META.id,
        seed=cfg.seed,
        elapsed_seconds=time.perf_counter() - t0,
        config=asdict(cfg),
        tensors={"rate_gap": torch.tensor([c_shannon, r_fbl], dtype=torch.float32)},
        metrics={
            "shannon_capacity_bps_hz": c_shannon,
            "fbl_achievable_rate_bps_hz": r_fbl,
            "channel_dispersion_v": v_disp,
            "shannon_overestimation_pct": shannon_overest_pct,
            "required_rate_bps_hz": req_rate,
            "achieved_bler_for_payload": achieved_bler,
        },
        diagnostics={"normalized_channel_power": 1.0, "uses_fbl_correction": True},
        parameter_provenance=prov,
        mechanisms_used=list(META.default_mechanisms),
        standard_ids_used=list(META.standard_ids),
    )
