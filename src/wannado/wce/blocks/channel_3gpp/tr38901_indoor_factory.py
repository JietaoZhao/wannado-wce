"""3GPP TR 38.901 Indoor Factory (InF-SL/DL/SH/DH/HH) Industrial Propagation (Rel-16+).

[Lineage & Intuition]
- Family / Era / Tier : channel_3gpp / standard / A (Industrial HRLLC Physical Layer Baseline)
- Parent              : `tr38901_uma_umi_rma`
- Key Diff            : Models dense metallic factory machinery clutter (r, d_clutter, h_c) and
                        InF-specific LoS/NLOS dual-clutter pathloss for smart manufacturing HRLLC.
- Extension Hook [AI-Prompt Hook]:
    1. Pair with finite-blocklength short packets -> `hrllc_fbl_urllc`.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
import torch

from wannado.wce.core.provenance import STANDARDS_PROVENANCE
from wannado.wce.core.types import ParamProvenance, ProvisionResult, WCEMeta
from wannado.wce.core.units import db_to_linear, dbm_to_watt, linear_to_db, thermal_noise_power_watt
from wannado.wce.core.utils import complex_normal, resolve_device, set_global_seed
from wannado.wce.standards.gpp_38901 import compute_los_probability_tr38901, compute_pathloss_tr38901_db

META = WCEMeta(
    id="tr38901_inf",
    name="3GPP TR 38.901 Indoor Factory (InF-SL/DL/SH/DH/HH)",
    family="channel_3gpp",
    era="standard",
    tier="A",
    output_kind="complex_channel_tensor",
    supports_torch_complex=True,
    supports_autograd=True,
    standard_ids=("gpp_tr38901", "gpp_tr38824", "itur_m2160"),
    parent_id="tr38901_uma_umi_rma",
    diff_from_parent="Captures factory metallic clutter density r, clutter height h_c, and sparse/dense BS elevation regimes.",
    summary_plain="Industrial smart-factory wireless environment calibrated to 3GPP TR 38.901 InF profiles for URLLC/HRLLC.",
    when_to_use="Industrial IoT, robotic factory control loops, and IMT-2030 HRLLC deployments.",
    when_not_to_use="Outdoor cellular macros (`tr38901_uma_umi_rma`) or outdoor vehicular (`v2x_vehicular_sl`).",
    hyperparam_intuition={
        "scenario": "SL=Sparse/Low, DL=Dense/Low, SH=Sparse/High, DH=Dense/High, HH=High/High.",
        "clutter_density_r": "Fraction of factory floor occupied by machinery clutter (e.g., 0.2 sparse, 0.6 dense).",
    },
    extension_hooks=("Connect SNR traces to finite-blocklength BLER checks -> `hrllc_fbl_urllc`",),
    module_path="wannado.wce.blocks.channel_3gpp.tr38901_indoor_factory",
    default_mechanisms=("tr38901_pathloss_los",),
    citation=STANDARDS_PROVENANCE["gpp_tr38901"].citation,
)


@dataclass
class TR38901InFConfig:
    scenario: Literal["InF-SL", "InF-DL", "InF-SH", "InF-DH", "InF-HH"] = "InF-DL"
    num_links: int = 256
    fc_ghz: float = 3.5
    bandwidth_hz: float = 100e6
    tx_power_dbm: float = 24.0
    noise_figure_db: float = 7.0
    h_bs_m: float = 2.0
    h_ut_m: float = 1.5
    h_clutter_m: float = 2.0
    clutter_density_r: float = 0.55
    d_clutter_m: float = 2.0
    d_max_m: float = 120.0
    device: str = "auto"
    seed: int = 42


def provision_tr38901_inf(config: TR38901InFConfig | None = None, **overrides: Any) -> ProvisionResult:
    """Provision 3GPP TR 38.901 Indoor Factory channel realizations and SNR distributions."""
    cfg = config or TR38901InFConfig()
    if overrides:
        cfg = TR38901InFConfig(**{**asdict(cfg), **overrides})
    t0 = time.perf_counter()
    rng = set_global_seed(cfg.seed)
    dev = resolve_device(cfg.device)

    d_2d = rng.uniform(5.0, cfg.d_max_m, size=cfg.num_links)
    p_los = compute_los_probability_tr38901(
        cfg.scenario,
        d_2d,
        h_ut_m=cfg.h_ut_m,
        clutter_density_r=cfg.clutter_density_r,
        d_clutter_m=cfg.d_clutter_m,
        h_c_m=cfg.h_clutter_m,
        h_bs_m=cfg.h_bs_m,
    )
    is_los = rng.uniform(0.0, 1.0, size=cfg.num_links) < p_los
    pl_db, sf_std = compute_pathloss_tr38901_db(
        cfg.scenario, d_2d, cfg.fc_ghz, is_los=is_los, h_bs_m=cfg.h_bs_m, h_ut_m=cfg.h_ut_m
    )
    total_loss_db = pl_db + rng.normal(0.0, sf_std, size=cfg.num_links)

    noise_w = thermal_noise_power_watt(cfg.bandwidth_hz, noise_figure_db=cfg.noise_figure_db)
    snr_lin = (dbm_to_watt(cfg.tx_power_dbm) * db_to_linear(-total_loss_db)) / max(noise_w, 1e-30)
    h_small = complex_normal((cfg.num_links, 1), variance=1.0, device=dev)

    prov = [
        ParamProvenance("scenario", cfg.scenario, "enum", "Indoor factory clutter profile", "gpp_tr38901", "Table 7.4.1-1 / 7.4.2-1", "3gpp_tr38901", "Factory floor blockage & dense scattering parameters."),
        ParamProvenance("clutter_density_r", cfg.clutter_density_r, "ratio", "Machinery clutter area fraction", "gpp_tr38901", "Table 7.4.2-1", "3gpp_tr38901", "Drives exponential LoS probability decay in factory aisles."),
    ]
    return ProvisionResult(
        block_id=META.id,
        seed=cfg.seed,
        elapsed_seconds=time.perf_counter() - t0,
        config=asdict(cfg),
        tensors={"h_small": h_small, "pathloss_db": torch.from_numpy(total_loss_db).to(torch.float32), "snr_db": torch.from_numpy(linear_to_db(snr_lin)).to(torch.float32)},
        metrics={
            "los_fraction": float(np.mean(is_los)),
            "mean_pathloss_db": float(np.mean(total_loss_db)),
            "p5_edge_snr_db": float(np.percentile(linear_to_db(snr_lin), 5.0)),
            "mean_channel_power": float(torch.mean(h_small.abs().pow(2)).item()),
        },
        diagnostics={"device": str(dev), "normalized_channel_power": 1.0, "shadow_fading_std_db": float(np.mean(sf_std))},
        parameter_provenance=prov,
        mechanisms_used=list(META.default_mechanisms),
        standard_ids_used=list(META.standard_ids),
    )
