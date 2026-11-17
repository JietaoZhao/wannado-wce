"""3GPP TR 37.885 V2X Vehicular Sidelink Channel (Urban Grid & 6-Lane Highway).

[Lineage & Intuition]
- Family / Era / Tier : channel_3gpp / standard / A
- Parent              : `tr38901_uma_umi_rma`
- Key Diff            : Introduces V2V NLOSv (15 dB vehicle-body blockage), 433x250m Urban Grid vs
                        80m-breakpoint Highway wrap-around, and dual-mobility Doppler f_d = (v1+v2)*fc/c.
- Extension Hook [AI-Prompt Hook]:
    1. Pair high V2V mobility with OTFS -> `otfs_delay_doppler`.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
import torch

from wannado.wce.core.provenance import STANDARDS_PROVENANCE
from wannado.wce.core.types import ParamProvenance, ProvisionResult, WCEMeta
from wannado.wce.core.units import db_to_linear, dbm_to_watt, doppler_max_hz, linear_to_db, thermal_noise_power_watt
from wannado.wce.core.utils import complex_normal, resolve_device, set_global_seed
from wannado.wce.standards.gpp_extensions import compute_v2x_tr37885_pathloss

META = WCEMeta(
    id="v2x_vehicular_sl",
    name="3GPP TR 37.885 V2X Urban Grid & Highway Sidelink Channel",
    family="channel_3gpp",
    era="standard",
    tier="A",
    output_kind="complex_channel_tensor",
    supports_torch_complex=True,
    supports_autograd=True,
    standard_ids=("gpp_tr37885", "itur_m2160"),
    parent_id="tr38901_uma_umi_rma",
    diff_from_parent="Adds NLOSv (15 dB vehicle blockage), 80m highway breakpoint, and dual-end relative Doppler.",
    summary_plain="Standardized 5G NR-V2X sidelink propagation at 5.9/28/63 GHz for urban intersections and high-speed highways.",
    when_to_use="Autonomous platooning, V2V sidelink resource allocation, and vehicular HRLLC.",
    when_not_to_use="Static indoor or aerial scenarios.",
    hyperparam_intuition={
        "scenario": "urban_grid (433x250m city blocks with corner NLOS) vs highway (6 lanes with 80m breakpoint).",
        "nlosv_prob": "Probability a truck/bus obstructs the direct V2V sightline (+15 dB loss).",
    },
    extension_hooks=("Connect to FBL short-packet CAM safety alerts -> `hrllc_fbl_urllc`",),
    module_path="wannado.wce.blocks.channel_3gpp.tr37885_v2x_sl",
    default_mechanisms=("clarke_jakes_doppler",),
    citation=STANDARDS_PROVENANCE["gpp_tr37885"].citation,
)


@dataclass
class V2XConfig:
    scenario: Literal["urban_grid", "highway"] = "highway"
    num_links: int = 256
    fc_ghz: float = 5.9
    bandwidth_hz: float = 20e6
    tx_power_dbm: float = 23.0
    noise_figure_db: float = 9.0
    speed_tx_kmh: float = 120.0
    speed_rx_kmh: float = 100.0
    nlosv_prob: float = 0.25
    d_max_m: float = 350.0
    device: str = "auto"
    seed: int = 42


def provision_v2x_vehicular_sl(config: V2XConfig | None = None, **overrides: Any) -> ProvisionResult:
    """Provision TR 37.885 V2V sidelink pathloss, NLOSv blockage, dual-mobility Doppler, and fading."""
    cfg = config or V2XConfig()
    if overrides:
        cfg = V2XConfig(**{**asdict(cfg), **overrides})
    t0 = time.perf_counter()
    rng = set_global_seed(cfg.seed)
    dev = resolve_device(cfg.device)

    dists = rng.uniform(10.0, cfg.d_max_m, size=cfg.num_links)
    is_nlosv = rng.uniform(0.0, 1.0, size=cfg.num_links) < cfg.nlosv_prob
    pl_los, sf_std = compute_v2x_tr37885_pathloss(cfg.scenario, dists, cfg.fc_ghz, state="LOS")
    pl_nlosv, _ = compute_v2x_tr37885_pathloss(cfg.scenario, dists, cfg.fc_ghz, state="NLOSv")
    total_pl = np.where(is_nlosv, pl_nlosv, pl_los) + rng.normal(0.0, sf_std, size=cfg.num_links)

    # Dual-end worst-case closing velocity Doppler (TR 37.885 opposite-direction vehicles)
    v_rel_mps = (abs(cfg.speed_tx_kmh) + abs(cfg.speed_rx_kmh)) / 3.6
    fd_dual = doppler_max_hz(v_rel_mps, cfg.fc_ghz * 1e9)

    noise_w = thermal_noise_power_watt(cfg.bandwidth_hz, noise_figure_db=cfg.noise_figure_db)
    snr_lin = (dbm_to_watt(cfg.tx_power_dbm) * db_to_linear(-total_pl)) / max(noise_w, 1e-30)
    h_small = complex_normal((cfg.num_links, 1), variance=1.0, device=dev)

    prov = [
        ParamProvenance("scenario", cfg.scenario, "enum", "V2X deployment layout", "gpp_tr37885", "Clause 6.1", "3gpp_tr37885", "Highway 6-lane 80m breakpoint vs Urban Grid 433x250m."),
        ParamProvenance("fc_ghz", cfg.fc_ghz, "GHz", "5.9 GHz ITS / mmWave V2X band", "gpp_tr37885", "Clause 6.2", "3gpp_tr37885", "Standardized ITS sidelink band."),
        ParamProvenance("nlosv_prob", cfg.nlosv_prob, "prob", "Vehicle blockage probability (+15 dB)", "gpp_tr37885", "Clause 6.2.1", "3gpp_tr37885", "Captures intermediate heavy-vehicle shadowing."),
    ]
    return ProvisionResult(
        block_id=META.id,
        seed=cfg.seed,
        elapsed_seconds=time.perf_counter() - t0,
        config=asdict(cfg),
        tensors={"h_small": h_small, "pathloss_db": torch.from_numpy(total_pl).to(torch.float32), "snr_db": torch.from_numpy(linear_to_db(snr_lin)).to(torch.float32)},
        metrics={
            "nlosv_fraction": float(np.mean(is_nlosv)),
            "dual_mobility_doppler_hz": fd_dual,
            "mean_pathloss_db": float(np.mean(total_pl)),
            "median_snr_db": float(np.median(linear_to_db(snr_lin))),
            "mean_channel_power": float(torch.mean(h_small.abs().pow(2)).item()),
        },
        diagnostics={"device": str(dev), "normalized_channel_power": 1.0, "shadow_fading_std_db": float(np.mean(sf_std))},
        parameter_provenance=prov,
        mechanisms_used=list(META.default_mechanisms),
        standard_ids_used=list(META.standard_ids),
    )
