"""UAV Air-to-Ground (A2G) Channel: 3GPP TR 36.777 & Al-Hourani Probabilistic LoS (2014).

[Lineage & Intuition]
- Family / Era / Tier : channel_3gpp / standard / S
- Parent              : `rician_los`
- Key Diff            : Captures the fundamental height trade-off: higher altitude raises elevation
                        angle (improving LoS probability) while increasing 3D path distance.
- Extension Hook [AI-Prompt Hook]:
    1. Pair UAV relay trajectory with mounted RIS -> `passive_ris_cascade`.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
import torch

from wannado.wce.core.provenance import MECHANISM_PROVENANCE
from wannado.wce.core.types import ParamProvenance, ProvisionResult, WCEMeta
from wannado.wce.core.units import db_to_linear, dbm_to_watt, linear_to_db, thermal_noise_power_watt
from wannado.wce.core.utils import complex_normal, resolve_device, set_global_seed
from wannado.wce.standards.gpp_extensions import AL_HOURANI_ENV_PARAMS, compute_uav_tr36777_pathloss

META = WCEMeta(
    id="uav_air_to_ground",
    name="UAV Air-to-Ground Channel (3GPP TR 36.777 + Al-Hourani 2014)",
    family="channel_3gpp",
    era="standard",
    tier="S",
    output_kind="complex_channel_tensor",
    supports_torch_complex=True,
    supports_autograd=True,
    standard_ids=("gpp_tr36777", "itur_m2160"),
    parent_id="rician_los",
    diff_from_parent="Adds elevation-angle LoS probability (Al-Hourani 2014) and 3GPP TR 36.777 aerial shadowing/pathloss.",
    summary_plain="Dual-mode UAV air-to-ground propagation combining Al-Hourani probabilistic LoS optimization and 3GPP TR 36.777 cellular aerial profiles.",
    when_to_use="UAV trajectory optimization, drone-BS placement, and cellular-connected drone communications.",
    when_not_to_use="Orbital satellite links (`ntn_leo_satellite`) or ground-only V2X (`v2x_vehicular_sl`).",
    hyperparam_intuition={
        "uav_altitude_m": "Hover altitude AGL (10 - 300 m); reveals optimal height balancing LoS vs FSPL.",
        "model_mode": "al_hourani (closed-form analytical optimization) vs tr36777 (cellular UMa/UMi/RMa-AV).",
    },
    extension_hooks=("Optimize 3D UAV trajectory with DRL via WirelessGymEnvAdapter",),
    module_path="wannado.wce.blocks.channel_3gpp.tr36777_uav_a2g",
    default_mechanisms=("al_hourani_a2g",),
    citation=MECHANISM_PROVENANCE["al_hourani_a2g"].citation,
)


@dataclass
class UAVA2GConfig:
    model_mode: Literal["al_hourani", "tr36777"] = "al_hourani"
    env_type: Literal["suburban", "urban", "dense_urban", "highrise_urban"] = "urban"
    tr36777_scenario: Literal["UMa-AV", "UMi-AV", "RMa-AV"] = "UMa-AV"
    uav_altitude_m: float = 100.0
    ground_radius_m: float = 300.0
    num_ground_users: int = 256
    fc_ghz: float = 2.4
    bandwidth_hz: float = 10e6
    tx_power_dbm: float = 23.0
    noise_figure_db: float = 7.0
    device: str = "auto"
    seed: int = 42


def provision_uav_air_to_ground(config: UAVA2GConfig | None = None, **overrides: Any) -> ProvisionResult:
    """Provision UAV air-to-ground probabilistic LoS, pathloss, SNR, and small-scale Rician fading."""
    cfg = config or UAVA2GConfig()
    if overrides:
        cfg = UAVA2GConfig(**{**asdict(cfg), **overrides})
    t0 = time.perf_counter()
    rng = set_global_seed(cfg.seed)
    dev = resolve_device(cfg.device)

    r_2d = np.sqrt(rng.uniform(1.0, cfg.ground_radius_m ** 2, size=cfg.num_ground_users))
    d_3d = np.sqrt(r_2d ** 2 + cfg.uav_altitude_m ** 2)
    elev_deg = np.degrees(np.arctan2(cfg.uav_altitude_m, r_2d))

    a, b, eta_los, eta_nlos = AL_HOURANI_ENV_PARAMS[cfg.env_type]
    p_los = 1.0 / (1.0 + a * np.exp(-b * (elev_deg - a)))
    is_los = rng.uniform(0.0, 1.0, size=cfg.num_ground_users) < p_los

    if cfg.model_mode == "al_hourani":
        fspl_db = 20.0 * np.log10(4.0 * math.pi * (cfg.fc_ghz * 1e9) * d_3d / 2.99792458e8)
        pl_expected_db = p_los * (fspl_db + eta_los) + (1.0 - p_los) * (fspl_db + eta_nlos)
        pl_sample_db = np.where(is_los, fspl_db + eta_los, fspl_db + eta_nlos)
        sf_std = 4.0
    else:
        pl_los_arr, sf_los = compute_uav_tr36777_pathloss(cfg.tr36777_scenario, r_2d, cfg.uav_altitude_m, cfg.fc_ghz, is_los=True)
        pl_nlos_arr, sf_nlos = compute_uav_tr36777_pathloss(cfg.tr36777_scenario, r_2d, cfg.uav_altitude_m, cfg.fc_ghz, is_los=False)
        pl_expected_db = p_los * pl_los_arr + (1.0 - p_los) * pl_nlos_arr
        pl_sample_db = np.where(is_los, pl_los_arr, pl_nlos_arr)
        sf_std = 0.5 * (sf_los + sf_nlos)
        pl_sample_db = pl_sample_db + rng.normal(0.0, sf_std, size=cfg.num_ground_users)

    noise_w = thermal_noise_power_watt(cfg.bandwidth_hz, noise_figure_db=cfg.noise_figure_db)
    snr_lin = (dbm_to_watt(cfg.tx_power_dbm) * db_to_linear(-pl_sample_db)) / max(noise_w, 1e-30)

    k_lin = torch.where(torch.from_numpy(is_los).to(dev), torch.full((cfg.num_ground_users,), 10.0, device=dev), torch.zeros((cfg.num_ground_users,), device=dev)).view(-1, 1)
    h_nlos = complex_normal((cfg.num_ground_users, 1), variance=1.0, device=dev)
    h_small = (torch.sqrt(k_lin / (k_lin + 1.0)) + torch.sqrt(1.0 / (k_lin + 1.0)) * h_nlos).to(torch.complex64)

    prov = [
        ParamProvenance("uav_altitude_m", cfg.uav_altitude_m, "m", "UAV flight height above ground", "gpp_tr36777", "Annex B", "3gpp_tr36777", "TR 36.777 covers 1.5 - 300 m UAV altitudes."),
        ParamProvenance("env_type", cfg.env_type, "enum", "Urban clutter constants (a, b, eta_LoS, eta_NLoS)", "al_hourani_a2g", "IEEE WCL 2014", "alhourani2014optimal", f"Uses (a={a}, b={b}, eta_LoS={eta_los} dB, eta_NLoS={eta_nlos} dB)."),
    ]
    return ProvisionResult(
        block_id=META.id,
        seed=cfg.seed,
        elapsed_seconds=time.perf_counter() - t0,
        config=asdict(cfg),
        tensors={"h_small": h_small, "pathloss_db": torch.from_numpy(pl_sample_db).to(torch.float32), "expected_pathloss_db": torch.from_numpy(pl_expected_db).to(torch.float32), "los_prob": torch.from_numpy(p_los).to(torch.float32), "snr_db": torch.from_numpy(linear_to_db(snr_lin)).to(torch.float32)},
        metrics={
            "mean_los_prob": float(np.mean(p_los)),
            "mean_elevation_deg": float(np.mean(elev_deg)),
            "mean_pathloss_db": float(np.mean(pl_sample_db)),
            "median_snr_db": float(np.median(linear_to_db(snr_lin))),
            "mean_channel_power": float(torch.mean(h_small.abs().pow(2)).item()),
        },
        diagnostics={"device": str(dev), "normalized_channel_power": 1.0},
        parameter_provenance=prov,
        mechanisms_used=list(META.default_mechanisms),
        standard_ids_used=list(META.standard_ids),
    )
