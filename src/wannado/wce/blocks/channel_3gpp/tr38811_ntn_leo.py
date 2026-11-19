"""3GPP TR 38.811 & TR 38.821 Non-Terrestrial Network (NTN) LEO Satellite Channel.

[Lineage & Intuition]
- Family / Era / Tier : channel_3gpp / standard / S (IMT-2030 Ubiquitous Connectivity Pillar)
- Parent              : `rician_los`
- Key Diff            : Models orbital slant range d(elev), elevation-dependent LoS probability,
                        clutter loss, Ricean K-factor, long propagation delay, and orbital Doppler.
- Extension Hook [AI-Prompt Hook]:
    1. Pair high orbital Doppler with Delay-Doppler modulation -> `otfs_delay_doppler`.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
import torch

from wannado.wce.core.provenance import STANDARDS_PROVENANCE
from wannado.wce.core.types import ParamProvenance, ProvisionResult, WCEMeta
from wannado.wce.core.units import (
    SPEED_OF_LIGHT_MPS,
    db_to_linear,
    dbm_to_watt,
    linear_to_db,
    thermal_noise_power_watt,
)
from wannado.wce.core.utils import complex_normal, resolve_device, set_global_seed
from wannado.wce.standards.etsi_ieee import itur_p618_rain_attenuation_db, itur_p676_path_absorption_db
from wannado.wce.standards.gpp_extensions import interpolate_ntn_params

_EARTH_RADIUS_KM = 6371.0
_MU_EARTH_KM3_S2 = 398600.4418

META = WCEMeta(
    id="ntn_leo_satellite",
    name="3GPP TR 38.811/38.821 LEO Satellite NTN Channel",
    family="channel_3gpp",
    era="standard",
    tier="S",
    output_kind="complex_channel_tensor",
    supports_torch_complex=True,
    supports_autograd=True,
    standard_ids=("gpp_tr38811", "gpp_tr38821", "itur_m2160"),
    parent_id="rician_los",
    diff_from_parent="Couples orbital slant distance, TR 38.811 elevation-indexed clutter/K-factor, delay, and orbital Doppler.",
    summary_plain="Standardized LEO/GEO satellite-to-ground channel across S-band (2 GHz) and Ka-band (20-30 GHz) with elevation sweep.",
    when_to_use="Satellite-terrestrial integrated networks, IMT-2030 Ubiquitous Connectivity, and orbital Doppler compensation.",
    when_not_to_use="Low-altitude drone A2G (`uav_air_to_ground`) or terrestrial urban canyons (`tr38901_uma_umi_rma`).",
    hyperparam_intuition={
        "orbit_altitude_km": "Satellite height above Earth (typically 600 km or 1200 km LEO in TR 38.821).",
        "elevation_deg": "Ground terminal look angle above horizon (10 deg minimum cutoff to 90 deg zenith).",
        "fc_ghz": "S-band (~2 GHz) vs Ka-band (~20 GHz downlink / 30 GHz uplink).",
    },
    extension_hooks=("Test Delay-Doppler resilience under ~40 kHz LEO Doppler -> `otfs_delay_doppler`",),
    module_path="wannado.wce.blocks.channel_3gpp.tr38811_ntn_leo",
    default_mechanisms=("shannon_awgn",),
    citation=STANDARDS_PROVENANCE["gpp_tr38811"].citation,
)


@dataclass
class NTNLEOConfig:
    env_type: Literal["dense_urban", "urban", "suburban", "rural"] = "suburban"
    orbit_altitude_km: float = 600.0
    elevation_deg: float = 30.0
    fc_ghz: float = 2.0
    bandwidth_hz: float = 30e6
    eirp_dbw: float = 48.0
    rx_antenna_gain_dbi: float = 0.0
    noise_figure_db: float = 7.0
    rain_rate_mm_hr: float = 0.0
    water_vapor_g_m3: float = 7.5
    num_samples: int = 256
    device: str = "auto"
    seed: int = 42


def _slant_range_km(alt_km: float, elev_deg: float) -> float:
    re = _EARTH_RADIUS_KM
    el = math.radians(elev_deg)
    return float(math.sqrt((re * math.sin(el)) ** 2 + 2.0 * re * alt_km + alt_km ** 2) - re * math.sin(el))


def provision_ntn_leo_satellite(config: NTNLEOConfig | None = None, **overrides: Any) -> ProvisionResult:
    """Provision 3GPP TR 38.811/38.821 LEO satellite channel link budget, band-aware clutter, gases, Doppler, and Rician fading."""
    cfg = config or NTNLEOConfig()
    if overrides:
        cfg = NTNLEOConfig(**{**asdict(cfg), **overrides})
    t0 = time.perf_counter()
    rng = set_global_seed(cfg.seed)
    dev = resolve_device(cfg.device)

    ntn = interpolate_ntn_params(cfg.env_type, cfg.elevation_deg, fc_ghz=cfg.fc_ghz)
    slant_km = _slant_range_km(cfg.orbit_altitude_km, cfg.elevation_deg)
    slant_m = slant_km * 1e3
    delay_ms = (slant_m / SPEED_OF_LIGHT_MPS) * 1e3

    # Orbital velocity v_orb = sqrt(mu / (Re + h)); max radial projection ~ v_orb * Re/(Re+h) * cos(elev)
    v_orb_mps = math.sqrt(_MU_EARTH_KM3_S2 / (_EARTH_RADIUS_KM + cfg.orbit_altitude_km)) * 1e3
    radial_scale = (_EARTH_RADIUS_KM / (_EARTH_RADIUS_KM + cfg.orbit_altitude_km)) * math.cos(math.radians(cfg.elevation_deg))
    doppler_hz = (v_orb_mps * radial_scale * cfg.fc_ghz * 1e9) / SPEED_OF_LIGHT_MPS

    fspl_db = 32.45 + 20.0 * math.log10(cfg.fc_ghz) + 20.0 * math.log10(slant_m)
    atm_gas_db = float(
        itur_p676_path_absorption_db(
            cfg.fc_ghz,
            slant_m,
            water_vapor_g_m3=cfg.water_vapor_g_m3,
            elevation_deg=cfg.elevation_deg,
        )
    )
    rain_db = (
        float(itur_p618_rain_attenuation_db(cfg.fc_ghz, slant_m, rain_rate_mm_hr=cfg.rain_rate_mm_hr, elevation_deg=cfg.elevation_deg))
        if cfg.rain_rate_mm_hr > 0.0
        else 0.0
    )

    is_los = rng.uniform(0.0, 1.0, size=cfg.num_samples) < ntn["los_prob"]
    clutter_db = np.where(is_los, 0.0, ntn["clutter_loss_db"])
    sf_std_arr = np.where(is_los, ntn["sf_std_db"], ntn["sf_nlos_db"])
    sf_db = rng.normal(0.0, sf_std_arr, size=cfg.num_samples)
    total_pl_db = fspl_db + clutter_db + atm_gas_db + rain_db + sf_db

    tx_dbm = cfg.eirp_dbw + 30.0
    rx_power_dbm = tx_dbm + cfg.rx_antenna_gain_dbi - total_pl_db
    noise_w = thermal_noise_power_watt(cfg.bandwidth_hz, noise_figure_db=cfg.noise_figure_db)
    snr_lin = np.array([dbm_to_watt(p) / max(noise_w, 1e-30) for p in rx_power_dbm], dtype=np.float64)

    k_lin = float(db_to_linear(ntn["rician_k_db"]))
    w_los = math.sqrt(k_lin / (k_lin + 1.0))
    w_nlos = math.sqrt(1.0 / (k_lin + 1.0))
    h_nlos = complex_normal((cfg.num_samples, 1), variance=1.0, device=dev)
    h_small = (w_los * torch.ones_like(h_nlos) + w_nlos * h_nlos).to(torch.complex64)

    prov = [
        ParamProvenance("orbit_altitude_km", cfg.orbit_altitude_km, "km", "LEO satellite altitude", "gpp_tr38821", "Clause 5.1", "3gpp_tr38821", "Standard TR 38.821 600 km / 1200 km LEO reference orbit."),
        ParamProvenance("elevation_deg", cfg.elevation_deg, "deg", "Elevation look angle above horizon", "gpp_tr38811", "Table 6.6.2-1/2/3", "3gpp_tr38811", f"Band-aware ({'Ka' if cfg.fc_ghz >= 10.0 else 'S'}-band) LoS prob={ntn['los_prob']:.3f}, K={ntn['rician_k_db']:.1f} dB, CL={ntn['clutter_loss_db']:.1f} dB."),
    ]
    return ProvisionResult(
        block_id=META.id,
        seed=cfg.seed,
        elapsed_seconds=time.perf_counter() - t0,
        config=asdict(cfg),
        tensors={"h_small": h_small, "pathloss_db": torch.from_numpy(total_pl_db).to(torch.float32), "snr_db": torch.from_numpy(linear_to_db(snr_lin)).to(torch.float32)},
        metrics={
            "slant_range_km": slant_km,
            "one_way_delay_ms": delay_ms,
            "orbital_doppler_hz": doppler_hz,
            "fspl_db": fspl_db,
            "atmospheric_gas_loss_db": atm_gas_db,
            "rain_attenuation_db": rain_db,
            "mean_clutter_loss_db": float(np.mean(clutter_db)),
            "los_probability": ntn["los_prob"],
            "rician_k_db": ntn["rician_k_db"],
            "median_snr_db": float(np.median(linear_to_db(snr_lin))),
            "mean_channel_power": float(torch.mean(h_small.abs().pow(2)).item()),
        },
        diagnostics={"device": str(dev), "normalized_channel_power": 1.0},
        parameter_provenance=prov,
        mechanisms_used=list(META.default_mechanisms),
        standard_ids_used=list(META.standard_ids),
    )
