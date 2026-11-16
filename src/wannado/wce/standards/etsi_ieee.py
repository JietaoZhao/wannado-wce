"""ETSI GR RIS, ITU-R P.676 atmospheric absorption, IEEE 802.11bf/bn, and 3GPP TR 36.814 traffic specs."""

from __future__ import annotations

import math

import numpy as np

# Major O2 (~60/118.75 GHz) and H2O (22.2/183.3/325.1/380/448/557/752 GHz) resonance peaks
# Used to compute fast spectroscopic specific attenuation gamma(f) in dB/km per ITU-R P.676 guidance.
_O2_RESONANCE_GHZ: tuple[tuple[float, float, float], ...] = (
    (60.0, 14.5, 4.0),
    (118.75, 1.8, 1.5),
)
_H2O_RESONANCE_GHZ: tuple[tuple[float, float, float], ...] = (
    (22.235, 0.16, 2.8),
    (183.31, 28.0, 4.2),
    (325.15, 38.0, 5.0),
    (380.20, 140.0, 6.0),
    (448.00, 120.0, 6.0),
    (556.93, 1800.0, 8.0),
    (752.03, 1600.0, 8.0),
)

FTP_TRAFFIC_SPECS: dict[str, dict[str, float | str]] = {
    "ftp_model_1": {
        "standard": "3GPP TR 36.814 Annex A.2.1.3",
        "file_size_bytes": 2_000_000.0,
        "alt_file_size_bytes": 500_000.0,
        "arrival_process": "Poisson session arrival rate lambda",
    },
    "ftp_model_2": {
        "standard": "3GPP TR 36.814 Annex A.2.1.3",
        "file_size_bytes": 500_000.0,
        "mean_reading_time_sec": 5.0,
        "arrival_process": "Sequential download + exponential reading time D",
    },
    "ftp_model_3": {
        "standard": "3GPP TR 36.814 / TR 38.913",
        "file_size_bytes": 500_000.0,
        "arrival_process": "Poisson packet arrivals per user",
    },
}

VOIP_TRAFFIC_SPEC: dict[str, float | str] = {
    "standard": "3GPP TR 36.814 Annex A.2.1.3",
    "codec_rate_kbps": 12.2,
    "active_packet_interval_ms": 20.0,
    "active_packet_bytes": 40.0,
    "voice_activity_factor": 0.5,
    "sid_interval_ms": 160.0,
    "sid_packet_bytes": 15.0,
}

XR_TRAFFIC_SPEC: dict[str, float | tuple[int, ...] | str] = {
    "standard": "3GPP TR 38.838 Clause 4-6",
    "frame_rates_fps": (30, 60, 90, 120),
    "pdb_ms_default": 10.0,
    "pdb_ms_relaxed": 20.0,
    "packet_satisfaction_threshold": 0.99,
    "user_satisfaction_threshold": 0.90,
}


def itur_p676_attenuation_db_per_km(
    fc_ghz: float | np.ndarray,
    *,
    water_vapor_g_m3: float = 7.5,
    pressure_hpa: float = 1013.25,
    temp_c: float = 15.0,
) -> np.ndarray:
    """Compute atmospheric gaseous specific attenuation gamma(f) = gamma_d(f) + gamma_w(f) in dB/km (ITU-R P.676)."""
    f = np.maximum(np.asarray(fc_ghz, dtype=np.float64), 0.1)
    p_scale = float(pressure_hpa) / 1013.25
    t_scale = 288.15 / max(200.0, 273.15 + float(temp_c))
    rho_scale = max(0.0, float(water_vapor_g_m3)) / 7.5

    # Dry-air continuum + O2 resonance bands
    gamma_d = 0.0025 * p_scale * (t_scale ** 2) * np.sqrt(f / 10.0)
    for f0, peak, width in _O2_RESONANCE_GHZ:
        w = width * p_scale
        gamma_d = gamma_d + peak * p_scale * (w ** 2) / (((f - f0) ** 2) + (w ** 2))

    # Water-vapor continuum + H2O spectroscopic absorption peaks
    gamma_w = 0.00018 * rho_scale * (f ** 1.7) * (t_scale ** 2.5) * 1e-2
    for f0, peak, width in _H2O_RESONANCE_GHZ:
        w = width * (0.6 + 0.4 * p_scale)
        gamma_w = gamma_w + peak * rho_scale * (w ** 2) / (((f - f0) ** 2) + (w ** 2))

    return np.asarray(gamma_d + gamma_w, dtype=np.float64)


def itur_p676_path_absorption_db(
    fc_ghz: float | np.ndarray,
    distance_m: float | np.ndarray,
    *,
    water_vapor_g_m3: float = 7.5,
) -> np.ndarray:
    """Total atmospheric absorption loss in dB over link distance_m."""
    gamma = itur_p676_attenuation_db_per_km(fc_ghz, water_vapor_g_m3=water_vapor_g_m3)
    return gamma * (np.asarray(distance_m, dtype=np.float64) / 1000.0)
