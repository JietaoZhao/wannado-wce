"""ETSI GR RIS, ITU-R P.676 atmospheric absorption, IEEE 802.11bf/bn, and 3GPP TR 36.814 traffic specs."""

from __future__ import annotations

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
    """Compute atmospheric gaseous specific attenuation gamma(f) = gamma_d(f) + gamma_w(f) in dB/km (ITU-R P.676).

    Applies Van Vleck-Weisskopf (f / f0)^2 line-shape scaling to spectroscopic O2/H2O
    resonances so sub-10 GHz tails remain < 0.02 dB/km while preserving 22.2/60/118.75/183.3 GHz peaks.
    """
    f = np.maximum(np.asarray(fc_ghz, dtype=np.float64), 0.1)
    p_scale = float(pressure_hpa) / 1013.25
    t_scale = 288.15 / max(200.0, 273.15 + float(temp_c))
    rho_scale = max(0.0, float(water_vapor_g_m3)) / 7.5

    # Dry-air Debye continuum + O2 resonance bands with Van Vleck-Weisskopf (f/f0)^2 scaling
    gamma_d = 0.0052 * p_scale * (t_scale ** 2) * (f ** 2) / ((f ** 2) + 1.0)
    for f0, peak, width in _O2_RESONANCE_GHZ:
        w = width * p_scale
        vvw = (f / f0) ** 2
        gamma_d = gamma_d + peak * p_scale * vvw * (w ** 2) / (((f - f0) ** 2) + (w ** 2))

    # Water-vapor continuum + H2O spectroscopic absorption peaks with Van Vleck-Weisskopf scaling
    gamma_w = 5.5e-5 * rho_scale * (f ** 1.85) * (t_scale ** 2.5)
    for f0, peak, width in _H2O_RESONANCE_GHZ:
        w = width * (0.6 + 0.4 * p_scale)
        vvw = (f / f0) ** 2
        gamma_w = gamma_w + peak * rho_scale * vvw * (w ** 2) / (((f - f0) ** 2) + (w ** 2))

    return np.asarray(gamma_d + gamma_w, dtype=np.float64)


def itur_p676_path_absorption_db(
    fc_ghz: float | np.ndarray,
    distance_m: float | np.ndarray,
    *,
    water_vapor_g_m3: float = 7.5,
    elevation_deg: float | None = None,
) -> np.ndarray:
    """Total atmospheric absorption loss in dB over terrestrial distance_m or slant tropospheric path."""
    gamma = itur_p676_attenuation_db_per_km(fc_ghz, water_vapor_g_m3=water_vapor_g_m3)
    dist_km = np.asarray(distance_m, dtype=np.float64) / 1000.0
    if elevation_deg is not None and elevation_deg > 1.0:
        # ITU-R P.676 Annex 2 equivalent tropospheric gas scale height (~6 km) along slant path
        sin_el = max(float(np.sin(np.deg2rad(np.clip(elevation_deg, 5.0, 90.0)))), 0.087)
        eff_km = np.minimum(dist_km, 6.0 / sin_el)
        return np.asarray(gamma * eff_km, dtype=np.float64)
    return np.asarray(gamma * dist_km, dtype=np.float64)


# ITU-R P.838-3 anchor points: (f_GHz, k_H, alpha_H, k_V, alpha_V)
_P838_F_GHZ = np.array(
    [1.0, 2.0, 4.0, 6.0, 10.0, 15.0, 20.0, 28.0, 38.0, 50.0, 60.0, 73.0, 85.0, 100.0],
    dtype=np.float64,
)
_P838_KH = np.array(
    [0.0000259, 0.0000847, 0.0001071, 0.0007056, 0.01217, 0.04481, 0.09164, 0.1964, 0.3844, 0.6600, 0.8606, 1.080, 1.240, 1.367],
    dtype=np.float64,
)
_P838_AH = np.array(
    [0.9691, 1.0664, 1.6009, 1.5900, 1.2571, 1.1233, 1.0568, 0.9679, 0.8852, 0.8084, 0.7656, 0.7200, 0.6920, 0.6639],
    dtype=np.float64,
)
_P838_KV = np.array(
    [0.0000308, 0.0000998, 0.0002461, 0.0004878, 0.01129, 0.05008, 0.09611, 0.1900, 0.3650, 0.6472, 0.8515, 1.072, 1.235, 1.368],
    dtype=np.float64,
)
_P838_AV = np.array(
    [0.8592, 0.9490, 1.2476, 1.5728, 1.2156, 1.0440, 0.9847, 0.9350, 0.8670, 0.7974, 0.7586, 0.7150, 0.6880, 0.6615],
    dtype=np.float64,
)


def itur_p618_rain_attenuation_db(
    fc_ghz: float | np.ndarray,
    distance_m: float | np.ndarray,
    *,
    rain_rate_mm_hr: float = 25.0,
    elevation_deg: float = 0.0,
    polarization_tilt_deg: float = 45.0,
) -> np.ndarray:
    """Compute rain attenuation in dB using ITU-R P.838-3 specific attenuation and P.618/P.530 path reduction."""
    r = max(0.0, float(rain_rate_mm_hr))
    dist_km = np.maximum(np.asarray(distance_m, dtype=np.float64), 0.0) / 1000.0
    if r == 0.0:
        return np.zeros_like(dist_km + np.asarray(fc_ghz, dtype=np.float64) * 0.0, dtype=np.float64)

    f = np.clip(np.asarray(fc_ghz, dtype=np.float64), 1.0, 100.0)
    log_f = np.log10(f)
    log_grid = np.log10(_P838_F_GHZ)

    k_h = np.power(10.0, np.interp(log_f, log_grid, np.log10(_P838_KH)))
    k_v = np.power(10.0, np.interp(log_f, log_grid, np.log10(_P838_KV)))
    a_h = np.interp(log_f, log_grid, _P838_AH)
    a_v = np.interp(log_f, log_grid, _P838_AV)

    elev_rad = np.deg2rad(float(np.clip(elevation_deg, 0.0, 90.0)))
    tau_rad = np.deg2rad(float(polarization_tilt_deg))
    cos2_elev = float(np.cos(elev_rad) ** 2)
    cos2_tau = float(np.cos(2.0 * tau_rad))

    k = 0.5 * (k_h + k_v + (k_h - k_v) * cos2_elev * cos2_tau)
    alpha = 0.5 * (k_h * a_h + k_v * a_v + (k_h * a_h - k_v * a_v) * cos2_elev * cos2_tau) / np.maximum(k, 1e-12)
    gamma_r = k * np.power(r, alpha)  # specific attenuation dB/km

    # Effective rain-cell path length (slant cap at h_R=4 km for NTN/UAV links, ITU-R P.530 reduction for terrestrial)
    if elevation_deg > 1.0:
        rain_ceiling_km = 4.0 / max(float(np.sin(elev_rad)), 0.05)
        l_slant = np.minimum(dist_km, rain_ceiling_km)
        r_fac = 1.0 / (1.0 + l_slant / max(1.0, 35.0 * float(np.exp(-0.015 * r))))
        l_eff = l_slant * r_fac
    else:
        r_fac = 1.0 / (1.0 + dist_km / max(1.0, 35.0 * float(np.exp(-0.015 * r))))
        l_eff = dist_km * r_fac

    return np.asarray(gamma_r * l_eff, dtype=np.float64)
