"""3GPP NR extensions: TS 38.211 numerology, TR 38.811/38.821 NTN, TR 36.777 UAV, TR 37.885 V2X, TR 38.838 XR, TR 38.848 A-IoT."""

from __future__ import annotations

import math
from typing import Literal

import numpy as np

# 3GPP TS 38.211 Clause 4.2-4.3: mu -> dict
NR_NUMEROLOGIES: dict[int, dict[str, float | str]] = {
    0: {"scs_khz": 15.0, "slots_per_subframe": 1, "slot_duration_ms": 1.0, "band_hint": "FR1 Macro / LTE Coexistence"},
    1: {"scs_khz": 30.0, "slots_per_subframe": 2, "slot_duration_ms": 0.5, "band_hint": "FR1 Mid-band TDD (n78/n41)"},
    2: {"scs_khz": 60.0, "slots_per_subframe": 4, "slot_duration_ms": 0.25, "band_hint": "FR1/FR2 Bridge / URLLC"},
    3: {"scs_khz": 120.0, "slots_per_subframe": 8, "slot_duration_ms": 0.125, "band_hint": "FR2 mmWave Data"},
    4: {"scs_khz": 240.0, "slots_per_subframe": 16, "slot_duration_ms": 0.0625, "band_hint": "FR2 Synchronization Bursts"},
}

# 3GPP TR 38.811 Table 6.6.1-1 & Table 6.6.2-1..3 elevation-dependent NTN parameters
# Indexed by elevation deg -> (los_prob, clutter_loss_nlos_db, rician_k_los_db, sf_los_db, sf_nlos_db)
NTN_ELEVATION_TABLE: dict[str, dict[int, tuple[float, float, float, float]]] = {
    "dense_urban": {
        10: (0.282, 34.3, 3.0, 3.5),
        20: (0.331, 30.9, 4.5, 3.5),
        30: (0.398, 29.0, 6.0, 3.5),
        40: (0.468, 27.7, 7.5, 3.5),
        50: (0.537, 26.8, 9.0, 3.5),
        60: (0.612, 26.2, 10.5, 3.5),
        70: (0.738, 25.8, 11.5, 3.5),
        80: (0.820, 25.5, 12.5, 3.5),
        90: (1.000, 25.5, 13.5, 3.5),
    },
    "urban": {
        10: (0.246, 34.3, 3.5, 4.0),
        20: (0.386, 30.9, 5.0, 4.0),
        30: (0.493, 29.0, 6.5, 4.0),
        40: (0.613, 27.7, 8.0, 4.0),
        50: (0.726, 26.8, 9.5, 4.0),
        60: (0.805, 26.2, 11.0, 4.0),
        70: (0.919, 25.8, 12.0, 4.0),
        80: (0.968, 25.5, 13.0, 4.0),
        90: (1.000, 25.5, 14.0, 4.0),
    },
    "suburban": {
        10: (0.782, 19.5, 5.5, 1.8),
        20: (0.869, 19.0, 7.0, 1.8),
        30: (0.919, 18.6, 8.5, 1.8),
        40: (0.929, 18.3, 10.0, 1.8),
        50: (0.935, 18.1, 11.5, 1.8),
        60: (0.940, 17.9, 13.0, 1.8),
        70: (0.949, 17.7, 14.5, 1.8),
        80: (0.952, 17.6, 15.5, 1.8),
        90: (1.000, 17.5, 16.5, 1.8),
    },
    "rural": {
        10: (0.782, 15.0, 6.0, 1.8),
        20: (0.869, 14.5, 8.0, 1.8),
        30: (0.919, 14.0, 9.5, 1.8),
        40: (0.929, 13.5, 11.0, 1.8),
        50: (0.935, 13.0, 12.5, 1.8),
        60: (0.940, 12.5, 14.0, 1.8),
        70: (0.949, 12.0, 15.0, 1.8),
        80: (0.952, 11.8, 16.0, 1.8),
        90: (1.000, 11.5, 17.0, 1.8),
    },
}

# TR 38.811 Table 6.6.2-1 / 6.6.2-2 / 6.6.2-3 S-band NLOS shadow fading std (dB)
_NTN_SBAND_SF_NLOS: dict[str, dict[int, float]] = {
    "dense_urban": {10: 15.5, 20: 13.9, 30: 12.4, 40: 11.7, 50: 10.6, 60: 10.0, 70: 9.5, 80: 9.0, 90: 8.8},
    "urban": {10: 15.5, 20: 13.9, 30: 12.4, 40: 11.7, 50: 10.6, 60: 9.8, 70: 9.2, 80: 8.8, 90: 8.5},
    "suburban": {10: 8.9, 20: 8.9, 30: 8.4, 40: 8.0, 50: 7.6, 60: 7.3, 70: 7.0, 80: 6.8, 90: 6.5},
    "rural": {10: 8.9, 20: 8.4, 30: 7.9, 40: 7.5, 50: 7.1, 60: 6.8, 70: 6.5, 80: 6.3, 90: 6.0},
}

# TR 38.811 Table 6.6.2-1 / 6.6.2-2 / 6.6.2-3 Ka-band NLOS clutter loss (dB) & NLOS SF std (dB)
_NTN_KABAND_CL_NLOS: dict[str, dict[int, tuple[float, float]]] = {
    "dense_urban": {10: (44.3, 15.5), 20: (39.9, 13.9), 30: (37.5, 12.4), 40: (35.8, 11.7), 50: (34.6, 10.6), 60: (33.8, 10.0), 70: (33.3, 9.5), 80: (33.0, 9.0), 90: (32.9, 8.8)},
    "urban": {10: (44.3, 15.5), 20: (39.9, 13.9), 30: (37.5, 12.4), 40: (35.8, 11.7), 50: (34.6, 10.6), 60: (33.8, 9.8), 70: (33.3, 9.2), 80: (33.0, 8.8), 90: (32.9, 8.5)},
    "suburban": {10: (29.5, 9.5), 20: (27.2, 8.9), 30: (25.6, 8.4), 40: (24.5, 8.0), 50: (23.7, 7.6), 60: (23.1, 7.3), 70: (22.7, 7.0), 80: (22.4, 6.8), 90: (22.0, 6.5)},
    "rural": {10: (24.0, 9.0), 20: (22.1, 8.4), 30: (20.8, 7.9), 40: (19.9, 7.5), 50: (19.2, 7.1), 60: (18.7, 6.8), 70: (18.3, 6.5), 80: (18.1, 6.3), 90: (17.9, 6.0)},
}

# Al-Hourani et al. (2014) environmental constants (a, b, eta_los_db, eta_nlos_db)
AL_HOURANI_ENV_PARAMS: dict[str, tuple[float, float, float, float]] = {
    "suburban": (4.88, 0.43, 0.1, 21.0),
    "urban": (9.61, 0.16, 1.0, 20.0),
    "dense_urban": (12.08, 0.11, 1.6, 23.0),
    "highrise_urban": (27.23, 0.08, 2.3, 34.0),
}


def get_nr_numerology(mu: int) -> dict[str, float | str]:
    """Return 3GPP TS 38.211 numerology dictionary for mu in {0, 1, 2, 3, 4}."""
    if mu not in NR_NUMEROLOGIES:
        raise ValueError(f"Unsupported NR numerology mu={mu}; expected one of {sorted(NR_NUMEROLOGIES.keys())}.")
    return dict(NR_NUMEROLOGIES[mu])


def interpolate_ntn_params(
    env_type: Literal["dense_urban", "urban", "suburban", "rural"],
    elevation_deg: float,
    *,
    fc_ghz: float = 2.0,
) -> dict[str, float]:
    """Interpolate elevation- and band-dependent NTN parameters from TR 38.811 Tables 6.6.1-1 & 6.6.2 (10..90 deg)."""
    key = "dense_urban" if env_type == "dense_urban" else env_type
    table = NTN_ELEVATION_TABLE[key]
    elev = float(np.clip(elevation_deg, 10.0, 90.0))
    angles = np.array(sorted(table.keys()), dtype=np.float64)
    vals = np.array([table[int(a)] for a in angles], dtype=np.float64)

    cl_vals = vals[:, 1]
    sf_los = float(np.interp(elev, angles, vals[:, 3]))
    # TR 38.811 Table 6.6.2 separates S-band (< 10 GHz) vs Ka-band (>= 10 GHz) NLOS clutter & SF
    if float(fc_ghz) >= 10.0:
        ka_tab = _NTN_KABAND_CL_NLOS[key]
        ka_arr = np.array([ka_tab[int(a)] for a in angles], dtype=np.float64)
        clutter_loss_db = float(np.interp(elev, angles, ka_arr[:, 0]))
        sf_nlos_db = float(np.interp(elev, angles, ka_arr[:, 1]))
    else:
        clutter_loss_db = float(np.interp(elev, angles, cl_vals))
        s_sf_tab = _NTN_SBAND_SF_NLOS[key]
        s_sf_arr = np.array([s_sf_tab[int(a)] for a in angles], dtype=np.float64)
        sf_nlos_db = float(np.interp(elev, angles, s_sf_arr))

    return {
        "los_prob": float(np.interp(elev, angles, vals[:, 0])),
        "clutter_loss_db": clutter_loss_db,
        "rician_k_db": float(np.interp(elev, angles, vals[:, 2])),
        "sf_std_db": sf_los,
        "sf_nlos_db": sf_nlos_db,
    }


def _normalize_uav_scenario(scenario: str) -> Literal["UMa-AV", "UMi-AV", "RMa-AV"]:
    norm = str(scenario).strip().replace("_", "-").upper()
    mapping: dict[str, Literal["UMa-AV", "UMi-AV", "RMa-AV"]] = {
        "UMA-AV": "UMa-AV",
        "UMI-AV": "UMi-AV",
        "RMA-AV": "RMa-AV",
    }
    if norm not in mapping:
        raise ValueError(f"Unsupported TR 36.777 UAV scenario '{scenario}'.")
    return mapping[norm]


def compute_uav_tr36777_los_probability(
    scenario: str,
    d_2d_m: np.ndarray,
    h_uav_m: float,
) -> np.ndarray:
    """Compute UAV LoS probability Pr_LoS per 3GPP TR 36.777 Table B-1 (up to 300 m altitude)."""
    from wannado.wce.standards.gpp_38901 import compute_los_probability_tr38901

    scen = _normalize_uav_scenario(scenario)
    d2 = np.maximum(np.asarray(d_2d_m, dtype=np.float64), 1.0)
    h = float(np.clip(h_uav_m, 1.5, 300.0))

    if scen == "UMa-AV":
        if h <= 22.5:
            return compute_los_probability_tr38901("UMa", d2, h_ut_m=h)
        if h > 100.0:
            return np.ones_like(d2, dtype=np.float64)
        d1 = max(460.0 * math.log10(h) - 700.0, 18.0)
        p1 = max(4300.0 * math.log10(h) - 3800.0, 18.0)
        return np.where(d2 <= d1, 1.0, np.clip(d1 / d2 + np.exp(-d2 / p1) * (1.0 - d1 / d2), 0.0, 1.0))

    if scen == "UMi-AV":
        if h <= 22.5:
            return compute_los_probability_tr38901("UMi", d2, h_ut_m=h)
        d1 = max(294.05 * math.log10(h) - 432.94, 18.0)
        p1 = max(233.98 * math.log10(h) - 0.95, 18.0)
        return np.where(d2 <= d1, 1.0, np.clip(d1 / d2 + np.exp(-d2 / p1) * (1.0 - d1 / d2), 0.0, 1.0))

    # RMa-AV (TR 36.777 Table B-1): 10m < h <= 40m -> probabilistic, h > 40m -> 1.0
    if h <= 10.0:
        return compute_los_probability_tr38901("RMa", d2, h_ut_m=h)
    if h > 40.0:
        return np.ones_like(d2, dtype=np.float64)
    d1 = max(1350.8 * math.log10(h) - 1602.0, 18.0)
    p1 = max(15021.0 * math.log10(h) - 16053.0, 1000.0)
    return np.where(d2 <= d1, 1.0, np.clip(d1 / d2 + np.exp(-d2 / p1) * (1.0 - d1 / d2), 0.0, 1.0))


def compute_uav_tr36777_pathloss(
    scenario: str,
    d_2d_m: np.ndarray,
    h_uav_m: float,
    fc_ghz: float,
    *,
    is_los: bool = True,
) -> tuple[np.ndarray, float]:
    """Compute UAV aerial pathloss & shadowing per 3GPP TR 36.777 Annex B (up to 300 m height)."""
    from wannado.wce.standards.gpp_38901 import compute_pathloss_tr38901_db

    scen = _normalize_uav_scenario(scenario)
    d2 = np.maximum(np.asarray(d_2d_m, dtype=np.float64), 1.0)
    h_uav = float(np.clip(h_uav_m, 1.5, 300.0))
    h_bs = {"UMa-AV": 25.0, "UMi-AV": 10.0, "RMa-AV": 35.0}[scen]

    # TR 36.777 Table B-2 delegates low-altitude UAVs to terrestrial TR 38.901 pathloss
    if (scen in ("UMa-AV", "UMi-AV") and h_uav <= 22.5) or (scen == "RMa-AV" and h_uav <= 10.0):
        base_scen = scen.split("-")[0]
        pl_t, sf_t = compute_pathloss_tr38901_db(base_scen, d2, fc_ghz, is_los=is_los, h_bs_m=h_bs, h_ut_m=h_uav)
        sf_scalar = float(np.mean(sf_t)) if isinstance(sf_t, np.ndarray) else float(sf_t)
        return pl_t, sf_scalar

    d3 = np.sqrt(d2 ** 2 + (h_uav - h_bs) ** 2)
    fc = float(fc_ghz)

    if scen == "UMa-AV":
        pl_los = 28.0 + 22.0 * np.log10(d3) + 20.0 * math.log10(fc)
        if is_los:
            return pl_los, float(4.64 * math.exp(-0.0066 * h_uav))
        pl_nlos_raw = -17.5 + (46.0 - 7.0 * math.log10(h_uav)) * np.log10(d3) + 20.0 * math.log10(40.0 * math.pi * fc / 3.0)
        return np.maximum(pl_los, pl_nlos_raw), 6.0

    if scen == "UMi-AV":
        # TR 36.777 Table B-2 UMi-AV LOS = max(FSPL, 30.9 + (22.25 - 0.5*log10(h))*log10(d3) + 20*log10(fc))
        pl_fspl = 32.4 + 20.0 * np.log10(d3) + 20.0 * math.log10(fc)
        pl_los = np.maximum(pl_fspl, 30.9 + (22.25 - 0.5 * math.log10(h_uav)) * np.log10(d3) + 20.0 * math.log10(fc))
        if is_los:
            return pl_los, float(max(2.0, 5.0 * math.exp(-0.01 * h_uav)))
        pl_nlos_raw = 32.4 + (43.2 - 7.6 * math.log10(h_uav)) * np.log10(d3) + 20.0 * math.log10(fc)
        return np.maximum(pl_los, pl_nlos_raw), 8.0

    # RMa-AV (3GPP TR 36.777 Table B-2)
    slope = max(23.9 - 1.8 * math.log10(h_uav), 20.0)
    pl_los = slope * np.log10(d3) + 20.0 * math.log10(40.0 * math.pi * fc / 3.0)
    if is_los:
        return pl_los, float(4.2 * math.exp(-0.0046 * h_uav))
    pl_nlos_raw = -12.0 + (35.0 - 5.3 * math.log10(h_uav)) * np.log10(d3) + 20.0 * math.log10(40.0 * math.pi * fc / 3.0)
    return np.maximum(pl_los, pl_nlos_raw), 6.0


def compute_v2x_tr37885_pathloss(
    scenario: Literal["urban_grid", "highway"],
    d_m: np.ndarray,
    fc_ghz: float,
    *,
    state: Literal["LOS", "NLOSv", "NLOS"] = "LOS",
) -> tuple[np.ndarray, float]:
    """Compute V2V/V2I sidelink pathloss & shadow fading std per 3GPP TR 37.885 Table 6.2.1-1."""
    d = np.maximum(np.asarray(d_m, dtype=np.float64), 1.0)
    fc = float(fc_ghz)
    if scenario == "urban_grid":
        pl_los = 38.77 + 16.7 * np.log10(d) + 18.2 * math.log10(fc)
        if state == "LOS":
            return pl_los, 3.0
        if state == "NLOSv":
            return pl_los + 15.0, 3.0
        pl_nlos = 36.85 + 30.0 * np.log10(d) + 18.9 * math.log10(fc)
        return np.maximum(pl_los, pl_nlos), 4.0

    # Highway single-slope LOS model (TR 37.885 Table 6.2.1-1: 32.4 + 20*log10(d) + 20*log10(fc), sigma_SF = 3 dB)
    pl_los = 32.4 + 20.0 * np.log10(d) + 20.0 * math.log10(fc)
    if state == "NLOSv":
        return pl_los + 15.0, 3.0
    if state == "NLOS":
        pl_nlos = np.maximum(pl_los, 36.85 + 30.0 * np.log10(d) + 18.9 * math.log10(fc))
        return pl_nlos, 4.0
    return pl_los, 3.0
