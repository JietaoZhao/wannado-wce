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

# 3GPP TR 38.811 Table 6.6.2-1 / 6.7.2 representative elevation-dependent NTN parameters
# Indexed by elevation deg -> (los_prob_urban, clutter_loss_sband_db, rician_k_los_db, sf_los_db)
NTN_ELEVATION_TABLE: dict[str, dict[int, tuple[float, float, float, float]]] = {
    "urban": {
        10: (0.246, 34.3, 3.5, 4.0),
        20: (0.386, 30.9, 5.0, 4.0),
        30: (0.493, 29.0, 6.5, 4.0),
        40: (0.613, 27.7, 8.0, 4.0),
        50: (0.726, 26.8, 9.5, 4.0),
        60: (0.805, 26.2, 11.0, 4.0),
        70: (0.919, 25.8, 12.0, 4.0),
        80: (0.968, 25.5, 13.0, 4.0),
        90: (0.992, 25.5, 14.0, 4.0),
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
        90: (0.998, 17.5, 16.5, 1.8),
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
        90: (0.998, 11.5, 17.0, 1.8),
    },
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


def interpolate_ntn_params(env_type: Literal["urban", "suburban", "rural"], elevation_deg: float) -> dict[str, float]:
    """Interpolate elevation-dependent NTN parameters from TR 38.811 brackets (10..90 deg)."""
    table = NTN_ELEVATION_TABLE[env_type]
    elev = float(np.clip(elevation_deg, 10.0, 90.0))
    angles = np.array(sorted(table.keys()), dtype=np.float64)
    vals = np.array([table[int(a)] for a in angles], dtype=np.float64)
    return {
        "los_prob": float(np.interp(elev, angles, vals[:, 0])),
        "clutter_loss_db": float(np.interp(elev, angles, vals[:, 1])),
        "rician_k_db": float(np.interp(elev, angles, vals[:, 2])),
        "sf_std_db": float(np.interp(elev, angles, vals[:, 3])),
    }


def compute_uav_tr36777_pathloss(
    scenario: Literal["UMa-AV", "UMi-AV", "RMa-AV"],
    d_2d_m: np.ndarray,
    h_uav_m: float,
    fc_ghz: float,
    *,
    is_los: bool = True,
) -> tuple[np.ndarray, float]:
    """Compute UAV aerial pathloss & shadowing per 3GPP TR 36.777 Annex B (up to 300 m height)."""
    d2 = np.maximum(np.asarray(d_2d_m, dtype=np.float64), 1.0)
    h_uav = float(np.clip(h_uav_m, 1.5, 300.0))
    h_bs = {"UMa-AV": 25.0, "UMi-AV": 10.0, "RMa-AV": 35.0}[scenario]
    d3 = np.sqrt(d2 ** 2 + (h_uav - h_bs) ** 2)
    fc = float(fc_ghz)

    if scenario == "UMa-AV":
        if is_los:
            pl = 28.0 + 22.0 * np.log10(d3) + 20.0 * math.log10(fc)
            sf = 4.64 * math.exp(-0.0066 * h_uav)
        else:
            pl = -17.5 + (46.0 - 7.0 * math.log10(h_uav)) * np.log10(d3) + 20.0 * math.log10(40.0 * math.pi * fc / 3.0)
            sf = 6.0
        return pl, float(sf)

    if scenario == "UMi-AV":
        if is_los:
            pl = np.maximum(28.0 + 22.0 * np.log10(d3) + 20.0 * math.log10(fc), 30.9 + (22.25 - 0.5 * math.log10(h_uav)) * np.log10(d3) + 20.0 * math.log10(fc))
            sf = max(2.0, 5.0 * math.exp(-0.01 * h_uav))
        else:
            pl = 32.4 + (43.2 - 7.6 * math.log10(h_uav)) * np.log10(d3) + 20.0 * math.log10(fc)
            sf = 8.0
        return pl, float(sf)

    # RMa-AV (3GPP TR 36.777 Table B-1)
    if is_los:
        slope = max(23.9 - 1.8 * math.log10(h_uav), 20.0)
        pl = slope * np.log10(d3) + 20.0 * math.log10(40.0 * math.pi * fc / 3.0)
        sf = 4.2 * math.exp(-0.0046 * h_uav)
    else:
        pl = -12.0 + (35.0 - 5.3 * math.log10(h_uav)) * np.log10(d3) + 20.0 * math.log10(40.0 * math.pi * fc / 3.0)
        sf = 6.0
    return pl, float(sf)


def compute_v2x_tr37885_pathloss(
    scenario: Literal["urban_grid", "highway"],
    d_m: np.ndarray,
    fc_ghz: float,
    *,
    state: Literal["LOS", "NLOSv", "NLOS"] = "LOS",
) -> tuple[np.ndarray, float]:
    """Compute V2V/V2I sidelink pathloss & shadow fading std per 3GPP TR 37.885 Clause 6."""
    d = np.maximum(np.asarray(d_m, dtype=np.float64), 1.0)
    fc = float(fc_ghz)
    if scenario == "urban_grid":
        pl_los = 38.77 + 16.7 * np.log10(d) + 18.2 * math.log10(fc)
        if state == "LOS":
            return pl_los, 5.2
        if state == "NLOSv":
            return pl_los + 15.0, 5.2
        pl_nlos = 36.85 + 30.0 * np.log10(d) + 18.9 * math.log10(fc)
        return np.maximum(pl_los, pl_nlos), 5.2

    # Highway two-slope model with d_BP = 80 m (TR 37.885 Table 6.2.1-1)
    pl_los = 32.4 + 20.0 * np.log10(d) + 20.0 * math.log10(fc)
    pl_bp = np.where(d <= 80.0, pl_los, pl_los + 12.0 * np.log10(d / 80.0))
    if state == "NLOSv":
        return pl_bp + 15.0, 3.3
    if state == "NLOS":
        pl_nlos = np.maximum(pl_bp, 36.85 + 30.0 * np.log10(d) + 18.9 * math.log10(fc))
        return pl_nlos, 5.2
    return pl_bp, 3.3
