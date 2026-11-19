"""3GPP TR 38.901 (0.5 - 100 GHz) exact pathloss, LoS probability, O2I penetration, and CDL/TDL tables."""

from __future__ import annotations

import math
from typing import Literal

import numpy as np

from wannado.wce.core.units import SPEED_OF_LIGHT_MPS

Scenario38901 = Literal[
    "UMa",
    "UMi",
    "RMa",
    "InH",
    "InF-SL",
    "InF-DL",
    "InF-SH",
    "InF-DH",
    "InF-HH",
]

# TR 38.901 Table 7.4.3-1 material penetration loss parameters: L = a + b * f_GHz (dB)
O2I_MATERIAL_COEFFS: dict[str, tuple[float, float]] = {
    "standard_glass": (2.0, 0.2),
    "irr_glass": (23.0, 0.3),
    "concrete": (5.0, 4.0),
    "wood": (4.85, 0.12),
}

# TR 38.901 Clause 7.7.2 TDL normalized tap delays (scale by DS_ns) and powers (dB)
TDL_PROFILES: dict[str, dict[str, np.ndarray | float]] = {
    "TDL-A": {
        "k_db": float("-inf"),
        "delays_norm": np.array([0.0000, 0.3819, 0.4025, 0.5868, 0.4610, 0.5375, 0.6708, 0.5750, 0.7618, 1.5375, 1.8978, 2.2242]),
        "powers_db": np.array([-13.4, 0.0, -2.2, -4.0, -6.0, -8.2, -9.9, -10.5, -7.5, -15.9, -6.6, -16.7]),
    },
    "TDL-B": {
        "k_db": float("-inf"),
        "delays_norm": np.array([0.0000, 0.1072, 0.2155, 0.2095, 0.2870, 0.2986, 0.3752, 0.5055, 0.3681, 0.3697, 0.5700, 0.5283]),
        "powers_db": np.array([0.0, -2.2, -4.0, -3.2, -9.8, -1.2, -3.4, -5.2, -7.6, -3.0, -8.9, -9.0]),
    },
    "TDL-C": {
        "k_db": float("-inf"),
        "delays_norm": np.array([0.0000, 0.2099, 0.2219, 0.2329, 0.2176, 0.6366, 0.6448, 0.6560, 0.6584, 0.7935, 0.8213, 0.9336]),
        "powers_db": np.array([-4.4, -1.2, -3.5, -5.2, -2.5, 0.0, -2.2, -3.9, -7.4, -7.1, -10.7, -11.1]),
    },
    "TDL-D": {
        "k_db": 13.3,
        "delays_norm": np.array([0.0000, 0.035, 0.612, 1.363, 1.405, 1.804, 2.596, 1.775, 4.042, 7.937]),
        "powers_db": np.array([-0.2, -13.5, -18.8, -21.0, -22.8, -17.9, -20.1, -21.9, -22.9, -27.8]),
    },
    "TDL-E": {
        "k_db": 22.0,
        "delays_norm": np.array([0.0000, 0.5133, 0.5440, 0.5630, 0.5440, 0.7112, 1.9092, 1.9293, 1.9589, 2.6426]),
        "powers_db": np.array([-0.03, -22.03, -15.8, -18.1, -19.8, -22.9, -22.4, -18.6, -20.8, -22.6]),
    },
}

# Representative CDL-A..E cluster summaries (normalized delay, power dB, AoD deg, AoA deg)
CDL_PROFILES: dict[str, dict[str, np.ndarray | float]] = {
    "CDL-A": {
        "k_db": float("-inf"),
        "delays_norm": np.array([0.0, 0.3819, 0.4025, 0.5868, 0.4610, 0.5375, 0.6708, 0.5750, 0.7618, 1.5375]),
        "powers_db": np.array([-13.4, 0.0, -2.2, -4.0, -6.0, -8.2, -9.9, -10.5, -7.5, -15.9]),
        "aod_deg": np.array([-178.1, -4.2, -4.2, -4.2, 90.2, 90.2, 90.2, 121.5, -81.7, 158.4]),
        "aoa_deg": np.array([51.3, -152.7, -152.7, -152.7, 76.6, 76.6, 76.6, -1.8, -41.9, 94.2]),
    },
    "CDL-B": {
        "k_db": float("-inf"),
        "delays_norm": np.array([0.0, 0.1072, 0.2155, 0.2095, 0.2870, 0.2986, 0.3752, 0.5055]),
        "powers_db": np.array([0.0, -2.2, -4.0, -3.2, -9.8, -1.2, -3.4, -5.2]),
        "aod_deg": np.array([9.3, 9.3, 9.3, -34.1, -65.4, -11.4, -11.4, -11.4]),
        "aoa_deg": np.array([-173.3, -173.3, -173.3, 125.5, -88.0, 155.1, 155.1, 155.1]),
    },
    "CDL-C": {
        "k_db": float("-inf"),
        "delays_norm": np.array([0.0, 0.2099, 0.2219, 0.2329, 0.2176, 0.6366, 0.6448, 0.6560]),
        "powers_db": np.array([-4.4, -1.2, -3.5, -5.2, -2.5, 0.0, -2.2, -3.9]),
        "aod_deg": np.array([-46.6, -22.8, -22.8, -22.8, -40.7, 0.3, 0.3, 0.3]),
        "aoa_deg": np.array([-101.0, 120.0, 120.0, 120.0, -127.5, 170.4, 170.4, 170.4]),
    },
    "CDL-D": {
        "k_db": 13.3,
        "delays_norm": np.array([0.0, 0.035, 0.612, 1.363, 1.405, 1.804, 2.596, 4.042]),
        "powers_db": np.array([-0.2, -13.5, -18.8, -21.0, -22.8, -17.9, -20.1, -22.9]),
        "aod_deg": np.array([0.0, 89.2, 89.2, 89.2, 13.0, 13.0, 13.0, 34.6]),
        "aoa_deg": np.array([-180.0, 89.2, 89.2, 89.2, 163.0, 163.0, 163.0, -137.0]),
    },
    "CDL-E": {
        "k_db": 22.0,
        "delays_norm": np.array([0.0, 0.5133, 0.5440, 0.5630, 0.7112, 1.9092, 1.9293, 2.6426]),
        "powers_db": np.array([-0.03, -22.03, -15.8, -18.1, -22.9, -22.4, -18.6, -22.6]),
        "aod_deg": np.array([0.0, 57.5, 57.5, 57.5, -20.1, 16.2, 16.2, 24.6]),
        "aoa_deg": np.array([-180.0, 18.2, 18.2, 18.2, 101.8, 112.9, 112.9, -155.7]),
    },
}


def compute_los_probability_tr38901(
    scenario: str,
    d_2d_out_m: np.ndarray,
    *,
    h_ut_m: float = 1.5,
    clutter_density_r: float = 0.2,
    d_clutter_m: float = 10.0,
    h_c_m: float = 2.0,
    h_bs_m: float = 8.0,
) -> np.ndarray:
    """Compute LoS probability Pr_LOS per 3GPP TR 38.901 Table 7.4.2-1."""
    d = np.maximum(np.asarray(d_2d_out_m, dtype=np.float64), 1e-3)
    if scenario == "UMi":
        p = np.where(d <= 18.0, 1.0, (18.0 / d) + np.exp(-d / 36.0) * (1.0 - 18.0 / d))
        return np.clip(p, 0.0, 1.0)

    if scenario == "UMa":
        c_h = 0.0 if h_ut_m <= 13.0 else ((h_ut_m - 13.0) / 10.0) ** 1.5
        c_val = np.where(d <= 18.0, 0.0, 1.25 * 1e-6 * (d ** 3) * np.exp(-d / 150.0)) * c_h
        p_base = np.where(d <= 18.0, 1.0, ((18.0 / d) + np.exp(-d / 63.0) * (1.0 - 18.0 / d)) * (1.0 + c_val))
        return np.clip(p_base, 0.0, 1.0)

    if scenario == "RMa":
        return np.where(d <= 10.0, 1.0, np.exp(-(d - 10.0) / 1000.0))

    if scenario == "InH":
        # Open-office TR 38.901 Table 7.4.2-1: 1 (d<=5), exp(-(d-5)/70.8) (5<d<=49), exp(-(d-49)/211.7)*0.54 (d>49)
        return np.where(
            d <= 5.0,
            1.0,
            np.where(d <= 49.0, np.exp(-(d - 5.0) / 70.8), np.exp(-(d - 49.0) / 211.7) * 0.54),
        )

    if scenario.startswith("InF"):
        if scenario == "InF-HH":
            return np.ones_like(d)
        if scenario in ("InF-SL", "InF-DL"):
            k_sub = -d_clutter_m / math.log(max(1e-6, 1.0 - clutter_density_r))
        else:
            denom = math.log(max(1e-6, 1.0 - clutter_density_r))
            scale = max(1e-3, (h_bs_m - h_ut_m) / max(1e-3, h_c_m - h_ut_m))
            k_sub = -(d_clutter_m / denom) * scale
        return np.exp(-d / max(k_sub, 1e-3))

    raise ValueError(f"Unsupported TR 38.901 scenario '{scenario}'.")


# TR 38.901 Table 7.5-6 LOS Ricean K-factor (mean_dB, std_dB)
TR38901_K_FACTOR_DB: dict[str, tuple[float, float]] = {
    "UMa": (9.0, 3.5),
    "UMi": (9.0, 5.0),
    "RMa": (7.0, 4.0),
    "InH": (7.0, 4.0),
}


def compute_tr38901_uma_he(
    d_2d_m: np.ndarray,
    h_ut_m: float,
    *,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Effective environment height h_E per 3GPP TR 38.901 Table 7.4.1-1 Note 1."""
    d2 = np.asarray(d_2d_m, dtype=np.float64)
    if h_ut_m <= 13.0:
        return np.ones_like(d2, dtype=np.float64)
    c_h = ((h_ut_m - 13.0) / 10.0) ** 1.5
    g_d = np.where(d2 <= 18.0, 0.0, (5.0 / 4.0) * ((d2 / 100.0) ** 3) * np.exp(-d2 / 150.0))
    prob_he1 = 1.0 / (1.0 + g_d * c_h)
    max_disc = max(12.0, math.floor((h_ut_m - 1.5) / 3.0) * 3.0)
    disc_choices = np.arange(12.0, max_disc + 0.1, 3.0, dtype=np.float64)
    if rng is None:
        mean_disc = float(np.mean(disc_choices))
        return prob_he1 * 1.0 + (1.0 - prob_he1) * mean_disc
    u = rng.uniform(0.0, 1.0, size=d2.shape)
    sampled_disc = rng.choice(disc_choices, size=d2.shape)
    return np.where(u < prob_he1, 1.0, sampled_disc)


def _select_sf_std(los_mask: np.ndarray, sf_los: float, sf_nlos: float) -> float | np.ndarray:
    """Return scalar sigma_SF when all links share the same state, else per-link array."""
    if bool(np.all(los_mask)):
        return float(sf_los)
    if bool(np.all(~los_mask)):
        return float(sf_nlos)
    return np.where(los_mask, float(sf_los), float(sf_nlos))


def compute_pathloss_tr38901_db(
    scenario: str,
    d_2d_m: np.ndarray,
    fc_ghz: float,
    *,
    is_los: np.ndarray | bool = True,
    h_bs_m: float = 25.0,
    h_ut_m: float = 1.5,
    avg_building_h_m: float = 5.0,
    avg_street_w_m: float = 20.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, float | np.ndarray]:
    """Compute (PathLoss_dB, shadow_fading_std_dB) per 3GPP TR 38.901 Table 7.4.1-1."""
    d2 = np.maximum(np.asarray(d_2d_m, dtype=np.float64), 1.0)
    d3 = np.sqrt(d2 ** 2 + (h_bs_m - h_ut_m) ** 2)
    fc = float(fc_ghz)
    los_mask = np.broadcast_to(np.asarray(is_los, dtype=bool), d2.shape)
    fc_hz = fc * 1e9

    if scenario == "UMa":
        h_e_uma = compute_tr38901_uma_he(d2, h_ut_m, rng=rng)
        d_bp_uma = 4.0 * np.maximum(1.0, h_bs_m - h_e_uma) * np.maximum(0.5, h_ut_m - h_e_uma) * fc_hz / SPEED_OF_LIGHT_MPS
        pl1 = 28.0 + 22.0 * np.log10(d3) + 20.0 * math.log10(fc)
        pl2 = 28.0 + 40.0 * np.log10(d3) + 20.0 * math.log10(fc) - 9.0 * np.log10(d_bp_uma ** 2 + (h_bs_m - h_ut_m) ** 2)
        pl_los = np.where(d2 <= d_bp_uma, pl1, pl2)
        pl_nlos_raw = 13.54 + 39.08 * np.log10(d3) + 20.0 * math.log10(fc) - 0.6 * (h_ut_m - 1.5)
        pl_nlos = np.maximum(pl_los, pl_nlos_raw)
        pl = np.where(los_mask, pl_los, pl_nlos)
        sf_std = _select_sf_std(los_mask, 4.0, 6.0)
        return pl, sf_std

    if scenario == "UMi":
        h_e = 1.0
        d_bp = 4.0 * max(1.0, h_bs_m - h_e) * max(0.5, h_ut_m - h_e) * fc_hz / SPEED_OF_LIGHT_MPS
        pl1 = 32.4 + 21.0 * np.log10(d3) + 20.0 * math.log10(fc)
        pl2 = 32.4 + 40.0 * np.log10(d3) + 20.0 * math.log10(fc) - 9.5 * math.log10(d_bp ** 2 + (h_bs_m - h_ut_m) ** 2)
        pl_los = np.where(d2 <= d_bp, pl1, pl2)
        pl_nlos_raw = 35.3 * np.log10(d3) + 22.4 + 21.3 * math.log10(fc) - 0.3 * (h_ut_m - 1.5)
        pl_nlos = np.maximum(pl_los, pl_nlos_raw)
        pl = np.where(los_mask, pl_los, pl_nlos)
        sf_std = _select_sf_std(los_mask, 4.0, 7.82)
        return pl, sf_std

    if scenario == "RMa":
        h = avg_building_h_m
        w = avg_street_w_m
        d_bp_rma = 2.0 * math.pi * h_bs_m * h_ut_m * fc_hz / SPEED_OF_LIGHT_MPS
        def _pl1(dist: np.ndarray | float) -> np.ndarray | float:
            return (
                20.0 * np.log10(40.0 * math.pi * dist * fc / 3.0)
                + min(0.03 * (h ** 1.72), 10.0) * np.log10(dist)
                - min(0.044 * (h ** 1.72), 14.77)
                + 0.002 * math.log10(max(h, 1.1)) * dist
            )
        pl_los = np.where(d2 <= d_bp_rma, _pl1(d3), _pl1(d_bp_rma) + 40.0 * np.log10(d3 / max(d_bp_rma, 1.0)))
        pl_nlos_raw = (
            161.04
            - 7.1 * math.log10(w)
            + 7.5 * math.log10(h)
            - (24.37 - 3.7 * ((h / h_bs_m) ** 2)) * math.log10(h_bs_m)
            + (43.42 - 3.1 * math.log10(h_bs_m)) * (np.log10(d3) - 3.0)
            + 20.0 * math.log10(fc)
            - (3.2 * (math.log10(11.75 * h_ut_m) ** 2) - 4.97)
        )
        pl = np.where(los_mask, pl_los, np.maximum(pl_los, pl_nlos_raw))
        sf_std = _select_sf_std(los_mask, 4.0, 8.0)
        return pl, sf_std

    if scenario == "InH":
        pl_los = 32.4 + 17.3 * np.log10(d3) + 20.0 * math.log10(fc)
        pl_nlos = np.maximum(pl_los, 38.3 * np.log10(d3) + 17.30 + 24.9 * math.log10(fc))
        pl = np.where(los_mask, pl_los, pl_nlos)
        sf_std = _select_sf_std(los_mask, 3.0, 8.03)
        return pl, sf_std

    if scenario.startswith("InF"):
        pl_los = 31.84 + 21.50 * np.log10(d3) + 19.00 * math.log10(fc)
        nlos_table = {
            "InF-SL": (33.00, 25.50, 5.7),
            "InF-DL": (18.60, 35.70, 7.2),
            "InF-SH": (32.40, 23.00, 5.9),
            "InF-DH": (33.63, 21.90, 4.0),
            "InF-HH": (31.84, 21.50, 4.3),
        }
        c0, c1, std_nlos = nlos_table[scenario]
        pl_nlos = np.maximum(pl_los, c0 + c1 * np.log10(d3) + 20.0 * math.log10(fc))
        pl = np.where(los_mask, pl_los, pl_nlos)
        return pl, _select_sf_std(los_mask, 4.3, std_nlos)

    raise ValueError(f"Unsupported TR 38.901 scenario '{scenario}'.")


def compute_o2i_penetration_db(
    fc_ghz: float,
    *,
    model: Literal["low_loss", "high_loss"] = "low_loss",
    d_2d_in_m: np.ndarray | float = 5.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Compute O2I building penetration loss per 3GPP TR 38.901 Table 7.4.3-2."""
    def _mat(name: str) -> float:
        a, b = O2I_MATERIAL_COEFFS[name]
        return a + b * float(fc_ghz)

    l_glass = _mat("standard_glass")
    l_irr = _mat("irr_glass")
    l_conc = _mat("concrete")

    if model == "low_loss":
        pl_tw = 5.0 - 10.0 * math.log10(0.3 * (10.0 ** (-l_glass / 10.0)) + 0.7 * (10.0 ** (-l_conc / 10.0)))
        sigma_p = 4.4
    else:
        pl_tw = 5.0 - 10.0 * math.log10(0.7 * (10.0 ** (-l_irr / 10.0)) + 0.3 * (10.0 ** (-l_conc / 10.0)))
        sigma_p = 6.5

    d_in = np.asarray(d_2d_in_m, dtype=np.float64)
    pl_in = 0.5 * d_in
    sf = rng.normal(0.0, sigma_p, size=d_in.shape) if rng is not None else np.zeros_like(d_in)
    return pl_tw + pl_in + sf
