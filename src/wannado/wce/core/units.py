"""Physical constants, RF unit conversions, Rayleigh distance, Doppler, and FBL special functions."""

from __future__ import annotations

import math

import numpy as np
import torch
from scipy import special

SPEED_OF_LIGHT_MPS: float = 2.99792458e8
BOLTZMANN_K_J_PER_K: float = 1.380649e-23
DEFAULT_TEMP_K: float = 290.0


def db_to_linear(db: float | np.ndarray | torch.Tensor) -> float | np.ndarray | torch.Tensor:
    """Convert power ratio in dB to linear scale: 10^(dB / 10)."""
    if isinstance(db, torch.Tensor):
        return torch.pow(10.0, db / 10.0)
    if isinstance(db, np.ndarray):
        return np.power(10.0, db / 10.0)
    return float(10.0 ** (float(db) / 10.0))


def linear_to_db(lin: float | np.ndarray | torch.Tensor, *, floor: float = 1e-30) -> float | np.ndarray | torch.Tensor:
    """Convert linear power ratio to dB: 10 * log10(max(lin, floor))."""
    if isinstance(lin, torch.Tensor):
        return 10.0 * torch.log10(torch.clamp(lin, min=floor))
    if isinstance(lin, np.ndarray):
        return 10.0 * np.log10(np.maximum(lin, floor))
    return float(10.0 * math.log10(max(float(lin), floor)))


def dbm_to_watt(dbm: float) -> float:
    """Convert absolute power in dBm to Watts."""
    return float(10.0 ** ((float(dbm) - 30.0) / 10.0))


def watt_to_dbm(watt: float, *, floor: float = 1e-30) -> float:
    """Convert absolute power in Watts to dBm."""
    return float(10.0 * math.log10(max(float(watt), floor)) + 30.0)


def thermal_noise_power_watt(
    bandwidth_hz: float,
    *,
    noise_figure_db: float = 7.0,
    temp_k: float = DEFAULT_TEMP_K,
) -> float:
    """Compute thermal noise power N = k_B * T * B * NF_linear in Watts."""
    if bandwidth_hz <= 0.0:
        raise ValueError(f"bandwidth_hz must be > 0, got {bandwidth_hz}.")
    nf_lin = float(db_to_linear(noise_figure_db))
    return float(BOLTZMANN_K_J_PER_K * temp_k * float(bandwidth_hz) * nf_lin)


def thermal_noise_power_dbm(
    bandwidth_hz: float,
    *,
    noise_figure_db: float = 7.0,
    temp_k: float = DEFAULT_TEMP_K,
) -> float:
    """Compute thermal noise power in dBm (e.g., ~ -174 dBm/Hz + 10*log10(B) + NF)."""
    return watt_to_dbm(thermal_noise_power_watt(bandwidth_hz, noise_figure_db=noise_figure_db, temp_k=temp_k))


def wavelength_m(fc_hz: float) -> float:
    """Carrier wavelength lambda = c / f_c in meters."""
    if fc_hz <= 0.0:
        raise ValueError(f"fc_hz must be > 0, got {fc_hz}.")
    return float(SPEED_OF_LIGHT_MPS / float(fc_hz))


def rayleigh_distance_m(aperture_m: float, fc_hz: float) -> float:
    """Classical Fraunhofer/Rayleigh far-field boundary Z_R = 2 * D^2 / lambda."""
    lam = wavelength_m(fc_hz)
    return float(2.0 * (float(aperture_m) ** 2) / lam)


def effective_rayleigh_distance_m(aperture_m: float, fc_hz: float, theta_rad: float = 0.0) -> float:
    """Effective Rayleigh distance scaling with broadside angle cos^2(theta) (Cui & Dai 2022)."""
    return float(rayleigh_distance_m(aperture_m, fc_hz) * (math.cos(theta_rad) ** 2))


def doppler_max_hz(speed_mps: float, fc_hz: float) -> float:
    """Maximum Doppler shift f_d = v * f_c / c in Hz."""
    return float(abs(float(speed_mps)) * float(fc_hz) / SPEED_OF_LIGHT_MPS)


def coherence_time_sec(fd_hz: float) -> float:
    """Clarke's 50% channel coherence time rule-of-thumb: T_c = 0.423 / f_d."""
    if fd_hz <= 0.0:
        return float("inf")
    return float(0.423 / float(fd_hz))


def qfunc(x: float | np.ndarray) -> float | np.ndarray:
    """Gaussian Q-function Q(x) = 0.5 * erfc(x / sqrt(2))."""
    res = 0.5 * special.erfc(np.asarray(x, dtype=np.float64) / math.sqrt(2.0))
    return float(res) if np.ndim(x) == 0 else res


def qfuncinv(p: float | np.ndarray) -> float | np.ndarray:
    """Inverse Gaussian Q-function Q^{-1}(p) = sqrt(2) * erfcinv(2p)."""
    p_arr = np.clip(np.asarray(p, dtype=np.float64), 1e-15, 1.0 - 1e-15)
    res = math.sqrt(2.0) * special.erfcinv(2.0 * p_arr)
    return float(res) if np.ndim(p) == 0 else res


def normalize_channel_power(h: torch.Tensor, *, target_power_per_entry: float = 1.0) -> torch.Tensor:
    """Normalize complex channel tensor so mean(|h|^2) equals target_power_per_entry."""
    cur = float(torch.mean(h.abs().pow(2)).item())
    if cur <= 1e-18:
        return h
    scale = math.sqrt(target_power_per_entry / cur)
    return h * scale
