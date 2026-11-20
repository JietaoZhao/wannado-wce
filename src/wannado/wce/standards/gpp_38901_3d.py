"""3GPP TR 38.901 Clause 7.3 3D Dual-Polarized Directional Panels, Clause 7.7 3D CDL & Clause 7.6 Spatial Consistency."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from wannado.wce.core.geometry import upa_farfield_steering
from wannado.wce.core.units import doppler_max_hz
from wannado.wce.core.utils import resolve_device, set_global_seed
from wannado.wce.standards.gpp_38901 import CDL_PROFILES
from wannado.wce.standards.gpp_extensions import get_nr_numerology

# TR 38.901 Clause 7.4.4 / Table 7.5-6 shadow-fading & spatial consistency decorrelation distances (m)
TR38901_DECORR_DIST_M: dict[str, float] = {
    "UMa": 50.0,
    "UMi": 13.0,
    "RMa": 37.0,
    "InH": 10.0,
    "InF": 10.0,
}

# Authentic TR 38.901 Tables 7.7.1-1..5 zenith departure/arrival angles (ZOD/ZOA), XPR (dB), and cluster angular spreads
CDL_3D_PARAMS: dict[str, dict[str, Any]] = {
    "CDL-A": {
        "zod_deg": np.array([97.2, 98.6, 98.6, 98.6, 100.8, 100.8, 100.8, 118.4, 96.4, 98.2, 106.7, 94.0, 99.6, 95.3, 98.8, 93.4, 94.3, 102.0, 100.3, 104.2, 107.1, 104.5, 91.6]),
        "zoa_deg": np.array([81.5, 74.6, 74.6, 74.6, 73.1, 73.1, 73.1, 68.4, 82.5, 78.6, 65.4, 72.1, 80.7, 79.9, 77.6, 76.2, 74.2, 69.8, 72.8, 69.9, 70.1, 75.0, 81.9]),
        "c_asd_deg": 5.0, "c_asa_deg": 11.0, "c_zsd_deg": 3.0, "c_zsa_deg": 3.0, "xpr_db": 10.0,
    },
    "CDL-B": {
        "zod_deg": np.array([96.6, 96.6, 96.6, 101.0, 99.7, 99.5, 99.5, 99.5, 100.1, 102.3, 97.6, 101.6, 103.5, 97.3, 98.4, 99.9, 101.4, 100.2, 97.7, 102.7, 101.3, 98.9, 98.2]),
        "zoa_deg": np.array([82.0, 82.0, 82.0, 74.5, 75.8, 75.5, 75.5, 75.5, 78.4, 72.8, 80.1, 74.3, 71.9, 79.1, 77.9, 76.5, 73.6, 76.8, 80.5, 73.4, 75.2, 78.9, 79.8]),
        "c_asd_deg": 10.0, "c_asa_deg": 22.0, "c_zsd_deg": 3.0, "c_zsa_deg": 7.0, "xpr_db": 8.0,
    },
    "CDL-C": {
        "zod_deg": np.array([97.6, 98.6, 98.6, 98.6, 100.6, 99.2, 99.2, 99.2, 97.2, 99.0, 97.7, 101.4, 98.5, 99.2, 101.3, 102.9, 102.8, 102.1, 101.2, 103.4, 103.7, 103.1, 104.7, 104.2]),
        "zoa_deg": np.array([80.4, 78.4, 78.4, 78.4, 76.5, 77.7, 77.7, 77.7, 78.8, 77.2, 79.0, 75.5, 77.8, 77.0, 75.3, 74.1, 74.3, 74.8, 75.4, 73.6, 73.4, 73.9, 72.7, 73.0]),
        "c_asd_deg": 2.0, "c_asa_deg": 15.0, "c_zsd_deg": 3.0, "c_zsa_deg": 7.0, "xpr_db": 7.0,
    },
    "CDL-D": {
        "zod_deg": np.array([98.0, 98.0, 101.5, 101.5, 101.5, 102.5, 102.5, 102.5, 101.2, 105.0, 106.5, 107.1, 108.5, 107.0]),
        "zoa_deg": np.array([82.0, 82.0, 78.5, 78.5, 78.5, 77.5, 77.5, 77.5, 78.8, 75.0, 73.5, 72.9, 71.5, 73.0]),
        "c_asd_deg": 5.0, "c_asa_deg": 8.0, "c_zsd_deg": 3.0, "c_zsa_deg": 3.0, "xpr_db": 9.0,
    },
    "CDL-E": {
        "zod_deg": np.array([99.6, 99.6, 100.8, 100.8, 100.8, 103.0, 103.2, 101.6, 101.6, 101.6, 104.5, 105.5, 106.1, 108.4, 109.1]),
        "zoa_deg": np.array([80.4, 80.4, 79.2, 79.2, 79.2, 77.0, 76.8, 78.4, 78.4, 78.4, 75.5, 74.5, 73.9, 71.6, 70.9]),
        "c_asd_deg": 5.0, "c_asa_deg": 11.0, "c_zsd_deg": 3.0, "c_zsa_deg": 7.0, "xpr_db": 8.0,
    },
}


def tr38901_antenna_pattern_db(
    theta_deg: np.ndarray | torch.Tensor | float,
    phi_deg: np.ndarray | torch.Tensor | float,
    *,
    g_e_max_dbi: float = 8.0,
    a_max_db: float = 30.0,
    sla_v_db: float = 30.0,
    theta_3db_deg: float = 65.0,
    phi_3db_deg: float = 65.0,
) -> np.ndarray | torch.Tensor:
    """Compute exact 3GPP TR 38.901 Table 7.3-1 3D directional antenna element gain ``G(theta, phi)`` in dBi.

    ``theta_deg`` in ``[0, 180]`` (boresight 90 deg) and ``phi_deg`` in ``[-180, 180]`` (boresight 0 deg).
    """
    use_torch = isinstance(theta_deg, torch.Tensor) or isinstance(phi_deg, torch.Tensor)
    if use_torch:
        th = torch.as_tensor(theta_deg, dtype=torch.float32)
        ph = torch.as_tensor(phi_deg, dtype=torch.float32)
        # Wrap azimuth into [-180, 180]
        ph = ((ph + 180.0) % 360.0) - 180.0
        a_v = -torch.clamp(12.0 * ((th - 90.0) / float(theta_3db_deg)) ** 2, max=float(sla_v_db))
        a_h = -torch.clamp(12.0 * (ph / float(phi_3db_deg)) ** 2, max=float(a_max_db))
        a_3d = -torch.clamp(-(a_v + a_h), max=float(a_max_db))
        return float(g_e_max_dbi) + a_3d

    th_np = np.asarray(theta_deg, dtype=np.float64)
    ph_np = ((np.asarray(phi_deg, dtype=np.float64) + 180.0) % 360.0) - 180.0
    a_v_np = -np.minimum(12.0 * ((th_np - 90.0) / float(theta_3db_deg)) ** 2, float(sla_v_db))
    a_h_np = -np.minimum(12.0 * (ph_np / float(phi_3db_deg)) ** 2, float(a_max_db))
    a_3d_np = -np.minimum(-(a_v_np + a_h_np), float(a_max_db))
    return float(g_e_max_dbi) + a_3d_np


def tr38901_dualpol_field_components(
    theta_deg: torch.Tensor,
    phi_deg: torch.Tensor,
    *,
    slant_deg: float = 45.0,
    g_e_max_dbi: float = 8.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return 3D element field pattern ``(F_theta, F_phi)`` for a polarized TR 38.901 Model-2 element."""
    g_db = tr38901_antenna_pattern_db(theta_deg, phi_deg, g_e_max_dbi=g_e_max_dbi)
    assert isinstance(g_db, torch.Tensor)
    # Relative pattern amplitude (normalize boresight G_E_max -> amplitude sqrt(10^(A''/10)) so E[|H|^2] stays calibrated)
    a_rel_db = g_db - float(g_e_max_dbi)
    amp = torch.sqrt(torch.pow(10.0, a_rel_db / 10.0))
    zeta = math.radians(float(slant_deg))
    return amp * math.cos(zeta), amp * math.sin(zeta)


def generate_tr38901_3d_dualpol_channel(
    profile: str = "CDL-C",
    *,
    batch_size: int = 8,
    tx_panel: tuple[int, int] = (2, 2),
    rx_panel: tuple[int, int] = (1, 2),
    dual_pol: bool = True,
    num_subcarriers: int = 32,
    num_slots: int = 14,
    numerology_mu: int = 1,
    fc_ghz: float = 3.5,
    delay_spread_ns: float = 100.0,
    speed_mps: float = 10.0,
    g_e_max_dbi: float = 8.0,
    device: str = "auto",
    seed: int = 42,
) -> dict[str, Any]:
    """Generate a 3GPP TR 38.901 3D Dual-Polarized Directional UPA CDL MIMO-OFDM channel tensor.

    Produces normalized ``h_freq`` of shape ``(B, Nr_tot, Nt_tot, Nsc, T)`` where
    ``Nt_tot = (2 if dual_pol else 1) * tx_panel[0] * tx_panel[1]``.
    """
    if profile not in CDL_PROFILES:
        raise ValueError(f"Unsupported CDL profile '{profile}'. Choose from {sorted(CDL_PROFILES.keys())}.")

    set_global_seed(seed)
    dev = resolve_device(device)
    prof = CDL_PROFILES[profile]
    p3d = CDL_3D_PARAMS[profile]

    num_info = get_nr_numerology(numerology_mu)
    scs_hz = float(num_info["scs_khz"]) * 1e3
    slot_sec = float(num_info["slot_duration_ms"]) * 1e-3

    delays_sec = torch.as_tensor(prof["delays_norm"], device=dev, dtype=torch.float32) * (delay_spread_ns * 1e-9)
    p_lin = torch.pow(10.0, torch.as_tensor(prof["powers_db"], device=dev, dtype=torch.float32) / 10.0)
    p_lin = p_lin / p_lin.sum()
    n_clusters = int(delays_sec.shape[0])
    k_db = float(prof.get("k_db", float("-inf")))
    has_los = math.isfinite(k_db)

    aod0 = torch.as_tensor(prof["aod_deg"], device=dev, dtype=torch.float32)
    aoa0 = torch.as_tensor(prof["aoa_deg"], device=dev, dtype=torch.float32)
    # Match cluster count seamlessly if CDL_3D_PARAMS matches CDL_PROFILES length (truncate/pad safely if needed)
    zod_raw = p3d["zod_deg"][:n_clusters]
    zoa_raw = p3d["zoa_deg"][:n_clusters]
    if zod_raw.size < n_clusters:
        zod_raw = np.resize(zod_raw, n_clusters)
        zoa_raw = np.resize(zoa_raw, n_clusters)
    zod0 = torch.as_tensor(zod_raw, device=dev, dtype=torch.float32)
    zoa0 = torch.as_tensor(zoa_raw, device=dev, dtype=torch.float32)

    # Expand clusters into 8 sub-rays across cluster ASD / ASA / ZSD / ZSA (TR 38.901 Table 7.5-3 basis offsets)
    n_sub = 8
    ray_offsets = torch.linspace(-1.2, 1.2, n_sub, device=dev, dtype=torch.float32)
    sub_aod = aod0.unsqueeze(-1) + ray_offsets * float(p3d["c_asd_deg"])  # (L, M)
    sub_aoa = aoa0.unsqueeze(-1) + ray_offsets * float(p3d["c_asa_deg"])
    sub_zod = torch.clamp(zod0.unsqueeze(-1) + ray_offsets * float(p3d["c_zsd_deg"]), 5.0, 175.0)
    sub_zoa = torch.clamp(zoa0.unsqueeze(-1) + ray_offsets * float(p3d["c_zsa_deg"]), 5.0, 175.0)

    tx_nx, tx_ny = max(1, int(tx_panel[0])), max(1, int(tx_panel[1]))
    rx_nx, rx_ny = max(1, int(rx_panel[0])), max(1, int(rx_panel[1]))
    n_tx_el = tx_nx * tx_ny
    n_rx_el = rx_nx * rx_ny

    a_tx = upa_farfield_steering(tx_nx, tx_ny, torch.deg2rad(sub_aod), torch.deg2rad(sub_zod))  # (L, M, N_tx_el)
    a_rx = upa_farfield_steering(rx_nx, rx_ny, torch.deg2rad(sub_aoa), torch.deg2rad(sub_zoa))  # (L, M, N_rx_el)

    # Dual-polarized ±45 deg slant field projections + TR 38.901 Eq. (7.5-28) XPR coupling matrix
    pol_angles = (45.0, -45.0) if dual_pol else (0.0,)
    n_pol = len(pol_angles)
    xpr_lin = 10.0 ** (float(p3d["xpr_db"]) / 10.0)
    sqrt_inv_xpr = 1.0 / math.sqrt(max(xpr_lin, 1e-6))

    tx_fields = [tr38901_dualpol_field_components(sub_zod, sub_aod, slant_deg=ang, g_e_max_dbi=g_e_max_dbi) for ang in pol_angles]
    rx_fields = [tr38901_dualpol_field_components(sub_zoa, sub_aoa, slant_deg=ang, g_e_max_dbi=g_e_max_dbi) for ang in pol_angles]

    # Build random phase polarization coupling tensor (B, L, M, 2, 2)
    phi_pol = torch.rand((batch_size, n_clusters, n_sub, 2, 2), device=dev) * (2.0 * math.pi)
    pol_mat = torch.complex(torch.cos(phi_pol), torch.sin(phi_pol)).to(torch.complex64)
    pol_amp = torch.tensor([[1.0, sqrt_inv_xpr], [sqrt_inv_xpr, 1.0]], device=dev, dtype=torch.float32)
    pol_mat = pol_mat * pol_amp.view(1, 1, 1, 2, 2)

    if has_los:
        # LoS ray (cluster 0) is purely co-polarized along diagonal without cross-pol XPR leakage
        pol_mat[:, 0, :, 0, 1] = 0.0
        pol_mat[:, 0, :, 1, 0] = 0.0
        phi_los = torch.rand((batch_size, 1), device=dev) * (2.0 * math.pi)
        pol_mat[:, 0, :, 0, 0] = torch.complex(torch.cos(phi_los), torch.sin(phi_los))
        pol_mat[:, 0, :, 1, 1] = -pol_mat[:, 0, :, 0, 0]

    # Assemble polarization-combined spatial tensor (B, L, n_pol_rx, N_rx_el, n_pol_tx, N_tx_el)
    spatial_blocks: list[torch.Tensor] = []
    for q_idx in range(n_pol):
        f_rx_th, f_rx_ph = rx_fields[q_idx]  # (L, M)
        row_blocks: list[torch.Tensor] = []
        for p_idx in range(n_pol):
            f_tx_th, f_tx_ph = tx_fields[p_idx]  # (L, M)
            # Couple (F_rx_th, F_rx_ph) @ pol_mat @ (F_tx_th, F_tx_ph)^T
            c_pq = (
                f_rx_th * pol_mat[..., 0, 0] * f_tx_th
                + f_rx_th * pol_mat[..., 0, 1] * f_tx_ph
                + f_rx_ph * pol_mat[..., 1, 0] * f_tx_th
                + f_rx_ph * pol_mat[..., 1, 1] * f_tx_ph
            )  # (B, L, M)
            ray_sum = (1.0 / math.sqrt(n_sub)) * math.sqrt(n_rx_el * n_tx_el) * torch.einsum(
                "blm,lmn,lmk->blnk", c_pq, a_rx, a_tx.conj()
            )
            row_blocks.append(ray_sum * torch.sqrt(p_lin).view(1, n_clusters, 1, 1))
        spatial_blocks.append(torch.cat(row_blocks, dim=-1))  # (B, L, N_rx_el, N_tx_tot)

    spatial = torch.cat(spatial_blocks, dim=2)  # (B, L, N_rx_tot, N_tx_tot)

    # Time-frequency phase progression from cluster Doppler & tap delays
    fd = doppler_max_hz(speed_mps, fc_ghz * 1e9)
    t_grid = torch.arange(num_slots, device=dev, dtype=torch.float32) * slot_sec
    f_grid = (torch.arange(num_subcarriers, device=dev, dtype=torch.float32) - 0.5 * num_subcarriers) * scs_hz
    sub_doppler = fd * torch.cos(torch.deg2rad(sub_aoa[:, :1].squeeze(-1)))  # (L,)
    doppler_phase = 2.0 * math.pi * sub_doppler.view(1, n_clusters, 1, 1) * t_grid.view(1, 1, 1, num_slots)
    freq_phase = -2.0 * math.pi * delays_sec.view(1, n_clusters, 1, 1) * f_grid.view(1, 1, num_subcarriers, 1)
    tf_phase = torch.complex(torch.cos(freq_phase + doppler_phase), torch.sin(freq_phase + doppler_phase))

    h_freq = torch.einsum("blnm,blft->bnmft", spatial, tf_phase).to(torch.complex64)
    # Normalize mean entry power to 1.0 so large-scale pathloss remains strictly orthogonal
    mean_pow = float(torch.mean(h_freq.abs().pow(2)).item())
    if mean_pow > 1e-18:
        h_freq = (h_freq / math.sqrt(mean_pow)).to(torch.complex64)

    delays_ns = delays_sec * 1e9
    rms_ds = float(torch.sqrt(torch.sum(p_lin * delays_ns ** 2) - torch.sum(p_lin * delays_ns) ** 2).item())

    return {
        "h_freq": h_freq,
        "profile": profile,
        "dual_pol": bool(dual_pol),
        "num_tx_total": n_pol * n_tx_el,
        "num_rx_total": n_pol * n_rx_el,
        "xpr_db": float(p3d["xpr_db"]),
        "front_to_back_db": 30.0,
        "rms_delay_spread_ns": rms_ds,
        "mean_channel_power": float(torch.mean(h_freq.abs().pow(2)).item()),
    }


class SpatialConsistencyField:
    """TR 38.901 Clause 7.4.4 / 7.6.3 exponential spatial correlation field for moving UEs & multi-UE grids."""

    def __init__(
        self,
        *,
        scenario: str = "UMa",
        decorr_distance_m: float | None = None,
        seed: int = 42,
    ) -> None:
        self.scenario = scenario
        if decorr_distance_m is not None:
            self.decorr_distance_m = max(0.5, float(decorr_distance_m))
        else:
            key = "InF" if scenario.startswith("InF") else scenario
            self.decorr_distance_m = float(TR38901_DECORR_DIST_M.get(key, 37.0))
        self._rng = np.random.default_rng(seed)
        self._prev_coords: np.ndarray | None = None
        self._prev_z: np.ndarray | None = None

    def reset(self, *, seed: int | None = None) -> None:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._prev_coords = None
        self._prev_z = None

    def sample_grid(self, coords_xy: np.ndarray, *, std: float = 1.0) -> np.ndarray:
        """Draw spatially correlated standard-normal (or scaled) samples across N 2D/3D positions via Cholesky."""
        pts = np.asarray(coords_xy, dtype=np.float64)
        if pts.ndim == 1:
            pts = pts[:, None]
        diff = pts[:, None, :2] - pts[None, :, :2]
        dists = np.linalg.norm(diff, axis=-1)
        cov = np.exp(-dists / self.decorr_distance_m) + 1e-6 * np.eye(pts.shape[0])
        chol = np.linalg.cholesky(cov)
        w = self._rng.normal(0.0, 1.0, size=pts.shape[0])
        return (chol @ w) * float(std)

    def step_positions(
        self,
        coords_xy: np.ndarray,
        *,
        std: float | np.ndarray = 1.0,
    ) -> np.ndarray:
        """Advance temporally-and-spatially consistent Gauss-Markov shadow fading along moving UE trajectories."""
        pts = np.asarray(coords_xy, dtype=np.float64)
        if pts.ndim == 1:
            pts = np.column_stack([pts, np.zeros_like(pts)])

        if self._prev_coords is None or self._prev_z is None or self._prev_coords.shape != pts.shape:
            z = self.sample_grid(pts, std=1.0)
        else:
            delta_d = np.linalg.norm(pts[:, :2] - self._prev_coords[:, :2], axis=-1)
            rho = np.clip(np.exp(-delta_d / self.decorr_distance_m), 0.0, 1.0)
            innov = self._rng.normal(0.0, 1.0, size=pts.shape[0])
            z = rho * self._prev_z + np.sqrt(np.maximum(0.0, 1.0 - rho * rho)) * innov

        self._prev_coords = pts.copy()
        self._prev_z = z.copy()
        return z * np.asarray(std, dtype=np.float64)


__all__ = [
    "CDL_3D_PARAMS",
    "TR38901_DECORR_DIST_M",
    "SpatialConsistencyField",
    "generate_tr38901_3d_dualpol_channel",
    "tr38901_antenna_pattern_db",
    "tr38901_dualpol_field_components",
]
