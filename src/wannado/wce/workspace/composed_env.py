"""Forward-Coupled Multi-Block ComposedWirelessEnv Matrix Execution Engine (v1.2)."""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import torch

from wannado.wce.core.adapters import ChannelModuleAdapter, WirelessGymEnvAdapter
from wannado.wce.core.geometry import nearfield_spherical_steering, upa_element_coordinates
from wannado.wce.core.types import ProvisionResult
from wannado.wce.core.units import (
    SPEED_OF_LIGHT_MPS,
    db_to_linear,
    dbm_to_watt,
    doppler_max_hz,
    linear_to_db,
    qfuncinv,
    rayleigh_distance_m,
    thermal_noise_power_watt,
    wavelength_m,
)
from wannado.wce.intake.spec import WCESpec
from wannado.wce.standards.etsi_ieee import itur_p618_rain_attenuation_db, itur_p676_path_absorption_db
from wannado.wce.standards.gpp_38901 import compute_los_probability_tr38901, compute_pathloss_tr38901_db
from wannado.wce.standards.gpp_38901_3d import SpatialConsistencyField
from wannado.wce.standards.gpp_extensions import (
    compute_uav_tr36777_los_probability,
    compute_uav_tr36777_pathloss,
    compute_v2x_tr37885_pathloss,
    interpolate_ntn_params,
)

_EARTH_RADIUS_KM = 6371.0
_MU_EARTH_KM3_S2 = 398600.4418


@dataclass
class ComposedSlotOutput:
    """Per-slot telemetry emitted by ComposedWirelessEnv.step()."""

    slot_index: int
    h_small: torch.Tensor
    h_csi_estimated: torch.Tensor
    mean_snr_db: float
    achievable_rate_bps_hz: float
    throughput_mbps: float
    arrived_bits: float
    queue_bits_remaining: float
    xr_pdb_satisfaction: float
    sensing_crb_range_m: float | None = None
    max_ris_modulus: float = 1.0
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "slot_index": self.slot_index,
            "mean_snr_db": self.mean_snr_db,
            "achievable_rate_bps_hz": self.achievable_rate_bps_hz,
            "throughput_mbps": self.throughput_mbps,
            "arrived_bits": self.arrived_bits,
            "queue_bits_remaining": self.queue_bits_remaining,
            "xr_pdb_satisfaction": self.xr_pdb_satisfaction,
            "sensing_crb_range_m": self.sensing_crb_range_m,
            "max_ris_modulus": self.max_ris_modulus,
            "diagnostics": self.diagnostics,
        }


def _factor_tx_dims(n_tx: int) -> tuple[int, int]:
    r = math.isqrt(max(1, n_tx))
    for d in range(r, 0, -1):
        if n_tx % d == 0:
            return n_tx // d, d
    return n_tx, 1


def _quantize_phase(theta: np.ndarray, bits: int) -> np.ndarray:
    if bits <= 0:
        return theta
    levels = float(2 ** int(bits))
    step = 2.0 * math.pi / levels
    return np.round(theta / step) * step


class ComposedWirelessEnv:
    """Forward-coupled multi-block wireless simulator executing a unified per-slot matrix + MAC chain.

    Couples all 4 compositional layers without future CSI leakage:
    ``Propagation & Topology (TR 38.901 UMa/UMi/RMa/InF | TR 36.777 UAV | TR 38.811 NTN | TR 37.885 V2X | P.676/P.618)``
      -> ``Arrays & Metasurfaces (Full-rank UPA / Spherical Near-Field / True H_d + H_r*Phi*G RIS Cascade)``
      -> ``Causal CSI & Spatial Precoding (Beamforming gain with CSI pointing mismatch, FBL, ISAC CRB, JSR)``
      -> ``Traffic Queue & MAC Evaluation (XR 38.838 PDB, FTP/Pareto backlog)``.
    """

    def __init__(
        self,
        spec: WCESpec,
        *,
        batch_size: int = 32,
        seed: int = 42,
        rain_rate_mm_hr: float = 0.0,
        enable_mu_mimo_interference: bool = False,
        enable_spatial_consistency: bool = False,
        rf_impairments: Any = None,
    ) -> None:
        self.spec = spec
        self.batch_size = max(1, int(batch_size))
        self.seed = int(seed)
        self.rain_rate_mm_hr = max(0.0, float(rain_rate_mm_hr))
        self.enable_mu_mimo_interference = bool(enable_mu_mimo_interference)
        self.enable_spatial_consistency = bool(enable_spatial_consistency)
        self.rf_impairments = rf_impairments
        self._rng = np.random.default_rng(self.seed)
        self._blocks = set(spec.blocks_selected)

        # Shared physical constants across all coupled blocks
        self.fc_hz = spec.carrier_fc_ghz * 1e9
        self.lam = wavelength_m(self.fc_hz)
        self.slot_sec = max(spec.slot_duration_ms * 1e-3, 1e-5)

        # Orbital Doppler adds when NTN is active alongside terminal velocity
        self.fd_hz = doppler_max_hz(spec.max_speed_mps, self.fc_hz)
        if "ntn_leo_satellite" in self._blocks:
            h_km = max(160.0, float(spec.ntn_orbit_altitude_km))
            v_orb = math.sqrt(_MU_EARTH_KM3_S2 / (_EARTH_RADIUS_KM + h_km)) * 1e3
            el_rad = math.radians(float(np.clip(spec.ntn_elevation_deg, 10.0, 90.0)))
            fd_orb = (v_orb * (_EARTH_RADIUS_KM / (_EARTH_RADIUS_KM + h_km)) * math.cos(el_rad) * self.fc_hz) / SPEED_OF_LIGHT_MPS
            self.fd_hz = max(self.fd_hz, fd_orb * 0.15)  # residual compensated satellite Doppler uncertainty

        from scipy import special

        self.temporal_rho = float(np.clip(special.j0(2.0 * math.pi * self.fd_hz * self.slot_sec), -0.999, 0.999))
        self.noise_w = thermal_noise_power_watt(spec.bandwidth_hz, noise_figure_db=spec.noise_figure_db)
        self.tx_w = dbm_to_watt(spec.tx_power_dbm)
        self.rayleigh_zr = rayleigh_distance_m(spec.aperture_m, self.fc_hz)

        # Uncapped TX geometry (supports full XL-MIMO apertures e.g. 256/512 elements)
        self._n_rx = max(1, int(spec.num_rx))
        self._n_tx = max(1, int(spec.num_tx))
        tx_nx, tx_ny = _factor_tx_dims(self._n_tx)
        self._elem_coords = upa_element_coordinates(tx_nx, tx_ny, self.fc_hz)

        self._has_ris = any(b in self._blocks for b in ("passive_ris_cascade", "star_ris", "active_ris_flexible"))
        self._n_ris = max(8, int(spec.num_ris_elements) if int(spec.num_ris_elements) > 0 else 64) if self._has_ris else 0
        self._g_ris: np.ndarray | None = None
        self._hr_ris: np.ndarray | None = None
        self._g_ris_prev: np.ndarray | None = None
        self._hr_ris_prev: np.ndarray | None = None

        self._slot_idx = 0
        self._queue_bits = np.zeros(self.batch_size, dtype=np.float64)
        self._total_arrivals = 0
        self._on_time_arrivals = 0
        self._history: list[ComposedSlotOutput] = []
        self._shadow_db = np.zeros(self.batch_size, dtype=np.float64)
        self._is_los = np.ones(self.batch_size, dtype=bool)
        self._altitudes_m = np.full(self.batch_size, float(spec.uav_altitude_m), dtype=np.float64)
        self._last_eff_snr_lin: np.ndarray | None = None
        self._last_se: np.ndarray | None = None
        self._effective_tx_gain_boost = 1.0
        self._coords_2d = np.zeros((self.batch_size, 2), dtype=np.float64)
        self._spatial_field = (
            SpatialConsistencyField(scenario=self._infer_38901_scenario(), seed=self.seed + 71)
            if self.enable_spatial_consistency
            else None
        )
        self._init_topology()

    def _infer_38901_scenario(self) -> str:
        topo = (self.spec.topology_type or "").lower()
        if "umi" in topo:
            return "UMi"
        if "rma" in topo:
            return "RMa"
        if "inh" in topo or "indoor" in topo:
            return "InH"
        return "UMa"

    def _draw_complex_cn(self, shape: tuple[int, ...]) -> np.ndarray:
        return (
            (self._rng.normal(size=shape) + 1j * self._rng.normal(size=shape)) / math.sqrt(2.0)
        ).astype(np.complex64)

    def _ntn_slant_m(self, elev_deg: float) -> float:
        re = _EARTH_RADIUS_KM
        h_km = max(160.0, float(self.spec.ntn_orbit_altitude_km))
        el = math.radians(float(np.clip(elev_deg, 10.0, 90.0)))
        slant_km = math.sqrt((re * math.sin(el)) ** 2 + 2.0 * re * h_km + h_km ** 2) - re * math.sin(el)
        return slant_km * 1e3

    def _compute_large_scale_from_distances(
        self,
        dists_2d: np.ndarray,
        *,
        altitudes_m: np.ndarray | None = None,
        rain_rate_mm_hr: float | None = None,
        redraw_shadow: bool = False,
        explicit_distance_override: bool = False,
    ) -> np.ndarray:
        dists = np.maximum(np.asarray(dists_2d, dtype=np.float64), 1.0)
        sf_db: float | np.ndarray
        elev_for_atm = 0.0

        if "ntn_leo_satellite" in self._blocks:
            elev_deg = float(np.clip(self.spec.ntn_elevation_deg, 10.0, 90.0))
            elev_for_atm = elev_deg
            slant_m = float(np.mean(dists)) if (explicit_distance_override and float(np.mean(dists)) > 1e4) else self._ntn_slant_m(elev_deg)
            self._dists_2d = np.full(self.batch_size, slant_m, dtype=np.float64)
            ntn = interpolate_ntn_params("urban", elev_deg, fc_ghz=self.spec.carrier_fc_ghz)
            if redraw_shadow:
                self._is_los = self._rng.uniform(0.0, 1.0, size=self.batch_size) < ntn["los_prob"]
            fspl_db = 32.45 + 20.0 * math.log10(max(self.spec.carrier_fc_ghz, 0.1)) + 20.0 * math.log10(slant_m)
            clutter_db = np.where(self._is_los, 0.0, ntn["clutter_loss_db"])
            # Satellite EIRP spot-beam + high-gain VSAT/phased-array link budget compensation (~38 dBi net E2E RX/array offset)
            antenna_comp_db = 38.0
            pl_db = fspl_db + clutter_db - antenna_comp_db + itur_p676_path_absorption_db(
                self.spec.carrier_fc_ghz,
                slant_m,
                elevation_deg=elev_deg,
            )
            sf_db = np.where(self._is_los, ntn["sf_std_db"], ntn["sf_nlos_db"])
            eff_dist_m = np.full(self.batch_size, slant_m, dtype=np.float64)

        elif "uav_air_to_ground" in self._blocks:
            h_arr = np.asarray(altitudes_m if altitudes_m is not None else self._altitudes_m, dtype=np.float64)
            h_scalar = float(np.mean(h_arr))
            if redraw_shadow:
                p_los = compute_uav_tr36777_los_probability("UMa-AV", dists, h_scalar)
                self._is_los = self._rng.uniform(0.0, 1.0, size=self.batch_size) < p_los
            pl_los_arr, sf_los = compute_uav_tr36777_pathloss("UMa-AV", dists, h_scalar, self.spec.carrier_fc_ghz, is_los=True)
            pl_nlos_arr, sf_nlos = compute_uav_tr36777_pathloss("UMa-AV", dists, h_scalar, self.spec.carrier_fc_ghz, is_los=False)
            if h_arr.size > 1 and float(np.ptp(h_arr)) > 1e-6:
                pl_vals: list[float] = []
                sf_vals: list[float] = []
                for idx in range(len(dists)):
                    los_i = bool(self._is_los[idx])
                    p_i, s_i = compute_uav_tr36777_pathloss(
                        "UMa-AV",
                        dists[idx : idx + 1],
                        float(h_arr[idx]),
                        self.spec.carrier_fc_ghz,
                        is_los=los_i,
                    )
                    pl_vals.append(float(p_i[0]))
                    sf_vals.append(float(s_i))
                pl_db = np.asarray(pl_vals, dtype=np.float64)
                sf_db = np.asarray(sf_vals, dtype=np.float64)
            else:
                pl_db = np.where(self._is_los, pl_los_arr, pl_nlos_arr)
                sf_db = np.where(self._is_los, sf_los, sf_nlos)
            elev_for_atm = float(np.degrees(np.arctan2(max(1.5, h_scalar), float(np.mean(dists)))))
            eff_dist_m = np.hypot(dists, h_arr)

        elif "v2x_vehicular_sl" in self._blocks:
            # TR 37.885 probabilistic LOS / NLOSv / NLOS mixture
            p_los_v2x = np.minimum(1.0, 1.05 * np.exp(-0.008 * dists))
            if redraw_shadow:
                u = self._rng.uniform(0.0, 1.0, size=self.batch_size)
                self._is_los = u < p_los_v2x
                self._v2x_nlosv = (u >= p_los_v2x) & (u < p_los_v2x + 0.5 * (1.0 - p_los_v2x))
            pl_l, sf_l = compute_v2x_tr37885_pathloss("highway", dists, self.spec.carrier_fc_ghz, state="LOS")
            pl_nv, sf_nv = compute_v2x_tr37885_pathloss("highway", dists, self.spec.carrier_fc_ghz, state="NLOSv")
            pl_n, sf_n = compute_v2x_tr37885_pathloss("highway", dists, self.spec.carrier_fc_ghz, state="NLOS")
            v_nv = getattr(self, "_v2x_nlosv", np.zeros(self.batch_size, dtype=bool))
            pl_db = np.where(self._is_los, pl_l, np.where(v_nv, pl_nv, pl_n))
            sf_db = np.where(self._is_los, sf_l, np.where(v_nv, sf_nv, sf_n))
            eff_dist_m = dists

        elif "tr38901_inf" in self._blocks:
            if redraw_shadow:
                p_los = compute_los_probability_tr38901("InF-DL", dists)
                self._is_los = self._rng.uniform(0.0, 1.0, size=self.batch_size) < p_los
            pl_db, sf_db = compute_pathloss_tr38901_db(
                "InF-DL",
                dists,
                min(self.spec.carrier_fc_ghz, 100.0),
                is_los=self._is_los,
                h_bs_m=8.0,
            )
            eff_dist_m = dists

        else:
            scen = self._infer_38901_scenario()
            if redraw_shadow:
                p_los = compute_los_probability_tr38901(scen, dists)
                self._is_los = self._rng.uniform(0.0, 1.0, size=self.batch_size) < p_los
            h_bs = 10.0 if scen == "UMi" else (35.0 if scen == "RMa" else (3.0 if scen == "InH" else 25.0))
            pl_db, sf_db = compute_pathloss_tr38901_db(
                scen,
                dists,
                min(self.spec.carrier_fc_ghz, 100.0),
                is_los=self._is_los,
                h_bs_m=h_bs,
                rng=self._rng,
            )
            eff_dist_m = dists

        # Add ITU-R P.676 gas absorption on terrestrial >70 GHz or Near-Field THz (NTN already applied slant P.676 above)
        if "ntn_leo_satellite" not in self._blocks and (
            self.spec.carrier_fc_ghz > 70.0 or "nearfield_xlmimo_thz" in self._blocks
        ):
            pl_db = pl_db + itur_p676_path_absorption_db(self.spec.carrier_fc_ghz, eff_dist_m, water_vapor_g_m3=7.5)

        r_mm = self.rain_rate_mm_hr if rain_rate_mm_hr is None else max(0.0, float(rain_rate_mm_hr))
        if r_mm > 0.0:
            pl_db = pl_db + itur_p618_rain_attenuation_db(
                self.spec.carrier_fc_ghz,
                eff_dist_m,
                rain_rate_mm_hr=r_mm,
                elevation_deg=elev_for_atm,
            )

        if redraw_shadow or (self._spatial_field is not None and explicit_distance_override):
            if self._spatial_field is not None:
                self._shadow_db = self._spatial_field.step_positions(self._coords_2d, std=sf_db)
            else:
                self._shadow_db = self._rng.normal(0.0, sf_db, size=self.batch_size)
        self._pathloss_db = np.asarray(pl_db, dtype=np.float64) + self._shadow_db
        return np.asarray(db_to_linear(-self._pathloss_db), dtype=np.float64)

    def _build_spatial_manifold_los(self, batch_n: int, dists_m: np.ndarray) -> np.ndarray:
        """Build normalized (B, N_rx, N_tx) spherical near-field or planar far-field steering tensor."""
        d_tensor = torch.as_tensor(dists_m, dtype=torch.float32)
        angles = torch.as_tensor(self._rng.uniform(-1.0, 1.0, size=batch_n), dtype=torch.float32)
        if "nearfield_xlmimo_thz" in self._blocks or float(np.min(dists_m)) < self.rayleigh_zr:
            pts = torch.stack([d_tensor * torch.sin(angles), d_tensor * torch.cos(angles), torch.zeros_like(d_tensor)], dim=-1)
            a_tx = nearfield_spherical_steering(self._elem_coords, pts, self.fc_hz)  # (B, Nt) unit-norm
        else:
            tx_idx = torch.arange(self._n_tx, dtype=torch.float32)
            phase = -math.pi * torch.sin(angles).unsqueeze(-1) * tx_idx
            a_tx = torch.complex(torch.cos(phase), torch.sin(phase)) / math.sqrt(self._n_tx)

        rx_angles = torch.as_tensor(self._rng.uniform(-1.2, 1.2, size=batch_n), dtype=torch.float32)
        rx_idx = torch.arange(self._n_rx, dtype=torch.float32)
        rx_phase = -math.pi * torch.sin(rx_angles).unsqueeze(-1) * rx_idx
        a_rx = torch.complex(torch.cos(rx_phase), torch.sin(rx_phase)) / math.sqrt(self._n_rx)
        phi0 = self._rng.uniform(0.0, 2.0 * math.pi, size=(batch_n, 1, 1))
        h_steer = math.sqrt(self._n_rx * self._n_tx) * np.exp(1j * phi0) * torch.einsum(
            "bn,bm->bnm", a_rx.to(torch.complex64), a_tx.to(torch.complex64).conj()
        ).numpy()
        return h_steer.astype(np.complex64)

    def _init_topology(self) -> None:
        if "ntn_leo_satellite" in self._blocks:
            slant_m = self._ntn_slant_m(self.spec.ntn_elevation_deg)
            self._dists_2d = np.full(self.batch_size, slant_m, dtype=np.float64)
        else:
            d_min = max(self.spec.min_tx_rx_distance_m, 2.0)
            d_max = max(d_min + 20.0, 250.0)
            self._dists_2d = self._rng.uniform(d_min, d_max, size=self.batch_size)
        angles_init = self._rng.uniform(0.0, 2.0 * math.pi, size=self.batch_size)
        self._coords_2d = np.column_stack(
            [self._dists_2d * np.cos(angles_init), self._dists_2d * np.sin(angles_init)]
        )
        self._altitudes_m = np.full(self.batch_size, float(self.spec.uav_altitude_m), dtype=np.float64)
        self._large_scale_gain = self._compute_large_scale_from_distances(
            self._dists_2d, altitudes_m=self._altitudes_m, redraw_shadow=True
        )

        # Initialize normalized complex small-scale state E[|H|^2] = 1.0 (Rank-1 LoS + NLoS Rician mix)
        h_scat = self._draw_complex_cn((self.batch_size, self._n_rx, self._n_tx))
        h_los_sp = self._build_spatial_manifold_los(self.batch_size, self._dists_2d)
        self._los_steering_base = h_los_sp
        self._h_nlos_state = h_scat.copy()
        self._k_ratio = np.where(self._is_los, 4.0, 0.0)[:, None, None]
        self._h_state = (
            np.sqrt(self._k_ratio / (self._k_ratio + 1.0)) * h_los_sp
            + np.sqrt(1.0 / (self._k_ratio + 1.0)) * self._h_nlos_state
        ).astype(np.complex64)
        self._h_prev_csi = self._h_state.copy()

        # Two-hop RIS matrix channels G[B, M, Nt] and H_r[B, Nr, M] (when RIS active)
        if self._has_ris and self._n_ris > 0:
            self._g_ris = self._draw_complex_cn((self.batch_size, self._n_ris, self._n_tx))
            self._hr_ris = self._draw_complex_cn((self.batch_size, self._n_rx, self._n_ris))
            self._g_ris_prev = self._g_ris.copy()
            self._hr_ris_prev = self._hr_ris.copy()
        else:
            self._g_ris = None
            self._hr_ris = None
            self._g_ris_prev = None
            self._hr_ris_prev = None

    def reset(self, *, seed: int | None = None) -> ComposedSlotOutput:
        """Reset channel trajectories, queue states, and slot counter deterministically."""
        if seed is not None:
            self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)
        self._slot_idx = 0
        self._queue_bits = np.zeros(self.batch_size, dtype=np.float64)
        self._total_arrivals = 0
        self._on_time_arrivals = 0
        self._history.clear()
        if self._spatial_field is not None:
            self._spatial_field.reset(seed=self.seed + 71)
        self._init_topology()
        return self.step()

    def sample_channel_tensor(self, batch_size: int) -> torch.Tensor:
        """Sample a stateless unit-power (E[|H|^2]=1) complex64 MIMO channel batch without mutating episode queues."""
        b = max(1, int(batch_size))
        h_scat = self._draw_complex_cn((b, self._n_rx, self._n_tx))
        d_sub = np.resize(self._dists_2d, b)
        los_mask = np.resize(self._is_los, b)
        h_los = self._build_spatial_manifold_los(b, d_sub)
        k_ratio = np.where(los_mask, 4.0, 0.0)[:, None, None]
        h_out = (np.sqrt(k_ratio / (k_ratio + 1.0)) * h_los + np.sqrt(1.0 / (k_ratio + 1.0)) * h_scat).astype(np.complex64)
        if self._has_ris and self._n_ris > 0:
            g_s = self._draw_complex_cn((b, self._n_ris, self._n_tx))
            hr_s = self._draw_complex_cn((b, self._n_rx, self._n_ris))
            h_ref = h_out[:, 0, :]
            v_d = h_ref.conj() / np.maximum(np.linalg.norm(h_ref, axis=-1, keepdims=True), 1e-9)
            g_p = np.einsum("bmk,bk->bm", g_s, v_d)
            theta_s = _quantize_phase(-np.angle(hr_s[:, 0, :] * g_p), int(self.spec.ris_phase_bits))
            amp_s = math.sqrt(0.5) if "star_ris" in self._blocks else (2.0 if "active_ris_flexible" in self._blocks else 1.0)
            phi_s = amp_s * np.exp(1j * theta_s)
            ris_base = 0.04 if "active_ris_flexible" in self._blocks else 0.08
            h_out = (h_out + ris_base * np.einsum("bnm,bm,bmk->bnk", hr_s, phi_s, g_s)).astype(np.complex64)
            mean_p = float(np.mean(np.abs(h_out) ** 2))
            if mean_p > 1e-12:
                h_out = (h_out / math.sqrt(mean_p)).astype(np.complex64)
        return torch.from_numpy(h_out)

    def _draw_slot_arrivals_bits(self) -> np.ndarray:
        tm = self.spec.traffic_model.lower()
        if "xr" in tm:
            fps = 90.0 if "90" in tm else 60.0
            period_slots = max(1, round((1.0 / fps) / self.slot_sec))
            if self._slot_idx % period_slots == 0:
                mean_frame_bits = 1.5e7 / fps
                return self._rng.normal(mean_frame_bits, 0.1 * mean_frame_bits, size=self.batch_size).clip(min=1e4)
            return np.zeros(self.batch_size, dtype=np.float64)
        if "pareto" in tm:
            prob = 0.25
            active = self._rng.uniform(0.0, 1.0, size=self.batch_size) < prob
            sizes = (self._rng.pareto(1.5, size=self.batch_size) + 1.0) * 1.6e4
            return np.where(active, np.minimum(sizes, 4e5), 0.0)
        arr = self._rng.poisson(0.15, size=self.batch_size)
        return arr * 6.4e4

    def step(
        self,
        *,
        action: Any = None,
        distances_m: np.ndarray | Sequence[float] | float | None = None,
        tx_pos_3d: np.ndarray | Sequence[float] | None = None,
        rx_pos_3d: np.ndarray | Sequence[float] | None = None,
        positions_3d: tuple[Any, Any] | None = None,
        packet_bits: np.ndarray | Sequence[float] | float | None = None,
        ris_scale: float | None = None,
        power_scale: float | None = None,
        per_user_power_weights: np.ndarray | Sequence[float] | None = None,
        rain_rate_mm_hr: float | None = None,
    ) -> ComposedSlotOutput:
        """Advance physical channel, 2-hop RIS cascade, causal CSI precoding, 6G PHY, and traffic queues by 1 slot."""
        if positions_3d is not None:
            tx_pos_3d, rx_pos_3d = positions_3d

        topology_updated = rain_rate_mm_hr is not None
        explicit_dist = False
        altitudes_override: np.ndarray | None = None
        if tx_pos_3d is not None or rx_pos_3d is not None:
            tx_arr = np.zeros((self.batch_size, 3), dtype=np.float64) if tx_pos_3d is None else np.asarray(tx_pos_3d, dtype=np.float64)
            rx_arr = np.zeros((self.batch_size, 3), dtype=np.float64) if rx_pos_3d is None else np.asarray(rx_pos_3d, dtype=np.float64)
            if tx_arr.ndim == 1:
                tx_arr = np.broadcast_to(tx_arr, (self.batch_size, 3))
            if rx_arr.ndim == 1:
                rx_arr = np.broadcast_to(rx_arr, (self.batch_size, 3))
            diff_3d = rx_arr - tx_arr
            d_horiz = np.hypot(diff_3d[:, 0], diff_3d[:, 1])
            d_3d = np.linalg.norm(diff_3d, axis=-1)
            self._dists_2d = np.maximum(d_horiz, np.maximum(d_3d * 0.25, 1.0))
            self._coords_2d = diff_3d[:, :2].copy()
            altitudes_override = np.maximum(np.abs(diff_3d[:, 2]), 1.5)
            self._altitudes_m = altitudes_override
            topology_updated = True
            explicit_dist = True
        elif distances_m is not None:
            d_arr = np.broadcast_to(np.asarray(distances_m, dtype=np.float64), (self.batch_size,))
            self._dists_2d = np.maximum(d_arr, 1.0)
            cur_ang = np.arctan2(self._coords_2d[:, 1], self._coords_2d[:, 0])
            self._coords_2d = np.column_stack([self._dists_2d * np.cos(cur_ang), self._dists_2d * np.sin(cur_ang)])
            topology_updated = True
            explicit_dist = True

        if topology_updated:
            self._large_scale_gain = self._compute_large_scale_from_distances(
                self._dists_2d,
                altitudes_m=altitudes_override,
                rain_rate_mm_hr=rain_rate_mm_hr,
                redraw_shadow=False,
                explicit_distance_override=explicit_dist,
            )

        act_power_mult = 1.0
        act_ris_mult = 1.0
        act_power_cost = 0.0
        act_label: int | float | str | None = None
        user_pow_weights = np.ones(self.batch_size, dtype=np.float64)
        if per_user_power_weights is not None:
            w_arr = np.broadcast_to(np.asarray(per_user_power_weights, dtype=np.float64), (self.batch_size,)).clip(min=0.05)
            user_pow_weights = w_arr / max(float(np.mean(w_arr)), 1e-9)

        if action is not None:
            if isinstance(action, (int, np.integer)) or (
                isinstance(action, np.ndarray) and action.ndim == 0 and np.issubdtype(action.dtype, np.integer)
            ):
                a_idx = int(action)
                act_label = a_idx
                if a_idx == 0:
                    act_power_mult, act_ris_mult, act_power_cost = 0.5, 0.85, 0.0
                elif a_idx == 1:
                    act_power_mult, act_ris_mult, act_power_cost = 1.0, 1.0, 0.08
                else:
                    act_power_mult, act_ris_mult, act_power_cost = 2.0, 1.35, 0.25
            else:
                a_arr = np.asarray(action, dtype=np.float64).reshape(-1)
                if a_arr.size >= 2:
                    # Continuous multi-user power allocation vector mapped via softmax weighting
                    act_label = f"vec[{a_arr.size}]"
                    tiled = np.resize(a_arr, self.batch_size)
                    exp_w = np.exp(np.clip(tiled - np.max(tiled), -6.0, 6.0))
                    user_pow_weights = (exp_w / np.mean(exp_w)).clip(min=0.05)
                    act_power_cost = 0.08
                else:
                    a_val = float(a_arr[0]) if a_arr.size > 0 else 0.0
                    act_label = a_val
                    act_power_mult = float(np.clip(0.5 + 0.75 * max(-1.0, min(2.0, a_val)) if a_val <= 1.0 else 0.5 * a_val, 0.2, 2.5))
                    act_ris_mult = float(np.clip(0.85 + 0.25 * max(0.0, a_val), 0.75, 1.5))
                    act_power_cost = 0.12 * act_power_mult

        if power_scale is not None:
            act_power_mult *= max(0.0, float(power_scale))
        if ris_scale is not None:
            act_ris_mult *= max(0.0, float(ris_scale))

        # Temporal Clarke J0(2*pi*f_d*dt) autoregressive step on NLoS scatterers while preserving specular LoS manifold
        rho = self.temporal_rho
        if self._slot_idx > 0:
            innov = self._draw_complex_cn((self.batch_size, self._n_rx, self._n_tx))
            self._h_nlos_state = (rho * self._h_nlos_state + math.sqrt(max(0.0, 1.0 - rho * rho)) * innov).astype(np.complex64)
            self._h_state = (
                np.sqrt(self._k_ratio / (self._k_ratio + 1.0)) * self._los_steering_base
                + np.sqrt(1.0 / (self._k_ratio + 1.0)) * self._h_nlos_state
            ).astype(np.complex64)
            if self._g_ris is not None and self._hr_ris is not None:
                in_g = self._draw_complex_cn((self.batch_size, self._n_ris, self._n_tx))
                in_r = self._draw_complex_cn((self.batch_size, self._n_rx, self._n_ris))
                self._g_ris = (rho * self._g_ris + math.sqrt(max(0.0, 1.0 - rho * rho)) * in_g).astype(np.complex64)
                self._hr_ris = (rho * self._hr_ris + math.sqrt(max(0.0, 1.0 - rho * rho)) * in_r).astype(np.complex64)

        h_small_t = torch.from_numpy(self._h_state.copy())

        # Layer 3a (Causal CSI construction BEFORE beamforming/RIS phase design -> never leaks future or beamforming gain!)
        nominal_snr_est = np.maximum(
            (self.tx_w * act_power_mult) * self._large_scale_gain / max(self.noise_w, 1e-30),
            0.1,
        )
        if self.spec.csi_assumption == "ideal_genie_baseline":
            h_csi_d = self._h_state.copy()
            g_csi = self._g_ris.copy() if self._g_ris is not None else None
            hr_csi = self._hr_ris.copy() if self._hr_ris is not None else None
        elif self.spec.csi_assumption == "delayed_feedback":
            h_csi_d = self._h_prev_csi.copy()
            g_csi = self._g_ris_prev.copy() if self._g_ris_prev is not None else None
            hr_csi = self._hr_ris_prev.copy() if self._hr_ris_prev is not None else None
        else:
            est_var = 1.0 / (1.0 + nominal_snr_est)
            scale_sig = np.sqrt(np.maximum(1.0 - est_var, 0.1))[:, None, None]
            scale_err = np.sqrt(est_var)[:, None, None]
            h_csi_d = (scale_sig * self._h_state + scale_err * self._draw_complex_cn(self._h_state.shape)).astype(np.complex64)
            if self._g_ris is not None and self._hr_ris is not None:
                g_csi = (scale_sig * self._g_ris + scale_err * self._draw_complex_cn(self._g_ris.shape)).astype(np.complex64)
                hr_csi = (scale_sig * self._hr_ris + scale_err * self._draw_complex_cn(self._hr_ris.shape)).astype(np.complex64)
            else:
                g_csi, hr_csi = None, None

        self._h_prev_csi = self._h_state.copy()
        if self._g_ris is not None and self._hr_ris is not None:
            self._g_ris_prev = self._g_ris.copy()
            self._hr_ris_prev = self._hr_ris.copy()

        # Layer 2: True Two-Hop Matrix Cascaded Channel H_eff = H_d + eta * H_r @ diag(Phi) @ G
        max_ris_mod = 1.0
        active_ris_noise_w = np.zeros(self.batch_size, dtype=np.float64)
        if self._has_ris and self._g_ris is not None and self._hr_ris is not None and g_csi is not None and hr_csi is not None:
            # Coherent cophasing along (H_r @ Phi @ G) @ v_dir where v_dir ~ conj(h_dir_ref):
            # Target positive real inner product per RIS atom -> theta_m = -angle(hr_m * (g_m @ v_dir)).
            h_dir_ref = h_csi_d[:, 0, :]  # (B, Nt)
            hr_row0 = hr_csi[:, 0, :]  # (B, M)
            v_dir = h_dir_ref.conj() / np.maximum(np.linalg.norm(h_dir_ref, axis=-1, keepdims=True), 1e-9)
            g_proj = np.einsum("bmk,bk->bm", g_csi, v_dir)
            opt_theta = -np.angle(hr_row0 * g_proj)
            phi_theta = _quantize_phase(opt_theta, int(self.spec.ris_phase_bits))

            # Link-budget calibrated cascade coupling scale: each passive RIS element reflects a small
            # normalized fraction of the direct link so M meta-atoms provide O(sqrt(M))..O(M) coherent array gain.
            ris_scale_base = 0.08
            if "active_ris_flexible" in self._blocks:
                # Active RIS gain calibrated with hop-2 amplified thermal noise floor (Zhang 2023)
                ris_amp = 2.0 * math.sqrt(max(act_ris_mult, 0.01))
                phi_vec = ris_amp * np.exp(1j * phi_theta)
                max_ris_mod = float(np.max(np.abs(phi_vec)))
                cascade_true = (0.65 * ris_scale_base) * np.einsum("bnm,bm,bmk->bnk", self._hr_ris, phi_vec, self._g_ris)
                cascade_est = (0.65 * ris_scale_base) * np.einsum("bnm,bm,bmk->bnk", hr_csi, phi_vec, g_csi)
                hr_energy = np.mean(np.abs(self._hr_ris) ** 2, axis=(1, 2))
                active_ris_noise_w = max(ris_amp ** 2 - 1.0, 0.0) * 0.12 * self.noise_w * hr_energy
            elif "star_ris" in self._blocks:
                # Passive STAR-RIS energy conservation: |beta_t|^2 + |beta_r|^2 <= 1 -> |phi_m| <= sqrt(0.5) <= 1.0
                passive_atten = min(max(act_ris_mult, 0.0), 1.0)
                amp_mod = math.sqrt(0.5) * passive_atten
                phi_vec = amp_mod * np.exp(1j * phi_theta)
                max_ris_mod = float(np.max(np.abs(phi_vec)))
                cascade_true = ris_scale_base * np.einsum("bnm,bm,bmk->bnk", self._hr_ris, phi_vec, self._g_ris)
                cascade_est = ris_scale_base * np.einsum("bnm,bm,bmk->bnk", hr_csi, phi_vec, g_csi)
            else:
                # Passive RIS energy conservation: |phi_m| <= 1.0 strictly (never amplify above unity)
                passive_atten = min(max(act_ris_mult, 0.0), 1.0)
                phi_vec = passive_atten * np.exp(1j * phi_theta)
                max_ris_mod = float(np.max(np.abs(phi_vec)))
                cascade_true = ris_scale_base * np.einsum("bnm,bm,bmk->bnk", self._hr_ris, phi_vec, self._g_ris)
                cascade_est = ris_scale_base * np.einsum("bnm,bm,bmk->bnk", hr_csi, phi_vec, g_csi)

            h_eff_true = self._h_state + cascade_true
            h_eff_est = h_csi_d + cascade_est
        else:
            h_eff_true = self._h_state
            h_eff_est = h_csi_d

        # Layer 3b: Real Matrix Precoding & Combining driven by Causal CSI (h_eff_est), evaluated on h_eff_true!
        mui_rx_w = np.zeros(self.batch_size, dtype=np.float64)
        mu_mimo_active = bool(self.enable_mu_mimo_interference and self.batch_size >= 2)
        if mu_mimo_active:
            # Multi-User Regularized Zero-Forcing (RZF) precoding across co-scheduled UEs in the batch
            if self._n_rx > 1:
                u_rx = h_eff_est[:, :, 0]
                u_norm = np.linalg.norm(u_rx, axis=-1, keepdims=True)
                u_rx = np.where(u_norm > 1e-9, u_rx / np.maximum(u_norm, 1e-9), 1.0 / math.sqrt(self._n_rx))
                h_mu_est = np.einsum("bn,bnk->bk", np.conj(u_rx), h_eff_est)  # (K, Nt)
                h_mu_true = np.einsum("bn,bnk->bk", np.conj(u_rx), h_eff_true)  # (K, Nt)
            else:
                h_mu_est = h_eff_est[:, 0, :]
                h_mu_true = h_eff_true[:, 0, :]

            if self._n_tx > 1:
                reg_alpha = float(self.batch_size) / max(float(np.median(nominal_snr_est)), 0.1)
                gram_mu = np.conj(h_mu_est).T @ h_mu_est + reg_alpha * np.eye(self._n_tx, dtype=np.complex64)
                w_rzf = np.linalg.solve(gram_mu, np.conj(h_mu_est).T)  # (Nt, K)
                col_norms = np.linalg.norm(w_rzf, axis=0, keepdims=True)
                w_mat = w_rzf / np.maximum(col_norms, 1e-9)
            else:
                w_mat = np.ones((1, self.batch_size), dtype=np.complex64)

            p_user = (self.tx_w * act_power_mult / float(self.batch_size)) * user_pow_weights  # (K,)
            coupling_sq = np.abs(h_mu_true @ w_mat) ** 2  # (K, K): row k = UE k, col j = beam j
            desired_gain = np.diag(coupling_sq)
            raw_rx_w = p_user * self._large_scale_gain * desired_gain
            # True inter-user co-channel spatial interference from all j != k
            total_coupled = coupling_sq @ p_user
            interf_gain_w = np.maximum(total_coupled - p_user * desired_gain, 0.0)
            mui_rx_w = self._large_scale_gain * interf_gain_w
        else:
            # Single-user / isolated link SVD top right-singular precoder & MRC combining
            if self._n_tx > 1:
                # Estimate dominant eigenmode beamformer v_tx from H_eff_est (no future/oracle leakage)
                gram_est = np.einsum("bnk,bnl->bkl", np.conj(h_eff_est), h_eff_est)  # (B, Nt, Nt)
                v = h_eff_est[:, 0, :].conj()  # (B, Nt)
                v_norm = np.linalg.norm(v, axis=-1, keepdims=True)
                v = np.where(v_norm > 1e-9, v / np.maximum(v_norm, 1e-9), 1.0 / math.sqrt(self._n_tx))
                for _ in range(3):
                    v = np.einsum("bkl,bl->bk", gram_est, v)
                    vn = np.linalg.norm(v, axis=-1, keepdims=True)
                    v = np.where(vn > 1e-9, v / np.maximum(vn, 1e-9), 1.0 / math.sqrt(self._n_tx))
                bf_vec = np.einsum("bnk,bk->bn", h_eff_true, v)  # (B, Nr)
                eff_channel_gain = np.sum(np.abs(bf_vec) ** 2, axis=-1)
            else:
                eff_channel_gain = np.sum(np.abs(h_eff_true[:, :, 0]) ** 2, axis=-1)

            raw_rx_w = (self.tx_w * act_power_mult * user_pow_weights) * self._large_scale_gain * eff_channel_gain
        if "isac_jsac_channel" in self._blocks:
            comm_frac = 0.6
            rx_w = raw_rx_w * comm_frac
            total_sense_noise = np.maximum(self.noise_w + active_ris_noise_w, 1e-30)
            # True two-way radar equation: P_sense * G_tx * G_rx * lambda^2 * sigma_RCS / ((4*pi)^3 * R^4)
            p_sense_w = (self.tx_w * act_power_mult * user_pow_weights) * (1.0 - comm_frac)
            rcs_m2 = 10.0
            radar_gain = (
                float(self._n_tx)
                * float(self._n_rx)
                * (self.lam ** 2)
                * rcs_m2
                / (((4.0 * math.pi) ** 3) * np.maximum(self._dists_2d, 1.0) ** 4)
            )
            sensing_snr = np.maximum(p_sense_w * radar_gain / total_sense_noise, 1e-9)
            bw = max(self.spec.bandwidth_hz, 1.0)
            b_rms2 = (bw ** 2) / 12.0
            round_trip_factor = 1.0 if "bistatic" in (self.spec.topology_type or "").lower() else 2.0
            crb_range_m = float(
                np.median(
                    SPEED_OF_LIGHT_MPS
                    / (
                        round_trip_factor
                        * 2.0
                        * math.sqrt(2.0)
                        * math.pi
                        * np.sqrt(np.maximum(b_rms2 * sensing_snr, 1e-12))
                    )
                )
            )
        else:
            rx_w = raw_rx_w
            crb_range_m = None

        jam_w = np.zeros(self.batch_size, dtype=np.float64)
        if "pls_covert_antijamming" in self._blocks:
            # Stochastic jammer fading power around -6 dB nominal JSR
            h_jam = np.abs(self._draw_complex_cn((self.batch_size,)).reshape(-1)) ** 2
            jam_w = np.median(rx_w) * 0.25 * h_jam

        eff_snr_post = rx_w / np.maximum(self.noise_w + active_ris_noise_w + jam_w + mui_rx_w, 1e-30)

        # Achievable Rate: Shannon or Polyanskiy FBL dispersion if HRLLC enabled (strictly <= Shannon)
        shannon_se = np.log2(1.0 + eff_snr_post)
        if "hrllc_fbl_urllc" in self._blocks:
            v_disp = (1.0 - (1.0 + eff_snr_post) ** (-2.0)) * (math.log2(math.e) ** 2)
            n_blk = 200.0
            se = np.clip(shannon_se - np.sqrt(np.maximum(v_disp, 0.0) / n_blk) * qfuncinv(1e-5), 0.0, shannon_se)
        else:
            se = shannon_se

        se_sum = float(np.sum(se))
        se_sq_sum = float(np.sum(se ** 2))
        jain_fair = (se_sum ** 2) / max(float(self.batch_size) * se_sq_sum, 1e-12)
        mean_mui_db = float(np.mean(linear_to_db(np.maximum(mui_rx_w, 1e-30)))) if mu_mimo_active else None

        # Layer 4: Packet arrivals & queue draining
        if packet_bits is not None:
            arr_bits = np.broadcast_to(np.asarray(packet_bits, dtype=np.float64), (self.batch_size,)).clip(min=0.0)
        else:
            arr_bits = self._draw_slot_arrivals_bits()
        cap_bits_per_slot = se * self.spec.bandwidth_hz * self.slot_sec
        self._queue_bits = np.maximum(self._queue_bits + arr_bits - cap_bits_per_slot, 0.0)
        self._last_eff_snr_lin = eff_snr_post.copy()
        self._last_se = se.copy()

        if np.any(arr_bits > 0):
            n_new = int(np.sum(arr_bits > 0))
            sat = int(np.sum((arr_bits > 0) & (self._queue_bits <= 0.5 * np.maximum(cap_bits_per_slot, 1e3))))
            self._total_arrivals += n_new
            self._on_time_arrivals += sat

        pdb_sat = float(self._on_time_arrivals / self._total_arrivals) if self._total_arrivals > 0 else 1.0
        out = ComposedSlotOutput(
            slot_index=self._slot_idx,
            h_small=h_small_t,
            h_csi_estimated=torch.from_numpy(h_eff_est.astype(np.complex64)),
            mean_snr_db=float(np.mean(linear_to_db(eff_snr_post))),
            achievable_rate_bps_hz=float(np.mean(se)),
            throughput_mbps=float(np.mean(se) * (self.spec.bandwidth_hz / 1e6)),
            arrived_bits=float(np.sum(arr_bits)),
            queue_bits_remaining=float(np.sum(self._queue_bits)),
            xr_pdb_satisfaction=pdb_sat,
            sensing_crb_range_m=crb_range_m,
            max_ris_modulus=max_ris_mod,
            diagnostics={
                "temporal_rho_target": self.temporal_rho,
                "is_passive_ris": ("passive_ris_cascade" in self._blocks or "star_ris" in self._blocks)
                and "active_ris_flexible" not in self._blocks,
                "max_ris_modulus": max_ris_mod,
                "has_future_csi_leakage": False,
                "rayleigh_distance_m": self.rayleigh_zr,
                "mean_distance_m": float(np.mean(self._dists_2d)),
                "los_fraction": float(np.mean(self._is_los)),
                "num_ris_elements": self._n_ris,
                "power_scale": act_power_mult,
                "action_applied": act_label,
                "action_power_cost": act_power_cost,
                "mu_mimo_interference_enabled": mu_mimo_active,
                "spatial_consistency_enabled": self.enable_spatial_consistency,
                "mean_inter_user_interference_db": mean_mui_db,
                "mu_mimo_sum_rate_bps_hz": se_sum if mu_mimo_active else None,
                "jain_fairness": jain_fair,
            },
        )
        self._history.append(out)
        self._slot_idx += 1
        return out

    def run_trajectory(self, num_slots: int = 20, *, seed: int | None = None) -> list[ComposedSlotOutput]:
        """Run a multi-slot deterministic wireless trajectory and return per-slot outputs."""
        self.reset(seed=seed)
        for _ in range(max(0, num_slots - 1)):
            self.step()
        return list(self._history)

    def to_rigor_result(self, num_slots: int = 20, *, seed: int | None = None) -> ProvisionResult:
        """Execute trajectory and package as a Stage 3 audit-ready ProvisionResult."""
        t0 = time.perf_counter()
        traj = self.run_trajectory(num_slots=num_slots, seed=seed)
        h_last = traj[-1].h_small
        rates = np.array([s.achievable_rate_bps_hz for s in traj], dtype=np.float64)
        snrs = np.array([s.mean_snr_db for s in traj], dtype=np.float64)

        if len(traj) >= 4:
            h_series = torch.stack([s.h_small for s in traj], dim=0)
            num = torch.sum(h_series[1:] * h_series[:-1].conj()).real.item()
            den = math.sqrt(float(torch.sum(h_series[1:].abs().pow(2)).item() * torch.sum(h_series[:-1].abs().pow(2)).item())) + 1e-12
            emp_rho = num / den
            # Theoretical lag-1 autocorrelation of Rician mix (static specular LoS + AR(1) Clarke NLoS)
            los_w = float(np.mean(self._k_ratio / (self._k_ratio + 1.0)))
            theo_rho = los_w + (1.0 - los_w) * self.temporal_rho
            lag_gap = abs(emp_rho - theo_rho)
        else:
            emp_rho = self.temporal_rho
            lag_gap = 0.0

        metrics: dict[str, float] = {
            "mean_spectral_efficiency_bps_hz": float(np.mean(rates)),
            "median_snr_db": float(np.median(snrs)),
            "mean_throughput_mbps": float(np.mean([s.throughput_mbps for s in traj])),
            "final_queue_bits": float(traj[-1].queue_bits_remaining),
            "xr_pdb_satisfaction": float(traj[-1].xr_pdb_satisfaction),
            "mean_channel_power": float(torch.mean(h_last.abs().pow(2)).item()),
            "empirical_lag1_autocorr": float(emp_rho),
        }
        if traj[-1].sensing_crb_range_m is not None:
            metrics["sensing_crb_range_m"] = float(traj[-1].sensing_crb_range_m)

        return ProvisionResult(
            block_id="composed_pipeline",
            seed=self.seed,
            elapsed_seconds=time.perf_counter() - t0,
            config=asdict(self.spec),
            tensors={"h_small": h_last, "snr_db": torch.from_numpy(snrs).to(torch.float32)},
            metrics=metrics,
            diagnostics={
                "has_future_csi_leakage": False,
                "lag1_autocorr_gap": lag_gap,
                "is_passive_ris": ("passive_ris_cascade" in self._blocks or "star_ris" in self._blocks)
                and "active_ris_flexible" not in self._blocks,
                "max_ris_modulus": traj[-1].max_ris_modulus,
                "min_tx_rx_distance_m": self.spec.min_tx_rx_distance_m,
                "rayleigh_distance_m": self.rayleigh_zr,
                "assumed_farfield": False,
            },
            parameter_provenance=list(self.spec.parameter_provenance),
            mechanisms_used=[],
            standard_ids_used=list(self.spec.standard_ids),
        )

    def as_channel_module(
        self,
        *,
        snr_db: float = 15.0,
        stateless: bool = True,
        rf_impairments: Any = None,
    ) -> ChannelModuleAdapter:
        """Expose composed complex64 small-scale channel into wannado-adv / Semantic AI training.

        Defaults to ``stateless=True`` so arbitrary training batch sizes (e.g. 64/128/256)
        work without mutating MAC queues or crashing on fixed ``self.batch_size``.
        """
        rf_mod = self.rf_impairments if rf_impairments is None else rf_impairments
        if stateless:
            return ChannelModuleAdapter(self.sample_channel_tensor, snr_db=snr_db, add_noise=True, rf_impairments=rf_mod)
        return ChannelModuleAdapter(lambda b: self.step().h_small[:b], snr_db=snr_db, add_noise=True, rf_impairments=rf_mod)

    def as_gym_env(
        self,
        *,
        max_steps: int = 50,
        continuous_actions: bool = False,
        multi_user: bool = False,
    ) -> WirelessGymEnvAdapter:
        """Expose causal wireless state to wannado-drl / Gymnasium RL agents.

        - ``multi_user=False`` (default): 4-dim global summary state & 3 discrete (or 1 continuous) action.
        - ``multi_user=True``: ``4 * batch_size + 4`` per-user state (estimated gain, SNR dB, queue backlog,
          LoS state + global metrics) and ``batch_size`` continuous power allocation actions.
        """
        step_state = {"count": 0}
        obs_dim = (4 * self.batch_size + 4) if multi_user else 4
        use_cont = bool(continuous_actions or multi_user)
        action_dim = self.batch_size if multi_user else (1 if use_cont else 3)

        def _obs(out: ComposedSlotOutput) -> np.ndarray:
            summary = np.array(
                [out.mean_snr_db, out.achievable_rate_bps_hz, math.log1p(out.queue_bits_remaining), out.xr_pdb_satisfaction],
                dtype=np.float32,
            )
            if not multi_user:
                return summary
            est_gain = np.log1p(np.mean(np.abs(out.h_csi_estimated.numpy()) ** 2, axis=(1, 2)))
            snr_u = linear_to_db(self._last_eff_snr_lin) if self._last_eff_snr_lin is not None else np.zeros(self.batch_size)
            q_u = np.log1p(self._queue_bits)
            los_u = self._is_los.astype(np.float64)
            per_user = np.concatenate([est_gain, snr_u, q_u, los_u]).astype(np.float32)
            return np.concatenate([per_user, summary])

        def _reset(seed: int | None) -> tuple[np.ndarray, dict[str, Any]]:
            step_state["count"] = 0
            out = self.reset(seed=seed)
            return _obs(out), out.to_dict()

        def _step(action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
            step_state["count"] += 1
            out = self.step(action=action)
            power_cost = float(out.diagnostics.get("action_power_cost", 0.0))
            reward = float(out.achievable_rate_bps_hz - 0.05 * math.log1p(out.queue_bits_remaining) - power_cost)
            trunc = step_state["count"] >= max_steps
            return _obs(out), reward, False, trunc, out.to_dict()

        return WirelessGymEnvAdapter(
            obs_dim=obs_dim,
            action_dim=action_dim,
            discrete_actions=not use_cont,
            reset_fn=_reset,
            step_fn=_step,
            max_steps=max_steps,
        )
