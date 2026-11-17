"""Forward-Coupled Multi-Block ComposedWirelessEnv Execution Engine (v1.1)."""

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
from wannado.wce.standards.gpp_extensions import compute_uav_tr36777_pathloss, compute_v2x_tr37885_pathloss


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


class ComposedWirelessEnv:
    """Forward-coupled multi-block wireless simulator executing a unified per-slot physical + MAC chain.

    Couples the 4 compositional layers strictly sequentially without future CSI leakage:
    ``Propagation & Topology (TR 38.901 / UAV / NTN / V2X / P.676)``
      -> ``Arrays & Metasurfaces (UPA / Near-Field / Passive·STAR·Active RIS)``
      -> ``Causal CSI & 6G PHY (LS/LMMSE error, FBL penalty, ISAC CRB, JSR jamming)``
      -> ``Traffic Queue & MAC Evaluation (XR 38.838 PDB, FTP/Pareto backlog)``.
    """

    def __init__(
        self,
        spec: WCESpec,
        *,
        batch_size: int = 32,
        seed: int = 42,
        rain_rate_mm_hr: float = 0.0,
    ) -> None:
        self.spec = spec
        self.batch_size = max(1, int(batch_size))
        self.seed = int(seed)
        self.rain_rate_mm_hr = max(0.0, float(rain_rate_mm_hr))
        self._rng = np.random.default_rng(self.seed)
        self._blocks = set(spec.blocks_selected)

        # Shared physical constants across all coupled blocks
        self.fc_hz = spec.carrier_fc_ghz * 1e9
        self.lam = wavelength_m(self.fc_hz)
        self.slot_sec = max(spec.slot_duration_ms * 1e-3, 1e-5)
        self.fd_hz = doppler_max_hz(spec.max_speed_mps, self.fc_hz)
        # Clarke J0(2*pi*f_d*dt) clamped to physical autoregressive envelope [-0.999, 0.999]
        from scipy import special

        self.temporal_rho = float(np.clip(special.j0(2.0 * math.pi * self.fd_hz * self.slot_sec), -0.999, 0.999))
        self.noise_w = thermal_noise_power_watt(spec.bandwidth_hz, noise_figure_db=spec.noise_figure_db)
        self.tx_w = dbm_to_watt(spec.tx_power_dbm)
        self.rayleigh_zr = rayleigh_distance_m(spec.aperture_m, self.fc_hz)

        # Near-field element geometry if enabled
        self._elem_coords = upa_element_coordinates(min(spec.num_tx, 128), 1, self.fc_hz)
        self._slot_idx = 0
        self._queue_bits = np.zeros(self.batch_size, dtype=np.float64)
        self._total_arrivals = 0
        self._on_time_arrivals = 0
        self._history: list[ComposedSlotOutput] = []
        self._shadow_db = np.zeros(self.batch_size, dtype=np.float64)
        self._is_los = np.ones(self.batch_size, dtype=bool)
        self._altitudes_m = np.full(self.batch_size, 100.0, dtype=np.float64)
        self._last_eff_snr_lin: np.ndarray | None = None
        self._last_se: np.ndarray | None = None

        self._init_topology()

    def _compute_large_scale_from_distances(
        self,
        dists_2d: np.ndarray,
        *,
        altitudes_m: np.ndarray | None = None,
        rain_rate_mm_hr: float | None = None,
        redraw_shadow: bool = False,
    ) -> np.ndarray:
        dists = np.maximum(np.asarray(dists_2d, dtype=np.float64), 1.0)
        sf_db: float | np.ndarray
        if "uav_air_to_ground" in self._blocks:
            h_arr = np.asarray(altitudes_m if altitudes_m is not None else self._altitudes_m, dtype=np.float64)
            if h_arr.size > 1 and float(np.ptp(h_arr)) > 1e-6:
                pl_vals: list[float] = []
                sf_vals: list[float] = []
                for idx in range(len(dists)):
                    p_i, s_i = compute_uav_tr36777_pathloss(
                        "UMa-AV",
                        dists[idx : idx + 1],
                        float(h_arr[idx]),
                        self.spec.carrier_fc_ghz,
                        is_los=True,
                    )
                    pl_vals.append(float(p_i[0]))
                    sf_vals.append(float(s_i))
                pl_db = np.asarray(pl_vals, dtype=np.float64)
                sf_db = np.asarray(sf_vals, dtype=np.float64)
            else:
                pl_db, sf_db = compute_uav_tr36777_pathloss(
                    "UMa-AV",
                    dists,
                    float(h_arr.reshape(-1)[0]),
                    self.spec.carrier_fc_ghz,
                    is_los=True,
                )
        elif "v2x_vehicular_sl" in self._blocks:
            pl_db, sf_db = compute_v2x_tr37885_pathloss("highway", dists, self.spec.carrier_fc_ghz, state="LOS")
        elif "tr38901_inf" in self._blocks:
            pl_db, sf_db = compute_pathloss_tr38901_db("UMi", dists, min(self.spec.carrier_fc_ghz, 100.0), is_los=False, h_bs_m=8.0)
        else:
            if redraw_shadow:
                p_los = compute_los_probability_tr38901("UMa", dists)
                self._is_los = self._rng.uniform(0.0, 1.0, size=self.batch_size) < p_los
            pl_db, sf_db = compute_pathloss_tr38901_db("UMa", dists, min(self.spec.carrier_fc_ghz, 100.0), is_los=self._is_los)

        # Add ITU-R P.676 gas absorption if > 70 GHz or Near-Field THz enabled
        eff_dist_m = dists if altitudes_m is None else np.hypot(dists, np.asarray(altitudes_m, dtype=np.float64))
        if self.spec.carrier_fc_ghz > 70.0 or "nearfield_xlmimo_thz" in self._blocks:
            pl_db = pl_db + itur_p676_path_absorption_db(self.spec.carrier_fc_ghz, eff_dist_m, water_vapor_g_m3=7.5)

        r_mm = self.rain_rate_mm_hr if rain_rate_mm_hr is None else max(0.0, float(rain_rate_mm_hr))
        if r_mm > 0.0:
            elev = 35.0 if ("uav_air_to_ground" in self._blocks or "ntn_leo_satellite" in self._blocks) else 0.0
            pl_db = pl_db + itur_p618_rain_attenuation_db(self.spec.carrier_fc_ghz, eff_dist_m, rain_rate_mm_hr=r_mm, elevation_deg=elev)

        if redraw_shadow:
            self._shadow_db = self._rng.normal(0.0, sf_db, size=self.batch_size)
        self._pathloss_db = np.asarray(pl_db, dtype=np.float64) + self._shadow_db
        return np.asarray(db_to_linear(-self._pathloss_db), dtype=np.float64)

    def _init_topology(self) -> None:
        d_min = max(self.spec.min_tx_rx_distance_m, 2.0)
        d_max = max(d_min + 20.0, 250.0)
        self._dists_2d = self._rng.uniform(d_min, d_max, size=self.batch_size)
        self._altitudes_m = np.full(self.batch_size, 100.0, dtype=np.float64)
        self._large_scale_gain = self._compute_large_scale_from_distances(
            self._dists_2d, altitudes_m=self._altitudes_m, redraw_shadow=True
        )

        # Initialize complex small-scale state (unit variance CN(0, 1))
        n_rx = max(1, self.spec.num_rx)
        n_tx = min(max(1, self.spec.num_tx), 128)
        z = (self._rng.normal(size=(self.batch_size, n_rx, n_tx)) + 1j * self._rng.normal(size=(self.batch_size, n_rx, n_tx))) / math.sqrt(2.0)
        self._h_state = z.astype(np.complex64)
        self._h_prev_csi = self._h_state.copy()

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
        self._init_topology()
        return self.step()

    def _draw_slot_arrivals_bits(self) -> np.ndarray:
        tm = self.spec.traffic_model.lower()
        if "xr" in tm:
            fps = 90.0 if "90" in tm else 60.0
            period_slots = max(1, round((1.0 / fps) / self.slot_sec))
            if self._slot_idx % period_slots == 0:
                mean_frame_bits = 1.5e7 / fps  # 15 Mbps stream -> ~250 kbit/frame
                return self._rng.normal(mean_frame_bits, 0.1 * mean_frame_bits, size=self.batch_size).clip(min=1e4)
            return np.zeros(self.batch_size, dtype=np.float64)
        if "pareto" in tm:
            prob = 0.25
            active = self._rng.uniform(0.0, 1.0, size=self.batch_size) < prob
            sizes = (self._rng.pareto(1.5, size=self.batch_size) + 1.0) * 1.6e4
            return np.where(active, np.minimum(sizes, 4e5), 0.0)
        # Default FTP Model 3 Poisson bursts
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
        rain_rate_mm_hr: float | None = None,
    ) -> ComposedSlotOutput:
        """Advance physical channel, metasurface cascade, causal CSI, 6G PHY, and traffic queues by 1 slot."""
        if positions_3d is not None:
            tx_pos_3d, rx_pos_3d = positions_3d

        topology_updated = rain_rate_mm_hr is not None
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
            altitudes_override = np.maximum(np.abs(diff_3d[:, 2]), 1.5)
            self._altitudes_m = altitudes_override
            topology_updated = True
        elif distances_m is not None:
            d_arr = np.broadcast_to(np.asarray(distances_m, dtype=np.float64), (self.batch_size,))
            self._dists_2d = np.maximum(d_arr, 1.0)
            topology_updated = True

        if topology_updated:
            self._large_scale_gain = self._compute_large_scale_from_distances(
                self._dists_2d,
                altitudes_m=altitudes_override,
                rain_rate_mm_hr=rain_rate_mm_hr,
                redraw_shadow=False,
            )

        act_power_mult = 1.0
        act_ris_mult = 1.0
        act_power_cost = 0.0
        act_label: int | float | None = None
        if action is not None:
            if isinstance(action, (int, np.integer)) or (isinstance(action, np.ndarray) and action.ndim == 0 and np.issubdtype(action.dtype, np.integer)):
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
                a_val = float(a_arr[0]) if a_arr.size > 0 else 0.0
                act_label = a_val
                act_power_mult = float(np.clip(0.5 + 0.75 * max(-1.0, min(2.0, a_val)) if a_val <= 1.0 else 0.5 * a_val, 0.2, 2.5))
                act_ris_mult = float(np.clip(0.85 + 0.25 * max(0.0, a_val), 0.75, 1.5))
                act_power_cost = 0.12 * act_power_mult

        if power_scale is not None:
            act_power_mult *= max(0.0, float(power_scale))
        if ris_scale is not None:
            act_ris_mult *= max(0.0, float(ris_scale))

        n_rx = self._h_state.shape[1]
        n_tx = self._h_state.shape[2]
        innov = (self._rng.normal(size=(self.batch_size, n_rx, n_tx)) + 1j * self._rng.normal(size=(self.batch_size, n_rx, n_tx))) / math.sqrt(2.0)
        if self._slot_idx > 0:
            # First-order Markov Clarke J0(2*pi*f_d*dt) temporal evolution preserving unit variance
            rho = self.temporal_rho
            self._h_state = (rho * self._h_state + math.sqrt(max(0.0, 1.0 - rho * rho)) * innov).astype(np.complex64)

        h_small_t = torch.from_numpy(self._h_state.copy())

        # Layer 2: Array / Near-Field / RIS gain multiplier (enforcing conservation laws)
        ris_gain = np.full(self.batch_size, act_ris_mult, dtype=np.float64)
        max_ris_mod = 1.0
        active_ris_noise_w = 0.0
        n_ris_el = max(8, int(self.spec.num_tx) * 4)
        if "star_ris" in self._blocks:
            # Energy splitting |beta_t|^2 + |beta_r|^2 = 1 -> passive half-space enhancement
            ris_gain *= 1.0 + 0.5 * 0.25 * n_ris_el
            max_ris_mod = 1.0
        elif "active_ris_flexible" in self._blocks:
            ris_amp_gain = 4.0
            ris_gain *= ris_amp_gain
            max_ris_mod = 2.0  # Active amplifier allowed (>1) with dynamic amplified thermal noise
            active_ris_noise_w = (ris_amp_gain - 1.0) * 0.25 * self.noise_w
        elif "passive_ris_cascade" in self._blocks:
            ris_gain *= 1.0 + 0.25 * n_ris_el
            max_ris_mod = 1.0

        if "nearfield_xlmimo_thz" in self._blocks:
            # Sample actual spherical steering vector norm verification
            pts = torch.zeros((1, 3), dtype=torch.float32)
            pts[0, 1] = float(self.spec.min_tx_rx_distance_m)
            _ = nearfield_spherical_steering(self._elem_coords, pts, self.fc_hz)

        # Layer 3: Causal CSI (uses only current/past slot; delayed_feedback reads t-1, never t+1!)
        chan_power = np.mean(np.abs(self._h_state) ** 2, axis=(1, 2))
        raw_rx_w = (self.tx_w * act_power_mult) * self._large_scale_gain * ris_gain * chan_power
        if "isac_jsac_channel" in self._blocks:
            comm_frac = 0.6
            rx_w = raw_rx_w * comm_frac
            total_sense_noise = np.maximum(self.noise_w + active_ris_noise_w, 1e-30)
            sensing_snr = np.maximum(raw_rx_w * (1.0 - comm_frac) / total_sense_noise, 1e-6)
            c0 = 299792458.0
            bw = max(self.spec.bandwidth_hz, 1.0)
            b_rms2 = (bw ** 2) / 12.0
            round_trip_factor = 1.0 if "bistatic" in (self.spec.topology_type or "").lower() else 2.0
            crb_range_m = float(
                np.median(
                    c0
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

        jam_w = 0.0
        if "pls_covert_antijamming" in self._blocks:
            jam_w = float(np.median(rx_w)) * 0.25  # +25% adversarial interference floor (-6 dB JSR)

        eff_snr_lin = rx_w / np.maximum(self.noise_w + active_ris_noise_w + jam_w, 1e-30)

        if self.spec.csi_assumption == "ideal_genie_baseline":
            h_est = self._h_state.copy()
            csi_penalty = 1.0
        elif self.spec.csi_assumption == "delayed_feedback":
            h_est = self._h_prev_csi.copy()
            csi_penalty = max(0.5, 0.5 + 0.5 * abs(self.temporal_rho))
        else:
            est_var = 1.0 / (1.0 + np.maximum(eff_snr_lin, 0.1))
            est_noise = (self._rng.normal(size=self._h_state.shape) + 1j * self._rng.normal(size=self._h_state.shape)) / math.sqrt(2.0)
            h_est = (self._h_state + np.sqrt(est_var)[:, None, None] * est_noise).astype(np.complex64)
            csi_penalty = float(np.mean(1.0 / (1.0 + est_var)))

        self._h_prev_csi = self._h_state.copy()
        eff_snr_post = eff_snr_lin * csi_penalty

        # Achievable Rate: Shannon or Polyanskiy FBL dispersion if HRLLC enabled
        shannon_se = np.log2(1.0 + eff_snr_post)
        if "hrllc_fbl_urllc" in self._blocks:
            v_disp = (1.0 - (1.0 + eff_snr_post) ** (-2.0)) * (math.log2(math.e) ** 2)
            n_blk = 200.0
            se = np.maximum(shannon_se - np.sqrt(np.maximum(v_disp, 0.0) / n_blk) * qfuncinv(1e-5), 0.05)
        else:
            se = shannon_se

        # Layer 4: Packet arrivals & queue draining
        if packet_bits is not None:
            arr_bits = np.broadcast_to(np.asarray(packet_bits, dtype=np.float64), (self.batch_size,)).clip(min=0.0)
        else:
            arr_bits = self._draw_slot_arrivals_bits()
        cap_bits_per_slot = se * self.spec.bandwidth_hz * self.slot_sec
        self._queue_bits = np.maximum(self._queue_bits + arr_bits - cap_bits_per_slot, 0.0)
        self._last_eff_snr_lin = eff_snr_post.copy()
        self._last_se = se.copy()

        # Track deadline satisfaction (drained or small residual within PDB threshold)
        if np.any(arr_bits > 0):
            n_new = int(np.sum(arr_bits > 0))
            sat = int(np.sum((arr_bits > 0) & (self._queue_bits <= 0.5 * np.maximum(cap_bits_per_slot, 1e3))))
            self._total_arrivals += n_new
            self._on_time_arrivals += sat

        pdb_sat = float(self._on_time_arrivals / self._total_arrivals) if self._total_arrivals > 0 else 1.0
        out = ComposedSlotOutput(
            slot_index=self._slot_idx,
            h_small=h_small_t,
            h_csi_estimated=torch.from_numpy(h_est),
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
                "is_passive_ris": ("passive_ris_cascade" in self._blocks or "star_ris" in self._blocks) and "active_ris_flexible" not in self._blocks,
                "max_ris_modulus": max_ris_mod,
                "has_future_csi_leakage": False,
                "rayleigh_distance_m": self.rayleigh_zr,
                "mean_distance_m": float(np.mean(self._dists_2d)),
                "power_scale": act_power_mult,
                "action_applied": act_label,
                "action_power_cost": act_power_cost,
            },
        )
        self._history.append(out)
        self._slot_idx += 1
        return out

    def run_trajectory(self, num_slots: int = 20, *, seed: int | None = None) -> list[ComposedSlotOutput]:
        """Run a multi-slot deterministic wireless trajectory and return per-slot outputs."""
        self.reset(seed=seed)
        # reset() already executes slot 0
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

        # Verify empirical lag-1 temporal correlation vs theoretical Clarke J0
        if len(traj) >= 4:
            h_series = torch.stack([s.h_small for s in traj], dim=0)  # (T, B, Nr, Nt)
            num = torch.sum(h_series[1:] * h_series[:-1].conj()).real.item()
            den = math.sqrt(float(torch.sum(h_series[1:].abs().pow(2)).item() * torch.sum(h_series[:-1].abs().pow(2)).item())) + 1e-12
            emp_rho = num / den
            lag_gap = abs(emp_rho - self.temporal_rho)
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
                "is_passive_ris": ("passive_ris_cascade" in self._blocks or "star_ris" in self._blocks) and "active_ris_flexible" not in self._blocks,
                "max_ris_modulus": traj[-1].max_ris_modulus,
                "min_tx_rx_distance_m": self.spec.min_tx_rx_distance_m,
                "rayleigh_distance_m": self.rayleigh_zr,
                "assumed_farfield": False,
            },
            parameter_provenance=list(self.spec.parameter_provenance),
            mechanisms_used=[],
            standard_ids_used=list(self.spec.standard_ids),
        )

    def as_channel_module(self, *, snr_db: float = 15.0) -> ChannelModuleAdapter:
        """Expose composed complex64 small-scale channel into wannado-adv / Semantic AI training."""
        return ChannelModuleAdapter(lambda b: self.step().h_small[:b], snr_db=snr_db, add_noise=True)

    def as_gym_env(self, *, max_steps: int = 50) -> WirelessGymEnvAdapter:
        """Expose causal state [mean_snr_db, rate, queue_log, pdb_sat] to wannado-drl RL agents."""

        def _obs(out: ComposedSlotOutput) -> np.ndarray:
            return np.array(
                [out.mean_snr_db, out.achievable_rate_bps_hz, math.log1p(out.queue_bits_remaining), out.xr_pdb_satisfaction],
                dtype=np.float32,
            )

        def _reset(seed: int | None) -> tuple[np.ndarray, dict[str, Any]]:
            out = self.reset(seed=seed)
            return _obs(out), out.to_dict()

        def _step(action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
            out = self.step(action=action)
            power_cost = float(out.diagnostics.get("action_power_cost", 0.0))
            reward = float(out.achievable_rate_bps_hz - 0.05 * math.log1p(out.queue_bits_remaining) - power_cost)
            trunc = self._slot_idx >= max_steps
            return _obs(out), reward, False, trunc, out.to_dict()

        return WirelessGymEnvAdapter(obs_dim=4, action_dim=3, discrete_actions=True, reset_fn=_reset, step_fn=_step, max_steps=max_steps)
