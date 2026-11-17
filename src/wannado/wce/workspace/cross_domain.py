"""Cross-domain non-telecom simulation adapters built on ComposedWirelessEnv.

Bridges 3GPP/ITU wireless physics into adjacent research fields without requiring
telecommunications expertise:
- Federated Learning / Distributed Edge AI (`FederatedWirelessRoundModel`)
- Robotics / UAV GIS / Autonomous Vehicle Control (`AutonomousLinkDelayModel`)
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from wannado.wce.core.units import linear_to_db, qfunc
from wannado.wce.intake.archetypes import build_wce_archetype
from wannado.wce.intake.spec import WCESpec
from wannado.wce.workspace.composed_env import ComposedWirelessEnv


@dataclass
class FederatedRoundResult:
    """Per-round uplink telemetry for wireless Federated Learning aggregation."""

    round_index: int
    num_clients: int
    participating_mask: np.ndarray
    participation_rate: float
    straggler_count: int
    per_client_snr_db: np.ndarray
    per_client_tx_time_sec: np.ndarray
    total_round_energy_j: float
    normalized_client_weights: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        return {
            "round_index": self.round_index,
            "num_clients": self.num_clients,
            "participating_mask": self.participating_mask.tolist(),
            "participation_rate": self.participation_rate,
            "straggler_count": self.straggler_count,
            "per_client_snr_db": self.per_client_snr_db.tolist(),
            "per_client_tx_time_sec": self.per_client_tx_time_sec.tolist(),
            "total_round_energy_j": self.total_round_energy_j,
            "normalized_client_weights": self.normalized_client_weights.tolist(),
        }


class FederatedWirelessRoundModel:
    """Stateful multi-round wireless FL uplink straggler and energy simulator."""

    def __init__(
        self,
        *,
        num_clients: int = 16,
        payload_bits_per_client: float = 2.0e6,
        deadline_sec: float = 0.5,
        wce_spec: WCESpec | None = None,
        rain_rate_mm_hr: float = 0.0,
        seed: int = 42,
    ) -> None:
        if num_clients < 1:
            raise ValueError(f"num_clients must be >= 1, got {num_clients}.")
        if payload_bits_per_client <= 0.0:
            raise ValueError("payload_bits_per_client must be > 0.")
        if deadline_sec <= 0.0:
            raise ValueError("deadline_sec must be > 0.")

        self.num_clients = int(num_clients)
        self.payload_bits_per_client = float(payload_bits_per_client)
        self.deadline_sec = float(deadline_sec)
        self.spec = wce_spec or build_wce_archetype("federated_edge")
        self.env = ComposedWirelessEnv(
            self.spec,
            batch_size=self.num_clients,
            seed=seed,
            rain_rate_mm_hr=rain_rate_mm_hr,
        )
        self.env.reset(seed=seed)
        self.round_index = 0

    def reset(self, *, seed: int | None = None) -> None:
        self.round_index = 0
        self.env.reset(seed=seed)

    def step_round(
        self,
        *,
        client_distances_m: np.ndarray | Sequence[float] | None = None,
        client_weights: np.ndarray | Sequence[float] | None = None,
        payload_bits: float | Sequence[float] | np.ndarray | None = None,
        deadline_sec: float | None = None,
        action: Any = None,
        rain_rate_mm_hr: float | None = None,
    ) -> FederatedRoundResult:
        """Execute one synchronous federated round over causal wireless fading."""
        bits_arr = np.broadcast_to(
            np.asarray(
                self.payload_bits_per_client if payload_bits is None else payload_bits,
                dtype=np.float64,
            ),
            (self.num_clients,),
        ).clip(min=1.0)
        eff_deadline = self.deadline_sec if deadline_sec is None else max(1e-6, float(deadline_sec))

        out = self.env.step(
            action=action,
            distances_m=client_distances_m,
            packet_bits=bits_arr,
            rain_rate_mm_hr=rain_rate_mm_hr,
        )
        se = (
            self.env._last_se
            if self.env._last_se is not None
            else np.full(self.num_clients, out.achievable_rate_bps_hz, dtype=np.float64)
        )
        snr_lin = (
            self.env._last_eff_snr_lin
            if self.env._last_eff_snr_lin is not None
            else np.ones(self.num_clients, dtype=np.float64)
        )
        bw_share = self.spec.bandwidth_hz / float(self.num_clients)
        rate_bps = np.maximum(se * bw_share, 1.0)
        tx_time_sec = bits_arr / rate_bps
        mask = tx_time_sec <= eff_deadline

        p_mult = float(out.diagnostics.get("power_scale", 1.0))
        round_energy = float(np.sum((self.env.tx_w * p_mult) * np.minimum(tx_time_sec, eff_deadline)))

        if client_weights is None:
            raw_w = np.ones(self.num_clients, dtype=np.float64)
        else:
            raw_w = np.broadcast_to(np.asarray(client_weights, dtype=np.float64), (self.num_clients,)).clip(min=0.0)
        active_w = np.where(mask, raw_w, 0.0)
        w_sum = float(np.sum(active_w))
        norm_w = active_w / w_sum if w_sum > 0.0 else np.zeros(self.num_clients, dtype=np.float64)

        res = FederatedRoundResult(
            round_index=self.round_index,
            num_clients=self.num_clients,
            participating_mask=mask,
            participation_rate=float(np.mean(mask)),
            straggler_count=int(self.num_clients - int(np.sum(mask))),
            per_client_snr_db=np.asarray(linear_to_db(snr_lin), dtype=np.float64),
            per_client_tx_time_sec=tx_time_sec,
            total_round_energy_j=round_energy,
            normalized_client_weights=norm_w,
        )
        self.round_index += 1
        return res


def simulate_federated_round(
    *,
    num_clients: int = 16,
    payload_bits_per_client: float = 2.0e6,
    deadline_sec: float = 0.5,
    client_distances_m: np.ndarray | Sequence[float] | None = None,
    wce_spec: WCESpec | None = None,
    rain_rate_mm_hr: float = 0.0,
    seed: int = 42,
) -> FederatedRoundResult:
    """Convenience single-round wireless Federated Learning participation evaluator."""
    model = FederatedWirelessRoundModel(
        num_clients=num_clients,
        payload_bits_per_client=payload_bits_per_client,
        deadline_sec=deadline_sec,
        wce_spec=wce_spec,
        rain_rate_mm_hr=rain_rate_mm_hr,
        seed=seed,
    )
    return model.step_round(client_distances_m=client_distances_m)


@dataclass
class LinkDeliveryResult:
    """Single-step packet latency and reliability telemetry for robotics / CAV / UAV loops."""

    connected: bool
    within_budget: bool
    snr_db: float
    rate_mbps: float
    air_latency_ms: float
    queue_wait_ms: float
    total_latency_ms: float
    packet_error_prob: float
    distance_m: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AutonomousLinkDelayModel:
    """Causal control/telemetry link latency & BLER model for Robotics / UAV / V2V simulators."""

    def __init__(
        self,
        *,
        wce_spec: WCESpec | None = None,
        preset: str = "v2v_control",
        latency_budget_ms: float = 10.0,
        packet_bits: float = 8000.0,
        rain_rate_mm_hr: float = 0.0,
        seed: int = 42,
    ) -> None:
        if latency_budget_ms <= 0.0 or packet_bits <= 0.0:
            raise ValueError("latency_budget_ms and packet_bits must be strictly positive.")
        self.spec = wce_spec or build_wce_archetype(preset)
        self.latency_budget_ms = float(latency_budget_ms)
        self.packet_bits = float(packet_bits)
        self.env = ComposedWirelessEnv(
            self.spec,
            batch_size=1,
            seed=seed,
            rain_rate_mm_hr=rain_rate_mm_hr,
        )
        self.env.reset(seed=seed)

    def evaluate_step(
        self,
        *,
        tx_pos_3d: np.ndarray | Sequence[float] | None = None,
        rx_pos_3d: np.ndarray | Sequence[float] | None = None,
        distance_m: float | None = None,
        packet_bits: float | None = None,
        latency_budget_ms: float | None = None,
        rain_rate_mm_hr: float | None = None,
        action: Any = None,
    ) -> LinkDeliveryResult:
        """Step causal channel with live 3D coordinates/weather and return latency + BLER."""
        bits = self.packet_bits if packet_bits is None else max(1.0, float(packet_bits))
        budget_ms = self.latency_budget_ms if latency_budget_ms is None else max(1e-3, float(latency_budget_ms))
        out = self.env.step(
            action=action,
            distances_m=distance_m,
            tx_pos_3d=tx_pos_3d,
            rx_pos_3d=rx_pos_3d,
            packet_bits=bits,
            rain_rate_mm_hr=rain_rate_mm_hr,
        )
        snr_lin = float(self.env._last_eff_snr_lin[0]) if self.env._last_eff_snr_lin is not None else 1.0
        se = float(self.env._last_se[0]) if self.env._last_se is not None else out.achievable_rate_bps_hz
        snr_db = float(linear_to_db(snr_lin))
        rate_bps = max(se * self.spec.bandwidth_hz, 1.0)
        rate_mbps = rate_bps / 1e6

        air_latency_ms = 1000.0 * bits / rate_bps
        queue_wait_ms = 1000.0 * float(self.env._queue_bits[0]) / rate_bps
        total_latency_ms = air_latency_ms + queue_wait_ms

        # Polyanskiy normal-approximation block error estimate over one slot blocklength
        n_sym = max(64.0, self.spec.bandwidth_hz * (budget_ms * 1e-3) * 0.1)
        req_se = bits / max(self.spec.bandwidth_hz * (budget_ms * 1e-3), 1.0)
        cap_shannon = math.log2(1.0 + max(snr_lin, 1e-12))
        v_disp = max((1.0 - (1.0 + max(snr_lin, 1e-12)) ** (-2.0)) * (math.log2(math.e) ** 2), 1e-6)
        arg = math.sqrt(n_sym / v_disp) * (cap_shannon - req_se) * math.log(2.0)
        bler = float(np.clip(qfunc(np.asarray(arg, dtype=np.float64)), 0.0, 1.0))

        eff_dist = float(self.env._dists_2d[0])
        connected = bool(snr_db > -5.0 and bler < 0.5)
        within_budget = bool(connected and total_latency_ms <= budget_ms and bler < 0.1)
        return LinkDeliveryResult(
            connected=connected,
            within_budget=within_budget,
            snr_db=snr_db,
            rate_mbps=rate_mbps,
            air_latency_ms=air_latency_ms,
            queue_wait_ms=queue_wait_ms,
            total_latency_ms=total_latency_ms,
            packet_error_prob=bler,
            distance_m=eff_dist,
        )


def evaluate_link_delivery(
    tx_pos_3d: np.ndarray | Sequence[float],
    rx_pos_3d: np.ndarray | Sequence[float],
    *,
    packet_bits: float = 8000.0,
    latency_budget_ms: float = 10.0,
    preset: str = "v2v_control",
    rain_rate_mm_hr: float = 0.0,
    seed: int = 42,
) -> LinkDeliveryResult:
    """One-shot 3D position-to-latency/reliability evaluator for robotics, UAV GIS, and CAVs."""
    model = AutonomousLinkDelayModel(
        preset=preset,
        latency_budget_ms=latency_budget_ms,
        packet_bits=packet_bits,
        rain_rate_mm_hr=rain_rate_mm_hr,
        seed=seed,
    )
    return model.evaluate_step(tx_pos_3d=tx_pos_3d, rx_pos_3d=rx_pos_3d)
