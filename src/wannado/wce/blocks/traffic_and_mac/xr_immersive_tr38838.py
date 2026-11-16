"""3GPP TR 38.838 Immersive Extended Reality (XR) Periodic Frame Traffic & PDB Satisfaction.

[Lineage & Intuition]
- Family / Era / Tier : traffic_and_mac / standard / S (IMT-2030 Immersive Communication Pillar)
- Parent              : `traffic_ftp_models`
- Key Diff            : Models periodic jittered video frames (30/60/90/120 fps), tight Packet Delay
                        Budgets (10-20 ms PDB), and 3GPP TR 38.838 >=99% packet / >=90% user KPIs.
- Extension Hook [AI-Prompt Hook]:
    1. Pair with delay-aware QoS DRL scheduler inside `multicell_interference_mac`.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch

from wannado.wce.core.provenance import STANDARDS_PROVENANCE
from wannado.wce.core.types import ParamProvenance, ProvisionResult, WCEMeta
from wannado.wce.core.utils import set_global_seed

META = WCEMeta(
    id="traffic_xr_immersive",
    name="3GPP TR 38.838 Immersive XR Traffic & PDB Satisfaction",
    family="traffic_and_mac",
    era="standard",
    tier="S",
    output_kind="traffic_queue_trace",
    supports_torch_complex=False,
    supports_autograd=False,
    standard_ids=("gpp_tr38838", "itur_m2160"),
    parent_id="traffic_ftp_models",
    diff_from_parent="Simulates periodic bursty XR frames (60/90/120 fps), jitter, and TR 38.838 99% PDB satisfaction.",
    summary_plain="IMT-2030 Immersive Communication XR video frame generator with TR 38.838 delay-budget satisfaction evaluator.",
    when_to_use="AR/VR/Cloud-XR scheduling, IMT-2030 Immersive Communication, and tail-latency QoS benchmarks.",
    when_not_to_use="Delay-tolerant FTP downloads (`traffic_ftp_models`).",
    hyperparam_intuition={
        "fps": "XR video frame rate (30, 60, 90, or 120 fps); higher fps shrinks inter-frame interval (8.33 ms at 120 fps).",
        "pdb_ms": "Packet Delay Budget (typically 10-20 ms in TR 38.838) before motion-to-photon stall occurs.",
        "mean_bitrate_mbps": "Target XR stream bitrate (e.g., 30-45 Mbps per eye/stream).",
    },
    extension_hooks=("Export XR frame deadlines directly as DRL queue observation features",),
    module_path="wannado.wce.blocks.traffic_and_mac.xr_immersive_tr38838",
    default_mechanisms=(),
    citation=STANDARDS_PROVENANCE["gpp_tr38838"].citation,
)


@dataclass
class XRTrafficConfig:
    num_users: int = 12
    num_slots: int = 1000
    slot_duration_ms: float = 0.5
    fps: int = 60
    pdb_ms: float = 10.0
    mean_bitrate_mbps: float = 30.0
    jitter_std_ms: float = 1.5
    link_capacity_mbps: float = 420.0
    seed: int = 42


def provision_traffic_xr_immersive(config: XRTrafficConfig | None = None, **overrides: Any) -> ProvisionResult:
    """Generate jittered periodic XR frame arrivals & evaluate TR 38.838 >=99% PDB satisfaction."""
    cfg = config or XRTrafficConfig()
    if overrides:
        cfg = XRTrafficConfig(**{**asdict(cfg), **overrides})
    t0 = time.perf_counter()
    rng = set_global_seed(cfg.seed)

    period_ms = 1000.0 / max(1, cfg.fps)
    mean_frame_bytes = (cfg.mean_bitrate_mbps * 1e6 / 8.0) / max(1, cfg.fps)
    total_ms = cfg.num_slots * cfg.slot_duration_ms
    arrivals = np.zeros((cfg.num_slots, cfg.num_users), dtype=np.float64)

    for u in range(cfg.num_users):
        phase_ms = rng.uniform(0.0, period_ms)
        t_nom = phase_ms
        while t_nom < total_ms:
            t_jit = max(0.0, t_nom + rng.normal(0.0, cfg.jitter_std_ms))
            slot = int(t_jit / cfg.slot_duration_ms)
            if 0 <= slot < cfg.num_slots:
                # Truncated Gaussian frame size jitter around mean_frame_bytes
                f_bytes = float(np.clip(rng.normal(mean_frame_bytes, 0.15 * mean_frame_bytes), 0.3 * mean_frame_bytes, 2.0 * mean_frame_bytes))
                arrivals[slot, u] += f_bytes
            t_nom += period_ms

    # Baseline FIFO / Equal-Share PDB check over the shared cell capacity
    bytes_per_slot_cap = (cfg.link_capacity_mbps * 1e6 / 8.0) * (cfg.slot_duration_ms * 1e-3)
    pdb_slots = max(1, int(round(cfg.pdb_ms / cfg.slot_duration_ms)))
    satisfied_users = 0
    packet_sat_rates: list[float] = []
    per_user_cap = bytes_per_slot_cap / max(1, cfg.num_users)
    for u in range(cfg.num_users):
        # Track frame FIFO with expiration at pdb_slots
        pending: list[tuple[int, float]] = []  # (arrival_slot, remaining_bytes)
        delivered_frames = 0
        total_frames = 0
        for s in range(cfg.num_slots):
            if arrivals[s, u] > 0:
                pending.append((s, float(arrivals[s, u])))
                total_frames += 1
            budget = per_user_cap
            new_pending: list[tuple[int, float]] = []
            for arr_s, rem in pending:
                if (s - arr_s) > pdb_slots:
                    continue  # Expired past PDB
                if budget > 0:
                    served = min(budget, rem)
                    rem -= served
                    budget -= served
                if rem <= 1e-6:
                    delivered_frames += 1
                else:
                    new_pending.append((arr_s, rem))
            pending = new_pending
        rate = delivered_frames / max(1, total_frames)
        packet_sat_rates.append(rate)
        if rate >= 0.99:
            satisfied_users += 1

    user_sat_frac = satisfied_users / max(1, cfg.num_users)
    prov = [
        ParamProvenance("fps", cfg.fps, "fps", "XR periodic frame rate", "gpp_tr38838", "Clause 4", "3gpp_tr38838", "Standard TR 38.838 evaluation rates (30/60/90/120 fps)."),
        ParamProvenance("pdb_ms", cfg.pdb_ms, "ms", "Packet Delay Budget deadline", "gpp_tr38838", "Clause 5", "3gpp_tr38838", "User satisfied iff >=99% frames delivered within PDB."),
    ]
    return ProvisionResult(
        block_id=META.id,
        seed=cfg.seed,
        elapsed_seconds=time.perf_counter() - t0,
        config=asdict(cfg),
        tensors={"arrivals_bytes": torch.from_numpy(arrivals).to(torch.float32)},
        metrics={
            "mean_packet_satisfaction": float(np.mean(packet_sat_rates)),
            "satisfied_user_fraction": user_sat_frac,
            "meets_tr38838_90pct_capacity": float(user_sat_frac >= 0.90),
            "total_offered_mbps": cfg.num_users * cfg.mean_bitrate_mbps,
        },
        diagnostics={"normalized_channel_power": 1.0},
        parameter_provenance=prov,
        mechanisms_used=[],
        standard_ids_used=list(META.standard_ids),
    )
