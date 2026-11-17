"""Poisson, Truncated Pareto Heavy-Tail, & On-Off Markov VoIP Traffic Generator.

[Lineage & Intuition]
- Family / Era / Tier : traffic_and_mac / classic / S
- Parent              : `traffic_ftp_models`
- Key Diff            : Supports mixed traffic distributions: memoryless Poisson packet arrivals,
                        truncated Pareto heavy-tailed payload sizes, and On-Off Markov VoIP speech spurts.
- Extension Hook [AI-Prompt Hook]:
    1. Multiplex sporadic URLLC Poisson bursts over background heavy-tailed Pareto flows.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
import torch

from wannado.wce.core.provenance import STANDARDS_PROVENANCE
from wannado.wce.core.types import ParamProvenance, ProvisionResult, WCEMeta
from wannado.wce.core.utils import set_global_seed

META = WCEMeta(
    id="traffic_poisson_pareto_voip",
    name="Poisson / Truncated Pareto / VoIP On-Off Traffic Generator",
    family="traffic_and_mac",
    era="classic",
    tier="S",
    output_kind="traffic_queue_trace",
    supports_torch_complex=False,
    supports_autograd=False,
    standard_ids=("gpp_tr36814", "gpp_tr38824"),
    parent_id="traffic_ftp_models",
    diff_from_parent="Covers Poisson sporadic arrivals, heavy-tailed truncated Pareto packet sizes, and 20ms/160ms On-Off VoIP.",
    summary_plain="Standard statistical traffic generator supporting Poisson arrivals, Pareto heavy-tail sizes, and Markov VoIP streams.",
    when_to_use="Mixed eMBB/mMTC/URLLC/VoIP queueing studies and wireless network packet arrival modeling.",
    when_not_to_use="Video-frame periodic XR streams (`traffic_xr_immersive`).",
    hyperparam_intuition={
        "traffic_kind": "poisson (fixed-size memoryless) | pareto (heavy-tail burst sizes) | voip_on_off (20ms active / 160ms SID).",
        "pareto_alpha": "Tail heaviness exponent (> 1); smaller alpha produces more extreme large-packet bursts.",
    },
    extension_hooks=("Combine Poisson URLLC + Pareto eMBB in `multicell_interference_mac`",),
    module_path="wannado.wce.blocks.traffic_and_mac.poisson_pareto_voip",
    default_mechanisms=(),
    citation=STANDARDS_PROVENANCE["gpp_tr36814"].citation,
)


@dataclass
class MixedTrafficConfig:
    traffic_kind: Literal["poisson", "pareto", "voip_on_off"] = "pareto"
    num_users: int = 16
    num_slots: int = 1000
    slot_duration_ms: float = 1.0
    arrival_rate_hz: float = 20.0
    fixed_packet_bytes: float = 1500.0
    pareto_scale_bytes: float = 512.0
    pareto_alpha: float = 1.5
    pareto_max_bytes: float = 65536.0
    seed: int = 42


def provision_traffic_poisson_pareto_voip(config: MixedTrafficConfig | None = None, **overrides: Any) -> ProvisionResult:
    """Generate Poisson, truncated Pareto heavy-tail, or 3GPP On-Off VoIP arrival trace A[T, U]."""
    cfg = config or MixedTrafficConfig()
    if overrides:
        cfg = MixedTrafficConfig(**{**asdict(cfg), **overrides})
    t0 = time.perf_counter()
    rng = set_global_seed(cfg.seed)

    dt_sec = cfg.slot_duration_ms * 1e-3
    arrivals = np.zeros((cfg.num_slots, cfg.num_users), dtype=np.float64)

    if cfg.traffic_kind == "poisson":
        counts = rng.poisson(cfg.arrival_rate_hz * dt_sec, size=(cfg.num_slots, cfg.num_users))
        arrivals = (counts * cfg.fixed_packet_bytes).astype(np.float64)
    elif cfg.traffic_kind == "pareto":
        counts = rng.poisson(cfg.arrival_rate_hz * dt_sec, size=(cfg.num_slots, cfg.num_users))
        total_events = int(np.sum(counts))
        if total_events > 0:
            # Pareto Type I: X = x_m * (1 + pareto(alpha)) clamped to pareto_max_bytes
            sizes = np.minimum(cfg.pareto_max_bytes, cfg.pareto_scale_bytes * (1.0 + rng.pareto(cfg.pareto_alpha, size=total_events)))
            idx = 0
            for t in range(cfg.num_slots):
                for u in range(cfg.num_users):
                    c = counts[t, u]
                    if c > 0:
                        arrivals[t, u] = float(np.sum(sizes[idx : idx + c]))
                        idx += c
    else:
        # 3GPP TR 36.814 On-Off Markov VoIP (VAF=0.5, 20 ms active 40B, 160 ms SID 15B)
        p_on_to_off = dt_sec / 1.0
        p_off_to_on = dt_sec / 1.0
        active_step = max(1, round(20.0 / cfg.slot_duration_ms))
        sid_step = max(1, round(160.0 / cfg.slot_duration_ms))
        state = rng.uniform(0.0, 1.0, size=cfg.num_users) < 0.5
        for t in range(cfg.num_slots):
            flip_off = state & (rng.uniform(0.0, 1.0, size=cfg.num_users) < p_on_to_off)
            flip_on = (~state) & (rng.uniform(0.0, 1.0, size=cfg.num_users) < p_off_to_on)
            state = (state & ~flip_off) | flip_on
            if t % active_step == 0:
                arrivals[t, state] += 40.0
            if t % sid_step == 0:
                arrivals[t, ~state] += 15.0

    nz = arrivals[arrivals > 0]
    offered_kbps = (float(np.sum(arrivals)) * 8.0 / max(cfg.num_slots * dt_sec, 1e-9)) / 1e3
    prov = [
        ParamProvenance("traffic_kind", cfg.traffic_kind, "enum", "Stochastic traffic law", "gpp_tr36814", "Annex A.2.1.3", "3gpp_tr36814", "Poisson memoryless arrivals, heavy-tail Pareto sizes, or 50% VAF Markov VoIP."),
    ]
    return ProvisionResult(
        block_id=META.id,
        seed=cfg.seed,
        elapsed_seconds=time.perf_counter() - t0,
        config=asdict(cfg),
        tensors={"arrivals_bytes": torch.from_numpy(arrivals).to(torch.float32)},
        metrics={
            "offered_traffic_kbps": offered_kbps,
            "mean_burst_bytes": float(np.mean(nz)) if nz.size else 0.0,
            "p99_burst_bytes": float(np.percentile(nz, 99.0)) if nz.size else 0.0,
            "heavy_tail_ratio_p99_over_mean": float(np.percentile(nz, 99.0)) / max(float(np.mean(nz)), 1e-9) if nz.size else 1.0,
        },
        diagnostics={"normalized_channel_power": 1.0},
        parameter_provenance=prov,
        mechanisms_used=[],
        standard_ids_used=list(META.standard_ids),
    )
