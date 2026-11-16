"""3GPP TR 36.814 / TR 38.913 FTP Traffic Models (Model 1, Model 2, Model 3).

[Lineage & Intuition]
- Family / Era / Tier : traffic_and_mac / standard / S
- Parent              : None (Standard non-full-buffer 3GPP arrival baseline)
- Key Diff            : Replaces unrealistic full-buffer traffic with 3GPP TR 36.814 bursty file downloads,
                        exponential reading times, and Poisson arrival intensities lambda.
- Extension Hook [AI-Prompt Hook]:
    1. Combine with heavy-tailed Pareto / VoIP -> `traffic_poisson_pareto_voip`.
    2. Upgrade to periodic bursty XR frames -> `traffic_xr_immersive`.
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
    id="traffic_ftp_models",
    name="3GPP TR 36.814 FTP Traffic Models (Model 1 / 2 / 3)",
    family="traffic_and_mac",
    era="standard",
    tier="S",
    output_kind="traffic_queue_trace",
    supports_torch_complex=False,
    supports_autograd=False,
    standard_ids=("gpp_tr36814", "gpp_tr38913"),
    parent_id=None,
    diff_from_parent="Standard 3GPP non-full-buffer FTP file arrival & reading-time trace generator.",
    summary_plain="Generates realistic 3GPP FTP Model 1/2/3 packet arrivals across UEs and slots instead of full-buffer queues.",
    when_to_use="5G/6G MAC scheduling, queue stability, and system-level throughput/delay evaluation.",
    when_not_to_use="Periodic XR video frames (`traffic_xr_immersive`) or pure PHY link-level SNR sweeps.",
    hyperparam_intuition={
        "model": "ftp_model_1 (Poisson session lambda, 2MB), ftp_model_2 (0.5MB + reading time), ftp_model_3 (Poisson packets, 0.5MB).",
        "arrival_rate_hz": "Mean session/packet arrival rate lambda per user in Hz (packets/sec).",
    },
    extension_hooks=("Feed generated arrival matrix A[T, U] into `multicell_interference_mac`",),
    module_path="wannado.wce.blocks.traffic_and_mac.ftp_traffic_3gpp",
    default_mechanisms=(),
    citation=STANDARDS_PROVENANCE["gpp_tr36814"].citation,
)


@dataclass
class FTPTrafficConfig:
    model: Literal["ftp_model_1", "ftp_model_2", "ftp_model_3"] = "ftp_model_3"
    num_users: int = 16
    num_slots: int = 1000
    slot_duration_ms: float = 1.0
    arrival_rate_hz: float = 1.5
    file_size_bytes: float | None = None
    mean_reading_time_sec: float = 2.0
    seed: int = 42


def provision_traffic_ftp_models(config: FTPTrafficConfig | None = None, **overrides: Any) -> ProvisionResult:
    """Generate per-slot per-user FTP traffic byte arrival traces A[T, U] per 3GPP TR 36.814."""
    cfg = config or FTPTrafficConfig()
    if overrides:
        cfg = FTPTrafficConfig(**{**asdict(cfg), **overrides})
    t0 = time.perf_counter()
    rng = set_global_seed(cfg.seed)

    default_bytes = 2_000_000.0 if cfg.model == "ftp_model_1" else 500_000.0
    f_bytes = float(cfg.file_size_bytes) if cfg.file_size_bytes is not None else default_bytes
    dt_sec = cfg.slot_duration_ms * 1e-3
    arrivals = np.zeros((cfg.num_slots, cfg.num_users), dtype=np.float64)

    if cfg.model in ("ftp_model_1", "ftp_model_3"):
        lam_slot = cfg.arrival_rate_hz * dt_sec
        counts = rng.poisson(lam_slot, size=(cfg.num_slots, cfg.num_users))
        arrivals = counts * f_bytes
    else:
        # FTP Model 2: burst arrival followed by exponential reading time D
        for u in range(cfg.num_users):
            t_cur = rng.exponential(cfg.mean_reading_time_sec)
            total_sec = cfg.num_slots * dt_sec
            while t_cur < total_sec:
                slot_idx = min(cfg.num_slots - 1, int(t_cur / dt_sec))
                arrivals[slot_idx, u] += f_bytes
                t_cur += cfg.mean_reading_time_sec * 0.1 + rng.exponential(cfg.mean_reading_time_sec)

    total_bits = float(np.sum(arrivals) * 8.0)
    duration_sec = cfg.num_slots * dt_sec
    offered_mbps = (total_bits / max(duration_sec, 1e-9)) / 1e6

    prov = [
        ParamProvenance("model", cfg.model, "enum", "3GPP FTP traffic model", "gpp_tr36814", "Annex A.2.1.3", "3gpp_tr36814", f"Standard file size {f_bytes/1e6:.2f} MB with {cfg.model} arrivals."),
        ParamProvenance("arrival_rate_hz", cfg.arrival_rate_hz, "Hz", "Poisson arrival rate lambda", "gpp_tr38913", "Clause 6", "3gpp_tr38913", "Governs offered network load and queue congestion."),
    ]
    return ProvisionResult(
        block_id=META.id,
        seed=cfg.seed,
        elapsed_seconds=time.perf_counter() - t0,
        config=asdict(cfg),
        tensors={"arrivals_bytes": torch.from_numpy(arrivals).to(torch.float32)},
        metrics={
            "offered_traffic_mbps": offered_mbps,
            "mean_arrivals_per_user": float(np.sum(arrivals > 0) / max(1, cfg.num_users)),
            "file_size_mb": f_bytes / 1e6,
            "active_slot_fraction": float(np.mean(arrivals > 0)),
        },
        diagnostics={"normalized_channel_power": 1.0},
        parameter_provenance=prov,
        mechanisms_used=[],
        standard_ids_used=list(META.standard_ids),
    )
