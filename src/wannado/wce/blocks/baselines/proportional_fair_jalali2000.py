"""Proportional Fair (PF) vs Max-SNR vs Round-Robin Multi-User Scheduler (Jalali et al., VTC 2000).

[Lineage & Intuition]
- Family / Era / Tier : baselines / classic / S (Mandatory MAC Scheduling Floor)
- Parent              : `multicell_interference_mac`
- Key Diff            : Quantifies throughput vs Jain's Fairness Index across PF, Max-CQI, and Round-Robin
                        under heterogeneous user pathlosses.
- Extension Hook [AI-Prompt Hook]:
    1. Any DRL scheduler (`wannado-drl`) MUST outperform both PF and Max-SNR under the target QoS metric!
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch

from wannado.wce.core.provenance import MECHANISM_PROVENANCE
from wannado.wce.core.types import ParamProvenance, ProvisionResult, WCEMeta
from wannado.wce.core.units import db_to_linear
from wannado.wce.core.utils import set_global_seed

META = WCEMeta(
    id="proportional_fair",
    name="Proportional Fair vs Max-SNR vs Round-Robin Scheduling Baselines",
    family="baselines",
    era="classic",
    tier="S",
    output_kind="baseline_policy_solution",
    supports_torch_complex=False,
    supports_autograd=False,
    standard_ids=("gpp_tr38913",),
    parent_id="multicell_interference_mac",
    diff_from_parent="Benchmarks Proportional Fair (R_i/T_i) against greedy Max-SNR and Round-Robin fairness.",
    summary_plain="Canonical wireless scheduler baseline suite measuring cell sum-throughput and Jain's Fairness Index.",
    when_to_use="Baseline floor for radio resource management, slicing, and DRL scheduling papers.",
    when_not_to_use="Physical-layer MIMO precoding (`wmmse_precoder`).",
    hyperparam_intuition={
        "pf_window_tc": "Exponential averaging window T_c in slots controlling short- vs long-term fairness.",
        "snr_spread_db": "Edge-to-center average SNR heterogeneity across users.",
    },
    extension_hooks=("Add delay-weighted M-LWDF priority term for XR/URLLC queues",),
    module_path="wannado.wce.blocks.baselines.proportional_fair_jalali2000",
    default_mechanisms=("jalali_proportional_fair",),
    citation=MECHANISM_PROVENANCE["jalali_proportional_fair"].citation,
)


@dataclass
class PFSchedulerConfig:
    num_users: int = 8
    num_slots: int = 600
    mean_snr_db: float = 10.0
    snr_spread_db: float = 12.0
    pf_window_tc: float = 40.0
    seed: int = 42


def _jains_fairness(x: np.ndarray) -> float:
    s1 = float(np.sum(x))
    s2 = float(np.sum(x ** 2))
    return (s1 ** 2) / max(len(x) * s2, 1e-12)


def provision_proportional_fair(config: PFSchedulerConfig | None = None, **overrides: Any) -> ProvisionResult:
    """Compare Proportional Fair, Max-SNR, and Round-Robin schedulers over asymmetric Rayleigh fading UEs."""
    cfg = config or PFSchedulerConfig()
    if overrides:
        cfg = PFSchedulerConfig(**{**asdict(cfg), **overrides})
    t0 = time.perf_counter()
    rng = set_global_seed(cfg.seed)

    # Heterogeneous user average SNRs (edge to cell center)
    user_snr_db = np.linspace(cfg.mean_snr_db - 0.5 * cfg.snr_spread_db, cfg.mean_snr_db + 0.5 * cfg.snr_spread_db, cfg.num_users)
    user_snr_lin = np.asarray(db_to_linear(user_snr_db), dtype=np.float64)
    fading = rng.exponential(1.0, size=(cfg.num_slots, cfg.num_users))
    inst_rates = np.log2(1.0 + fading * user_snr_lin[None, :])  # (T, K)

    def _sim(mode: str) -> tuple[float, float]:
        thr_avg = np.ones(cfg.num_users) * 0.5
        tot_served = np.zeros(cfg.num_users)
        alpha = 1.0 / max(1.0, cfg.pf_window_tc)
        for t in range(cfg.num_slots):
            r_t = inst_rates[t]
            if mode == "pf":
                u = int(np.argmax(r_t / np.maximum(thr_avg, 1e-6)))
            elif mode == "max_snr":
                u = int(np.argmax(r_t))
            else:
                u = t % cfg.num_users
            served = np.zeros(cfg.num_users)
            served[u] = r_t[u]
            tot_served += served
            thr_avg = (1.0 - alpha) * thr_avg + alpha * served
        return float(np.sum(tot_served) / cfg.num_slots), _jains_fairness(tot_served)

    pf_sum, pf_fair = _sim("pf")
    max_sum, max_fair = _sim("max_snr")
    rr_sum, rr_fair = _sim("rr")

    prov = [
        ParamProvenance("pf_window_tc", cfg.pf_window_tc, "slots", "Exponential throughput averaging constant T_c", "jalali_proportional_fair", "IEEE VTC 2000", "jalali2000data", "Balances multi-user diversity gain against edge-user fairness."),
    ]
    return ProvisionResult(
        block_id=META.id,
        seed=cfg.seed,
        elapsed_seconds=time.perf_counter() - t0,
        config=asdict(cfg),
        tensors={"inst_rates": torch.from_numpy(inst_rates).to(torch.float32)},
        metrics={
            "pf_sum_rate_bps_hz": pf_sum,
            "pf_jains_fairness": pf_fair,
            "max_snr_sum_rate_bps_hz": max_sum,
            "max_snr_jains_fairness": max_fair,
            "rr_sum_rate_bps_hz": rr_sum,
            "rr_jains_fairness": rr_fair,
        },
        diagnostics={"normalized_channel_power": 1.0},
        parameter_provenance=prov,
        mechanisms_used=list(META.default_mechanisms),
        standard_ids_used=list(META.standard_ids),
    )
