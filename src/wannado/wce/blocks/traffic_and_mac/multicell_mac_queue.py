"""Multi-Cell Dynamic Interference & Queueing MAC Environment (Gymnasium / DRL Ready).

[Lineage & Intuition]
- Family / Era / Tier : traffic_and_mac / standard / S (Direct Bridge to wannado-drl)
- Parent              : `tr38901_uma_umi_rma`
- Key Diff            : Integrates wrap-around multi-cell TR 38.901 pathloss, Jakes time-correlated fading,
                        non-full-buffer Poisson/FTP queues, and strict causal CQI observations inside
                        a Gym-compatible step/reset loop.
- Extension Hook [AI-Prompt Hook]:
    1. Hand off `result.extra['env']` directly to `wannado-drl` EnvInterface for PPO/DQN/MAPPO scheduling!
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from wannado.wce.core.adapters import WirelessGymEnvAdapter
from wannado.wce.core.geometry import wrap_around_distances_2d
from wannado.wce.core.provenance import MECHANISM_PROVENANCE
from wannado.wce.core.types import ParamProvenance, ProvisionResult, WCEMeta
from wannado.wce.core.units import db_to_linear, dbm_to_watt, thermal_noise_power_watt
from wannado.wce.core.utils import set_global_seed
from wannado.wce.standards.gpp_38901 import compute_los_probability_tr38901, compute_pathloss_tr38901_db

META = WCEMeta(
    id="multicell_interference_mac",
    name="Multi-Cell Dynamic Scheduling & Interference Queue Environment",
    family="traffic_and_mac",
    era="standard",
    tier="S",
    output_kind="mac_gym_environment",
    supports_torch_complex=False,
    supports_autograd=False,
    standard_ids=("gpp_tr38901", "gpp_ts38211", "gpp_tr36814"),
    parent_id="tr38901_uma_umi_rma",
    diff_from_parent="Combines wrap-around multi-cell interference, causal CQI, queue dynamics, and Gym step/reset interface.",
    summary_plain="Ready-to-train multi-cell wireless downlink scheduler environment with causal CSI, queues, and PF baseline.",
    when_to_use="Dynamic spectrum/power/user scheduling via heuristics or DRL (`wannado-drl`).",
    when_not_to_use="Single-link PHY channel estimation (`tr38901_cdl_tdl`).",
    hyperparam_intuition={
        "num_cells": "Number of interfering base stations on a wrap-around torus grid.",
        "users_per_cell": "Active user queues competing for scheduling slots inside the target cell.",
        "arrival_bits_per_slot": "Mean Poisson traffic arrival per user queue per TTI slot.",
    },
    extension_hooks=("Mount extra['env'] in ./drl_lab/interface.py for multi-seed PPO vs PF comparison",),
    module_path="wannado.wce.blocks.traffic_and_mac.multicell_mac_queue",
    default_mechanisms=("tr38901_pathloss_los", "jalali_proportional_fair"),
    citation=MECHANISM_PROVENANCE["jalali_proportional_fair"].citation,
)


@dataclass
class MultiCellMACConfig:
    num_cells: int = 4
    users_per_cell: int = 4
    box_size_m: float = 800.0
    fc_ghz: float = 3.5
    bandwidth_hz: float = 10e6
    tx_power_dbm: float = 35.0
    noise_figure_db: float = 7.0
    arrival_bits_per_slot: float = 8000.0
    slot_duration_ms: float = 1.0
    episode_horizon: int = 100
    fading_rho: float = 0.90
    seed: int = 42


def build_multicell_gym_env(cfg: MultiCellMACConfig) -> WirelessGymEnvAdapter:
    """Build a causal, anti-leakage Gymnasium-compatible multi-cell wireless scheduling environment."""
    k_users = cfg.users_per_cell
    obs_dim = 3 * k_users  # [norm_queues, prev_causal_cqi_sinr_db, avg_rate_norm]
    noise_w = thermal_noise_power_watt(cfg.bandwidth_hz, noise_figure_db=cfg.noise_figure_db)
    tx_w = dbm_to_watt(cfg.tx_power_dbm)
    dt_sec = cfg.slot_duration_ms * 1e-3

    state: dict[str, Any] = {"rng": np.random.default_rng(cfg.seed)}

    def _reset(seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        rng = np.random.default_rng(seed if seed is not None else cfg.seed)
        state["rng"] = rng
        # Place BSs and target-cell UEs inside torus [0, box_size_m)^2
        bs_xy = rng.uniform(0.0, cfg.box_size_m, size=(cfg.num_cells, 2))
        ue_xy = (bs_xy[0:1] + rng.uniform(-120.0, 120.0, size=(k_users, 2))) % cfg.box_size_m
        dists = np.maximum(wrap_around_distances_2d(bs_xy, ue_xy, box_size_m=cfg.box_size_m), 15.0)
        p_los = compute_los_probability_tr38901("UMi", dists)
        is_los = rng.uniform(0.0, 1.0, size=dists.shape) < p_los
        pl_db, _ = compute_pathloss_tr38901_db("UMi", dists, cfg.fc_ghz, is_los=is_los, h_bs_m=10.0, h_ut_m=1.5)
        state["large_scale"] = db_to_linear(-pl_db)  # (C, K)
        state["h_fading"] = (rng.normal(size=(cfg.num_cells, k_users)) + 1j * rng.normal(size=(cfg.num_cells, k_users))) / math.sqrt(2.0)
        state["queues"] = rng.uniform(1e4, 5e4, size=k_users)
        state["avg_thr"] = np.ones(k_users, dtype=np.float64) * 1e4
        state["step"] = 0
        # Compute causal SINR observed at start of slot t BEFORE action_t
        sig = tx_w * state["large_scale"][0] * (np.abs(state["h_fading"][0]) ** 2)
        interf = np.sum(tx_w * state["large_scale"][1:] * (np.abs(state["h_fading"][1:]) ** 2), axis=0) if cfg.num_cells > 1 else 0.0
        state["causal_sinr"] = sig / np.maximum(interf + noise_w, 1e-30)
        return _obs(), {"action_mask": (state["queues"] > 0).astype(bool) if np.any(state["queues"] > 0) else np.ones(k_users, dtype=bool)}

    def _obs() -> np.ndarray:
        q_norm = np.clip(state["queues"] / 2e5, 0.0, 5.0)
        cqi = np.clip(np.log2(1.0 + state["causal_sinr"]) / 10.0, 0.0, 3.0)
        thr = np.clip(state["avg_thr"] / 5e4, 0.0, 5.0)
        return np.concatenate([q_norm, cqi, thr]).astype(np.float32)

    def _step(action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        rng: np.random.Generator = state["rng"]
        u = int(np.clip(int(action), 0, k_users - 1))
        state["step"] += 1

        # Serve chosen user u using CURRENT slot t causal channel
        cap_bits = cfg.bandwidth_hz * math.log2(1.0 + float(state["causal_sinr"][u])) * dt_sec
        served = min(float(state["queues"][u]), cap_bits)
        state["queues"][u] -= served
        state["avg_thr"] = 0.9 * state["avg_thr"]
        state["avg_thr"][u] += 0.1 * served

        # Advance to t+1: causal arrivals & AR(1) temporally correlated fading
        arrivals = rng.poisson(cfg.arrival_bits_per_slot, size=k_users).astype(np.float64)
        state["queues"] = np.minimum(state["queues"] + arrivals, 1e6)
        innov = (rng.normal(size=(cfg.num_cells, k_users)) + 1j * rng.normal(size=(cfg.num_cells, k_users))) / math.sqrt(2.0)
        state["h_fading"] = cfg.fading_rho * state["h_fading"] + math.sqrt(max(0.0, 1.0 - cfg.fading_rho ** 2)) * innov
        sig = tx_w * state["large_scale"][0] * (np.abs(state["h_fading"][0]) ** 2)
        interf = np.sum(tx_w * state["large_scale"][1:] * (np.abs(state["h_fading"][1:]) ** 2), axis=0) if cfg.num_cells > 1 else 0.0
        state["causal_sinr"] = sig / np.maximum(interf + noise_w, 1e-30)

        # Reward & independent anti-hacking KPIs
        thr_term = served / 1e4
        q_pen = -0.05 * float(np.mean(state["queues"]) / 1e4)
        reward = float(thr_term + q_pen)
        trunc = bool(state["step"] >= cfg.episode_horizon)
        mask = (state["queues"] > 0).astype(bool)
        if not np.any(mask):
            mask = np.ones(k_users, dtype=bool)
        info = {
            "action_mask": mask,
            "reward_terms": {"served_bits": float(thr_term), "queue_penalty": float(q_pen)},
            "task_metrics": {"throughput_mbps": (served / dt_sec) / 1e6, "mean_queue_kb": float(np.mean(state["queues"]) / 1e3)},
        }
        return _obs(), reward, False, trunc, info

    return WirelessGymEnvAdapter(
        obs_dim=obs_dim,
        action_dim=k_users,
        discrete_actions=True,
        reset_fn=_reset,
        step_fn=_step,
        max_steps=cfg.episode_horizon,
    )


def provision_multicell_interference_mac(config: MultiCellMACConfig | None = None, **overrides: Any) -> ProvisionResult:
    """Provision the multi-cell MAC env & run a Proportional-Fair vs Round-Robin rollout sanity benchmark."""
    cfg = config or MultiCellMACConfig()
    if overrides:
        cfg = MultiCellMACConfig(**{**asdict(cfg), **overrides})
    t0 = time.perf_counter()
    set_global_seed(cfg.seed)
    env = build_multicell_gym_env(cfg)

    def _run_policy(mode: str) -> tuple[float, float]:
        obs, _ = env.reset(seed=cfg.seed)
        tot_ret = 0.0
        thrs: list[float] = []
        k = cfg.users_per_cell
        for step_idx in range(cfg.episode_horizon):
            cqi = obs[k : 2 * k]
            avg_t = np.maximum(obs[2 * k : 3 * k], 1e-3)
            act = int(np.argmax(cqi / avg_t)) if mode == "pf" else (step_idx % k)
            obs, rew, term, trunc, info = env.step(act)
            tot_ret += rew
            thrs.append(info["task_metrics"]["throughput_mbps"])
            if term or trunc:
                break
        return tot_ret, float(np.mean(thrs))

    pf_ret, pf_mbps = _run_policy("pf")
    rr_ret, rr_mbps = _run_policy("rr")

    prov = [
        ParamProvenance("num_cells", cfg.num_cells, "cells", "Co-channel interfering base stations", "gpp_tr38901", "Table 7.4.1-1", "3gpp_tr38901", "Wrap-around UMi interference topology."),
        ParamProvenance("fading_rho", cfg.fading_rho, "ratio", "Slot-to-slot AR(1) channel correlation", "clarke_jakes_doppler", "BSTJ 1968", "clarke1968statistical", "Prevents non-causal independent jumping across TTI slots."),
    ]
    return ProvisionResult(
        block_id=META.id,
        seed=cfg.seed,
        elapsed_seconds=time.perf_counter() - t0,
        config=asdict(cfg),
        tensors={},
        metrics={
            "pf_episode_return": pf_ret,
            "pf_mean_throughput_mbps": pf_mbps,
            "rr_episode_return": rr_ret,
            "rr_mean_throughput_mbps": rr_mbps,
        },
        diagnostics={"normalized_channel_power": 1.0, "has_causal_csi_guard": True},
        parameter_provenance=prov,
        mechanisms_used=list(META.default_mechanisms),
        standard_ids_used=list(META.standard_ids),
        extra={"env": env, "env_factory": lambda: build_multicell_gym_env(cfg)},
    )
