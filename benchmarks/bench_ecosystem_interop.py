"""Optional Standalone Ecosystem Interop Check: Gym Multi-User Power Control & Complex64 Baseband JSR.

Runs cleanly with ``wannado-wce`` alone (CI-safe) and exercises ``wannado.drl`` / ``wannado.adv``
when those independent sibling libraries happen to be installed in the environment.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch
from torch import nn

from wannado.wce import ComposedWirelessEnv, compose_wce_spec


class _CoherentQPSKReceiver(nn.Module):
    """Differentiable complex64 baseband QPSK matched-filter receiver built on WCE channel snapshots."""

    def __init__(self, channel_mod: nn.Module, h_snap: torch.Tensor) -> None:
        super().__init__()
        self.channel_mod = channel_mod
        self.register_buffer("h_snap", h_snap)
        constellation = torch.tensor(
            [1.0 + 1.0j, -1.0 + 1.0j, -1.0 - 1.0j, 1.0 - 1.0j],
            dtype=torch.complex64,
        ) / np.sqrt(2.0)
        self.register_buffer("constellation", constellation)

    def forward(self, rx_iq_2ch: torch.Tensor) -> torch.Tensor:
        """Accept real-imag ``(B, 2, Nr)`` RF waveform tensor and output 4-class QPSK logits."""
        y_complex = torch.complex(rx_iq_2ch[:, 0, :], rx_iq_2ch[:, 1, :]).to(torch.complex64)
        # h_snap: (B, Nr, Nt) -> use Tx antenna 0 spatial signature for coherent MRC combining
        h0 = (
            self.h_snap[:, :, 0]
            if self.h_snap.ndim == 3
            else self.h_snap.reshape(y_complex.shape[0], -1)[:, : y_complex.shape[-1]]
        )
        norm_sq = torch.sum(h0.abs().pow(2), dim=-1, keepdim=True).clamp(min=1e-6)
        s_hat = torch.sum(torch.conj(h0) * y_complex, dim=-1, keepdim=True) / norm_sq
        dist_sq = torch.abs(s_hat - self.constellation.unsqueeze(0)).pow(2)
        return -4.0 * dist_sq


def main() -> None:
    root = Path(__file__).resolve().parent
    outputs_dir = root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    rows: list[list[object]] = []

    # 1. Multi-user continuous power allocation Gym contract + action sensitivity
    spec, _ = compose_wce_spec("64-element passive RIS MIMO at 28 GHz for multi-user beamforming")
    env = ComposedWirelessEnv(spec, batch_size=4, seed=42)
    gym_env = env.as_gym_env(max_steps=20, multi_user=True)

    obs0, _ = gym_env.reset(seed=42)
    assert obs0.shape == (4 * 4 + 4,)
    assert gym_env.action_space.shape == (4,)

    obs_u, rew_uniform, _, _, _ = gym_env.step(np.zeros(4, dtype=np.float32))
    gym_env.reset(seed=42)
    obs_s, rew_skewed, _, _, _ = gym_env.step(np.array([4.0, -4.0, -4.0, -4.0], dtype=np.float32))

    # Multi-user obs packs [est_gain (B), per_user_snr_db (B), log1p_queue (B), is_los (B), summary (4)]
    snr_u = np.asarray(obs_u[4:8], dtype=float)
    snr_s = np.asarray(obs_s[4:8], dtype=float)
    user0_boost_db = float(snr_s[0] - snr_u[0])

    # Confirm exact 20-step truncation boundary
    gym_env.reset(seed=42)
    trunc_at_20 = False
    for step_idx in range(1, 21):
        _, _, _, trunc, _ = gym_env.step(np.zeros(4, dtype=np.float32))
        if step_idx < 20:
            assert not trunc
        else:
            trunc_at_20 = bool(trunc)
    assert trunc_at_20

    rows.extend(
        [
            ["wce_gym_multiuser", "uniform_step_reward", float(rew_uniform)],
            ["wce_gym_multiuser", "skewed_step_reward", float(rew_skewed)],
            ["wce_gym_multiuser", "user0_snr_boost_db", user0_boost_db],
            ["wce_gym_multiuser", "truncated_at_step_20", 1.0 if trunc_at_20 else 0.0],
        ]
    )

    # 2. Optional standalone wannado.drl interop check
    try:
        from wannado.drl import PPOConfig, train_ppo

        def _env_factory() -> Any:
            s, _ = compose_wce_spec("64-element passive RIS MIMO at 28 GHz for multi-user beamforming")
            return ComposedWirelessEnv(s, batch_size=4, seed=42).as_gym_env(max_steps=20, multi_user=True)

        train_res = train_ppo(
            _env_factory,
            PPOConfig(
                total_timesteps=256,
                n_steps=64,
                minibatch_size=32,
                update_epochs=2,
                hidden_dims=(32, 32),
                eval_episodes=2,
                seed=42,
                device="cpu",
            ),
        )
        mean_ret = float(train_res.eval_stats.mean_return)
        assert np.isfinite(mean_ret)
        rows.extend(
            [
                ["wannado_drl_ppo", "installed", 1.0],
                ["wannado_drl_ppo", "mean_eval_return", mean_ret],
                ["wannado_drl_ppo", "elapsed_seconds", float(train_res.elapsed_seconds)],
            ]
        )
    except ImportError:
        rows.append(["wannado_drl_ppo", "installed", 0.0])

    # 3. Complex64 baseband receiver + optional wannado.adv JSR robustness check
    torch.manual_seed(42)
    channel_mod = env.as_channel_module(snr_db=15.0, stateless=True)
    batch = 64
    h_snap = channel_mod.channel_sampler(batch).to(torch.complex64)
    nt = h_snap.shape[-1] if h_snap.ndim == 3 else 1
    labels = torch.randint(0, 4, (batch,))
    constellation = torch.tensor(
        [1.0 + 1.0j, -1.0 + 1.0j, -1.0 - 1.0j, 1.0 - 1.0j],
        dtype=torch.complex64,
    ) / np.sqrt(2.0)
    x_tx = torch.zeros((batch, nt), dtype=torch.complex64)
    x_tx[:, 0] = constellation[labels]
    y_rx = channel_mod(x_tx, h=h_snap)
    rx_iq_2ch = torch.stack([y_rx.real, y_rx.imag], dim=1)

    receiver = _CoherentQPSKReceiver(channel_mod, h_snap).eval()
    clean_preds = receiver(rx_iq_2ch).argmax(dim=-1)
    clean_acc = float((clean_preds == labels).float().mean().item())
    rows.append(["wce_complex64_receiver", "clean_qpsk_acc", clean_acc])

    try:
        from wannado.adv import pgd

        atk_low = pgd(receiver, rx_iq_2ch, labels, eps=-5.0, norm="jsr_db", steps=6, clip_bounds=None, seed=42)
        atk_high = pgd(receiver, rx_iq_2ch, labels, eps=3.0, norm="jsr_db", steps=6, clip_bounds=None, seed=42)
        acc_low = float((receiver(atk_low.x_adv).argmax(dim=-1) == labels).float().mean().item())
        acc_high = float((receiver(atk_high.x_adv).argmax(dim=-1) == labels).float().mean().item())
        assert clean_acc >= acc_high - 1e-6
        rows.extend(
            [
                ["wannado_adv_jsr", "installed", 1.0],
                ["wannado_adv_jsr", "qpsk_acc_jsr_minus5db", acc_low],
                ["wannado_adv_jsr", "qpsk_acc_jsr_plus3db", acc_high],
            ]
        )
    except ImportError:
        rows.append(["wannado_adv_jsr", "installed", 0.0])

    csv_path = outputs_dir / "ecosystem_interop_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["component", "metric", "value"])
        w.writerows(rows)

    print(f"Generated ecosystem interop metrics: {csv_path}")
    for comp, metric, val in rows:
        print(f"  [{comp}] {metric} = {val:.4f}")


if __name__ == "__main__":
    main()
