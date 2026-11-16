"""Unified output adapters bridging provisioned environments to PyTorch, Gymnasium/DRL, and Adv eval."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn


class ChannelModuleAdapter(nn.Module):
    """Differentiable PyTorch nn.Module wrapping complex baseband wireless propagation + AWGN.

    Plug-and-play compatible with ``wannado.adv.VictimInterface`` (``complex64`` IQ, ``norm='jsr_db'``)
    and end-to-end semantic/neural receiver training.
    """

    def __init__(
        self,
        channel_sampler: Callable[[int], torch.Tensor],
        *,
        snr_db: float = 15.0,
        add_noise: bool = True,
    ) -> None:
        super().__init__()
        self.channel_sampler = channel_sampler
        self.snr_db = float(snr_db)
        self.add_noise = bool(add_noise)

    def forward(self, x_tx: torch.Tensor, *, h: torch.Tensor | None = None) -> torch.Tensor:
        batch = x_tx.shape[0]
        h_eff = h if h is not None else self.channel_sampler(batch).to(x_tx.device)
        if h_eff.ndim == x_tx.ndim + 1 and x_tx.ndim == 2:
            # Matrix-vector product y = H x for MIMO (B, Nr, Nt) @ (B, Nt, 1)
            y = torch.bmm(h_eff.to(torch.complex64), x_tx.to(torch.complex64).unsqueeze(-1)).squeeze(-1)
        elif h_eff.shape == x_tx.shape or h_eff.ndim <= x_tx.ndim:
            y = h_eff.to(torch.complex64) * x_tx.to(torch.complex64)
        else:
            y = h_eff.to(torch.complex64).reshape(batch, -1)[:, : x_tx.shape[-1]] * x_tx.to(torch.complex64)

        if not self.add_noise:
            return y

        sig_pow = torch.mean(y.abs().pow(2)).clamp(min=1e-12)
        noise_var = sig_pow / (10.0 ** (self.snr_db / 10.0))
        std_each = torch.sqrt(noise_var / 2.0)
        noise = torch.complex(
            torch.randn_like(y.real) * std_each,
            torch.randn_like(y.imag) * std_each,
        )
        return y + noise


@dataclass
class SimpleBoxSpace:
    """Zero-dependency Box space duck-type compatible with Gymnasium & wannado-drl UniversalGymAdapter."""

    low: np.ndarray
    high: np.ndarray
    shape: tuple[int, ...]
    dtype: np.dtype = np.dtype(np.float32)

    def sample(self, rng: np.random.Generator | None = None) -> np.ndarray:
        gen = rng if rng is not None else np.random.default_rng()
        lo = np.where(np.isfinite(self.low), self.low, -1.0)
        hi = np.where(np.isfinite(self.high), self.high, 1.0)
        return gen.uniform(lo, hi, size=self.shape).astype(self.dtype)


@dataclass
class SimpleDiscreteSpace:
    """Zero-dependency Discrete space duck-type compatible with Gymnasium & wannado-drl."""

    n: int
    shape: tuple[int, ...] = ()
    dtype: np.dtype = np.dtype(np.int64)

    def sample(self, rng: np.random.Generator | None = None) -> int:
        gen = rng if rng is not None else np.random.default_rng()
        return int(gen.integers(0, self.n))


class WirelessGymEnvAdapter:
    """Lightweight Gymnasium-compatible environment wrapper (works with or without gymnasium installed)."""

    def __init__(
        self,
        *,
        obs_dim: int,
        action_dim: int,
        discrete_actions: bool = True,
        reset_fn: Callable[[int | None], tuple[np.ndarray, dict[str, Any]]] | None = None,
        step_fn: Callable[[Any], tuple[np.ndarray, float, bool, bool, dict[str, Any]]] | None = None,
        max_steps: int = 200,
    ) -> None:
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.discrete_actions = bool(discrete_actions)
        self._reset_fn = reset_fn
        self._step_fn = step_fn
        self.max_steps = int(max_steps)
        self._step_count = 0
        self._rng = np.random.default_rng(0)

        try:
            from gymnasium import spaces

            self.observation_space: Any = spaces.Box(
                low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
            )
            self.action_space: Any = (
                spaces.Discrete(self.action_dim)
                if self.discrete_actions
                else spaces.Box(low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32)
            )
        except Exception:  # noqa: BLE001
            self.observation_space = SimpleBoxSpace(
                low=np.full(self.obs_dim, -np.inf, dtype=np.float32),
                high=np.full(self.obs_dim, np.inf, dtype=np.float32),
                shape=(self.obs_dim,),
            )
            self.action_space = (
                SimpleDiscreteSpace(n=self.action_dim)
                if self.discrete_actions
                else SimpleBoxSpace(
                    low=-np.ones(self.action_dim, dtype=np.float32),
                    high=np.ones(self.action_dim, dtype=np.float32),
                    shape=(self.action_dim,),
                )
            )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._step_count = 0
        if self._reset_fn is not None:
            return self._reset_fn(seed)
        return np.zeros(self.obs_dim, dtype=np.float32), {}

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self._step_count += 1
        if self._step_fn is not None:
            return self._step_fn(action)
        obs = np.zeros(self.obs_dim, dtype=np.float32)
        trunc = self._step_count >= self.max_steps
        return obs, 0.0, False, trunc, {}


def export_dataset_npz(tensors: dict[str, Any], path: str | Path) -> Path:
    """Save provisioned PyTorch/NumPy channel & traffic arrays to a compressed .npz archive."""
    out = Path(path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    np_dict = {
        k: (v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v))
        for k, v in tensors.items()
    }
    np.savez_compressed(out, **np_dict)
    return out


def export_dataset_pt(tensors: dict[str, Any], path: str | Path) -> Path:
    """Save provisioned tensors cleanly as a PyTorch .pt state bundle."""
    out = Path(path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    pt_dict = {
        k: (v.detach().cpu() if isinstance(v, torch.Tensor) else torch.as_tensor(v))
        for k, v in tensors.items()
    }
    torch.save(pt_dict, out)
    return out
