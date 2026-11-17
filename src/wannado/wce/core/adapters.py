"""Unified output adapters bridging provisioned environments to PyTorch, Gymnasium/DRL, and Adv eval."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


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
    dtype: np.dtype[Any] = np.dtype(np.float32)  # noqa: RUF009

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
    dtype: np.dtype[Any] = np.dtype(np.int64)  # noqa: RUF009

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
    np.savez_compressed(str(out), **np_dict)  # type: ignore[arg-type]
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


def adapt_external_channel_tensor(
    raw_channel: Any,
    *,
    source_layout: str = "auto",
    strip_large_scale_gain: bool = True,
    complex_pair_last_dim: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Normalize TensorFlow/NumPy/Torch simulator outputs into canonical ``torch.complex64``.

    Handles:
    - ``sionna_cir`` 7D layout ``(B, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time)``:
      sums over ``num_paths`` and folds singleton link dims or exports ``(B, Nr, Nt, T)``.
    - ``deepmimo`` / ``quadriga`` complex tensors ``(B, Nr, Nt, ...)`` or real/imag last-dim pairs.
    - Separates large-scale pathgain (`10^(-PL/10)` -> returned as optional per-sample gain) from
      unit-power small-scale fading (`E[|H|^2] = 1.0`) to prevent accidental 80-120 dB SNR inflation.
    """
    if isinstance(raw_channel, torch.Tensor):
        t = raw_channel.detach().clone()
    elif hasattr(raw_channel, "numpy"):
        t = torch.from_numpy(np.asarray(raw_channel.numpy()))
    else:
        t = torch.from_numpy(np.asarray(raw_channel))

    if complex_pair_last_dim and not t.is_complex() and t.shape[-1] == 2:
        t = torch.complex(t[..., 0].float(), t[..., 1].float())
    elif not t.is_complex():
        t = t.to(torch.complex64)
    else:
        t = t.to(torch.complex64)

    if source_layout == "sionna_cir" or (source_layout == "auto" and t.ndim == 7):
        # (B, rx, rx_ant, tx, tx_ant, paths, time) -> sum paths -> squeeze single rx/tx -> (B, Nr, Nt, T)
        h_sum = t.sum(dim=5)  # sum over paths
        b, n_rx, rx_a, n_tx, tx_a, n_t = h_sum.shape
        t = h_sum.permute(0, 1, 2, 3, 4, 5).reshape(b, n_rx * rx_a, n_tx * tx_a, n_t)
        if n_t == 1:
            t = t.squeeze(-1)
    elif t.ndim == 1:
        t = t.unsqueeze(-1).unsqueeze(-1)
    elif t.ndim == 2:
        # (B, N_ant) -> (B, 1, N_ant)
        t = t.unsqueeze(1)

    if not strip_large_scale_gain:
        return t, None

    reduce_dims = tuple(range(1, t.ndim))
    sample_pow = torch.mean(t.abs().pow(2), dim=reduce_dims, keepdim=True)
    global_pow = float(torch.mean(sample_pow).item())
    if global_pow <= 0.0 or not np.isfinite(global_pow):
        raise ValueError(f"Non-finite or zero external channel power ({global_pow}); cannot normalize safely.")
    scale = torch.sqrt(sample_pow.clamp(min=1e-24))
    h_small = t / scale
    large_scale_power = sample_pow.reshape(t.shape[0])
    return h_small, large_scale_power


def verify_external_channel_adapter(
    h_tensor: Any,
    *,
    snr_db: float = 15.0,
    expect_normalized: bool = True,
) -> dict[str, Any]:
    """Run built-in contract checks on an adapted external channel tensor (finiteness, power, autograd)."""
    warnings: list[str] = []
    errors: list[str] = []

    h_t = h_tensor if isinstance(h_tensor, torch.Tensor) else torch.as_tensor(h_tensor)
    if not h_t.is_complex():
        errors.append(f"Expected complex64 channel tensor, got dtype={h_t.dtype}.")
    if not (torch.isfinite(h_t.real).all().item() and torch.isfinite(h_t.imag).all().item()):
        errors.append("Adapted channel tensor contains NaN or Inf entries.")

    mean_pow = float(torch.mean(h_t.abs().pow(2)).item()) if h_t.numel() > 0 else 0.0
    if expect_normalized and not (0.5 <= mean_pow <= 1.5):
        errors.append(
            f"Channel power normalization drift: mean |H|^2 = {mean_pow:.4e} (expected ~1.0 in [0.5, 1.5]). "
            "Separate large-scale pathloss via adapt_external_channel_tensor(..., strip_large_scale_gain=True)."
        )

    autograd_ok = False
    if h_t.ndim >= 3 and h_t.shape[0] >= 1 and not errors:
        h_3d = h_t[..., 0] if h_t.ndim > 3 else h_t
        n_tx = h_3d.shape[-1]
        x = torch.randn((h_3d.shape[0], n_tx), dtype=torch.float32) + 1j * torch.randn((h_3d.shape[0], n_tx), dtype=torch.float32)
        x = x.to(torch.complex64).detach().requires_grad_(True)
        adapter = ChannelModuleAdapter(lambda b: h_3d[:b], snr_db=snr_db, add_noise=True)
        y = adapter(x)
        loss = torch.mean(y.abs().pow(2))
        loss.backward()
        autograd_ok = x.grad is not None and bool(torch.isfinite(x.grad.real).all().item())
        if not autograd_ok:
            warnings.append("ChannelModuleAdapter autograd check produced None or non-finite gradients.")
    elif h_t.ndim < 3:
        warnings.append(f"Channel tensor rank={h_t.ndim} < 3; reshape to (batch, num_rx, num_tx) for full MIMO support.")

    return {
        "ok": len(errors) == 0,
        "shape": list(h_t.shape),
        "dtype": str(h_t.dtype),
        "mean_channel_power": mean_pow,
        "autograd_ok": autograd_ok,
        "warnings": warnings,
        "errors": errors,
    }
