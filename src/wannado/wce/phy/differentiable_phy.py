"""PyTorch-Native End-to-End Differentiable Wireless Link, Soft Demapper, and RIS Autograd Engine."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from wannado.wce.workspace.link_level_phy import ModulationOrder, build_qam_constellation


def compute_bit_metric_rate(llrs: torch.Tensor, bits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute differentiable BICM BCE loss and Achievable Information Rate (bits/channel-use).

    Uses ``LLR > 0`` favoring bit ``0`` (matching ``wannado-wce`` demappers): ``log P(b=1)/P(b=0) = -LLR``.
    """
    b_f = bits.to(dtype=torch.float32)
    # Sum over bits-per-symbol axis if last dim > 1, returning exact bits/symbol BICM AIR
    bce_nats = F.binary_cross_entropy_with_logits(-llrs.to(torch.float32), b_f, reduction="mean")
    bps = float(bits.shape[-1]) if bits.ndim >= 2 else 1.0
    air_bps_sym = bps * (1.0 - bce_nats / math.log(2.0))
    return bce_nats, air_bps_sym


class DifferentiableSoftMapperDemapper(nn.Module):
    """End-to-end differentiable Gray QAM mapper (optional geometric constellation shaping) + Log-MAP LLR demapper."""

    def __init__(
        self,
        modulation: ModulationOrder = "16QAM",
        *,
        learnable_constellation: bool = False,
    ) -> None:
        super().__init__()
        self.modulation = modulation
        base_sym, bit_table = build_qam_constellation(modulation, device="cpu")
        self.bps = int(bit_table.shape[1])
        self.num_points = int(base_sym.shape[0])

        self.register_buffer("base_constellation", base_sym)
        self.register_buffer("bit_table", bit_table)
        powers = (1 << torch.arange(self.bps - 1, -1, -1, dtype=torch.int64))
        self.register_buffer("powers", powers)

        bit_key = (bit_table * powers.unsqueeze(0)).sum(dim=-1)
        inv_lut = torch.zeros(self.num_points, dtype=torch.int64)
        inv_lut[bit_key] = torch.arange(self.num_points, dtype=torch.int64)
        self.register_buffer("inv_lut", inv_lut)

        self.learnable_constellation = bool(learnable_constellation)
        if self.learnable_constellation:
            self.constellation_delta = nn.Parameter(torch.zeros(self.num_points, 2, dtype=torch.float32))
        else:
            self.register_parameter("constellation_delta", None)

    def get_constellation(self) -> torch.Tensor:
        """Return current unit-average-power constellation ``(M,) complex64`` with autograd."""
        base: torch.Tensor = self.base_constellation  # type: ignore[assignment]
        if self.constellation_delta is None:
            return base
        offset = torch.complex(self.constellation_delta[:, 0], self.constellation_delta[:, 1])
        raw = base + offset
        power = torch.mean(raw.real.pow(2) + raw.imag.pow(2)).clamp(min=1e-12)
        return raw / torch.sqrt(power)

    def encode(self, bits: torch.Tensor) -> torch.Tensor:
        """Map binary bit tensor ``(..., bps)`` to complex constellation symbols ``(...,)``."""
        lut: torch.Tensor = self.inv_lut  # type: ignore[assignment]
        pw: torch.Tensor = self.powers  # type: ignore[assignment]
        int_bits = (bits > 0.5).to(torch.int64)
        idx = lut[(int_bits * pw).sum(dim=-1)]
        return self.get_constellation()[idx]

    def demap_llr(self, x_hat: torch.Tensor, sinr: torch.Tensor | float) -> torch.Tensor:
        """Differentiable Log-MAP LLR demapper preserving gradients to equalized symbols & constellation."""
        sym = self.get_constellation()  # (M,)
        bt: torch.Tensor = self.bit_table  # type: ignore[assignment]
        diff = x_hat.unsqueeze(-1) - sym.view((1,) * x_hat.ndim + (self.num_points,))
        dist2 = diff.real.pow(2) + diff.imag.pow(2)  # autograd-safe squared Euclidean distance

        sinr_t = sinr if isinstance(sinr, torch.Tensor) else torch.tensor(float(sinr), device=x_hat.device)
        sinr_t = sinr_t.to(device=x_hat.device, dtype=torch.float32).unsqueeze(-1)
        scaled = -sinr_t * dist2
        # Shift by detached max log-prob so exp is strictly <= 1.0 (numerically exact Log-MAP)
        shifted = scaled - scaled.amax(dim=-1, keepdim=True).detach()

        llrs: list[torch.Tensor] = []
        for b_idx in range(self.bps):
            mask_0 = bt[:, b_idx] == 0
            lse0 = torch.logsumexp(shifted[..., mask_0], dim=-1)
            lse1 = torch.logsumexp(shifted[..., ~mask_0], dim=-1)
            llrs.append(torch.clamp(lse0 - lse1, -30.0, 30.0))
        return torch.stack(llrs, dim=-1)


class DifferentiableRISChannel(nn.Module):
    """Differentiable 2-hop MIMO Reconfigurable Intelligent Surface cascade ``H_d + eta * H_r @ Phi @ G``."""

    def __init__(
        self,
        num_ris_elements: int,
        *,
        coupling_scale: float = 0.08,
        amplitude_gain: float = 1.0,
        active_ris: bool = False,
        learnable_phases: bool = True,
    ) -> None:
        super().__init__()
        self.num_ris_elements = int(num_ris_elements)
        self.coupling_scale = float(coupling_scale)
        self.amplitude_gain = float(amplitude_gain)
        self.active_ris = bool(active_ris)
        if learnable_phases:
            self.ris_phases = nn.Parameter(torch.zeros(self.num_ris_elements, dtype=torch.float32))
        else:
            self.register_buffer("ris_phases", torch.zeros(self.num_ris_elements, dtype=torch.float32))

    def forward_channel(
        self,
        h_direct: torch.Tensor,
        h_ris_rx: torch.Tensor,
        h_bs_ris: torch.Tensor,
        *,
        ris_phases: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Synthesize cascaded MIMO channel ``(B, Nr, Nt)`` with unit-modulus passive phase autograd."""
        theta = ris_phases if ris_phases is not None else self.ris_phases
        if not self.active_ris and self.amplitude_gain > 1.0 + 1e-6:
            raise ValueError("Passive RIS amplitude_gain must be <= 1.0 by energy conservation.")
        phi = float(self.amplitude_gain) * torch.complex(torch.cos(theta), torch.sin(theta)).to(torch.complex64)
        cascade = torch.einsum("bnm,m,bmk->bnk", h_ris_rx.to(torch.complex64), phi, h_bs_ris.to(torch.complex64))
        return h_direct.to(torch.complex64) + self.coupling_scale * cascade


class DifferentiableWirelessLink(nn.Module):
    """End-to-end PyTorch GPU-ready differentiable transceiver + RF + MIMO/RIS channel + Soft Demapper."""

    def __init__(
        self,
        modulation: ModulationOrder = "16QAM",
        *,
        equalizer: str = "mmse",
        learnable_constellation: bool = False,
        rf_impairments: nn.Module | None = None,
        neural_equalizer: nn.Module | None = None,
        ris_channel: DifferentiableRISChannel | None = None,
    ) -> None:
        super().__init__()
        self.mapper_demapper = DifferentiableSoftMapperDemapper(
            modulation,
            learnable_constellation=learnable_constellation,
        )
        self.equalizer = equalizer
        self.rf_impairments = rf_impairments
        self.neural_equalizer = neural_equalizer
        self.ris_channel = ris_channel

    def forward(
        self,
        bits: torch.Tensor,
        *,
        h_mimo: torch.Tensor | None = None,
        h_direct: torch.Tensor | None = None,
        h_ris_rx: torch.Tensor | None = None,
        h_bs_ris: torch.Tensor | None = None,
        precoder: torch.Tensor | None = None,
        snr_db: float = 12.0,
    ) -> dict[str, Any]:
        """Run full end-to-end forward pass preserving gradients w.r.t. TX, Precoder, RF, RIS & Neural RX.

        Expected ``bits`` shape: ``(B, U, Ns, bps)`` where ``Ns`` matches transmitted streams.
        """
        if h_mimo is None:
            if self.ris_channel is None or h_direct is None or h_ris_rx is None or h_bs_ris is None:
                raise ValueError("Provide either h_mimo or (ris_channel + h_direct + h_ris_rx + h_bs_ris).")
            h_chan = self.ris_channel.forward_channel(h_direct, h_ris_rx, h_bs_ris)
        else:
            h_chan = h_mimo.to(torch.complex64)

        if h_chan.ndim != 3:
            raise ValueError(f"Expected 3D MIMO channel tensor (B, Nr, Nt), got {tuple(h_chan.shape)}.")
        x_syms = self.mapper_demapper.encode(bits)  # (B, U, Ns)
        if x_syms.ndim == 2:
            x_syms = x_syms.unsqueeze(-1)
        n_streams = x_syms.shape[-1]

        if precoder is not None:
            # Normalize per-batch precoder Frobenius power ||W||_F^2 == n_streams (or total unit power)
            w_p = precoder.to(torch.complex64)
            w_pow = torch.sum(w_p.real.pow(2) + w_p.imag.pow(2), dim=(-2, -1), keepdim=True).clamp(min=1e-12)
            w_norm = w_p / torch.sqrt(w_pow)
            h_eff = torch.matmul(h_chan, w_norm)  # (B, Nr, Ns)
        else:
            h_eff = h_chan[:, :, :n_streams] / math.sqrt(float(max(1, n_streams)))

        # Apply channel across symbol uses U: y_clean[B, U, Nr]
        y_clean = torch.einsum("bnm,bum->bun", h_eff, x_syms)
        if self.rf_impairments is not None:
            y_clean = self.rf_impairments(y_clean.transpose(1, 2)).transpose(1, 2)

        snr_lin = 10.0 ** (float(snr_db) / 10.0)
        noise_var = 1.0 / max(snr_lin, 1e-12)
        std_rx = math.sqrt(noise_var / 2.0)
        noise = torch.complex(
            torch.randn_like(y_clean.real) * std_rx,
            torch.randn_like(y_clean.imag) * std_rx,
        )
        y_rx = y_clean + noise

        if self.neural_equalizer is not None:
            x_hat = self.neural_equalizer(y_rx, h_eff, noise_var)
            post_sinr = torch.full(x_hat.shape, snr_lin, device=x_hat.device, dtype=torch.float32)
        else:
            gram = torch.matmul(h_eff.conj().transpose(-1, -2), h_eff)
            reg = noise_var if self.equalizer == "mmse" else 1e-5
            eye = torch.eye(n_streams, device=h_eff.device, dtype=torch.complex64).unsqueeze(0)
            w_eq = torch.matmul(torch.linalg.inv(gram + reg * eye), h_eff.conj().transpose(-1, -2))  # (B, Ns, Nr)
            x_raw = torch.einsum("bmr,bur->bum", w_eq, y_rx)
            wh = torch.matmul(w_eq, h_eff)  # (B, Ns, Ns)
            diag_gain = torch.diagonal(wh, dim1=-2, dim2=-1).real.clamp(min=1e-5)  # (B, Ns)
            x_hat = x_raw / diag_gain.unsqueeze(1)
            interf = torch.sum((wh - torch.diag_embed(torch.diagonal(wh, dim1=-2, dim2=-1))).real.pow(2) + (wh - torch.diag_embed(torch.diagonal(wh, dim1=-2, dim2=-1))).imag.pow(2), dim=-1)
            noise_enh = noise_var * torch.sum(w_eq.real.pow(2) + w_eq.imag.pow(2), dim=-1)
            post_sinr = (diag_gain.pow(2) / (interf + noise_enh).clamp(min=1e-12)).unsqueeze(1).expand_as(x_hat.real)

        llrs = self.mapper_demapper.demap_llr(x_hat, post_sinr)
        bce_loss, bicm_air = compute_bit_metric_rate(llrs, bits)
        evm_sq = torch.mean((x_hat - x_syms).real.pow(2) + (x_hat - x_syms).imag.pow(2))
        evm_pct = 100.0 * torch.sqrt(evm_sq.clamp(min=1e-12))

        return {
            "llrs": llrs,
            "x_hat": x_hat,
            "bce_loss": bce_loss,
            "bicm_air_bps_sym": bicm_air,
            "evm_pct": evm_pct,
            "h_eff": h_eff,
        }


__all__ = [
    "DifferentiableRISChannel",
    "DifferentiableSoftMapperDemapper",
    "DifferentiableWirelessLink",
    "compute_bit_metric_rate",
]
