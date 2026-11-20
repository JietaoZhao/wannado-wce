"""Differentiable PyTorch RF Front-End Hardware Impairments (PA, Phase Noise, CFO, IQ Imbalance, ADC)."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn

from wannado.wce.core.utils import make_torch_generator


@dataclass
class RFImpairmentConfig:
    """Physically parameterized RF front-end impairment configuration."""

    pa_enabled: bool = True
    pa_ibo_db: float = 6.0
    pa_smoothness_p: float = 2.0
    pa_phase_alpha: float = 0.0
    pa_phase_beta: float = 1.0
    phase_noise_enabled: bool = True
    phase_noise_dbc_hz: float = -95.0
    phase_noise_offset_hz: float = 100e3
    sample_rate_hz: float = 30.72e6
    cfo_hz: float = 0.0
    cfo_ppm: float = 0.0
    carrier_fc_ghz: float = 28.0
    iq_gain_imbalance_db: float = 0.0
    iq_phase_imbalance_deg: float = 0.0
    adc_bits: int | None = None
    adc_clip_sigma: float = 3.0

    def effective_cfo_hz(self) -> float:
        """Resolve carrier frequency offset in Hz from explicit ``cfo_hz`` or ``cfo_ppm``."""
        if abs(float(self.cfo_hz)) > 0.0:
            return float(self.cfo_hz)
        return float(self.cfo_ppm) * 1e-6 * float(self.carrier_fc_ghz) * 1e9

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _as_complex(x: torch.Tensor) -> torch.Tensor:
    if x.is_complex():
        return x
    return torch.complex(x.to(torch.float32), torch.zeros_like(x, dtype=torch.float32))


def _safe_mag(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Autograd-safe complex magnitude avoiding zero-amplitude square-root NaN gradients."""
    return torch.sqrt(x.real.pow(2) + x.imag.pow(2) + eps)


def apply_rapp_pa(
    x: torch.Tensor,
    *,
    ibo_db: float = 6.0,
    smoothness_p: float = 2.0,
    phase_alpha: float = 0.0,
    phase_beta: float = 1.0,
) -> torch.Tensor:
    """Apply differentiable Rapp Solid-State Power Amplifier (SSPA) AM/AM + optional Saleh AM/PM distortion.

    Input ``ibo_db`` sets the saturation amplitude ``A_sat = rms(|x|) * 10^(ibo_db / 20)`` relative to the
    detached per-sample RMS amplitude along the last dimension so small-signal unity gain holds for large IBO.
    """
    xc = _as_complex(x)
    p = max(0.5, float(smoothness_p))
    mag = _safe_mag(xc)
    rms = torch.sqrt(torch.mean((xc.real.pow(2) + xc.imag.pow(2)).detach(), dim=-1, keepdim=True).clamp(min=1e-12))
    a_sat = rms * (10.0 ** (float(ibo_db) / 20.0))
    r_norm = (mag / a_sat.clamp(min=1e-9)).clamp(max=1e4)
    am_gain = torch.pow(1.0 + torch.pow(r_norm, 2.0 * p), -0.5 / p)
    y = xc * am_gain.to(xc.real.dtype)

    if abs(float(phase_alpha)) > 1e-9:
        r2 = r_norm.pow(2)
        phi = float(phase_alpha) * r2 / (1.0 + max(1e-6, float(phase_beta)) * r2)
        y = y * torch.complex(torch.cos(phi), torch.sin(phi)).to(xc.dtype)
    return y


def apply_phase_noise(
    x: torch.Tensor,
    *,
    phase_noise_dbc_hz: float = -95.0,
    freq_offset_hz: float = 100e3,
    sample_rate_hz: float = 30.72e6,
    seed: int | None = None,
) -> torch.Tensor:
    """Apply Wiener free-running oscillator + white phase noise along the last sample axis."""
    xc = _as_complex(x)
    fs = max(float(sample_rate_hz), 1.0)
    l_lin = 10.0 ** (float(phase_noise_dbc_hz) / 10.0)
    # 3 dB Lorentzian oscillator linewidth Delta_f = 2*pi * f_off^2 * 10^(L/10)
    linewidth_hz = 2.0 * math.pi * (float(freq_offset_hz) ** 2) * l_lin
    sigma_rw = math.sqrt(max(2.0 * math.pi * linewidth_hz / fs, 0.0))
    # Integrated DSB white thermal floor variance over Nyquist bandwidth fs
    sigma_white = math.sqrt(max(fs * l_lin, 0.0))

    gen = make_torch_generator(seed, xc.device) if seed is not None else None
    noise_rw = torch.randn(xc.shape, device=xc.device, dtype=xc.real.dtype, generator=gen) * sigma_rw
    noise_rw[..., 0] = 0.0
    phi = torch.cumsum(noise_rw, dim=-1)
    if sigma_white > 0.0:
        phi = phi + torch.randn(xc.shape, device=xc.device, dtype=xc.real.dtype, generator=gen) * sigma_white
    rot = torch.complex(torch.cos(phi), torch.sin(phi)).to(xc.dtype)
    return xc * rot


def apply_cfo(
    x: torch.Tensor,
    *,
    cfo_hz: float = 0.0,
    sample_rate_hz: float = 30.72e6,
) -> torch.Tensor:
    """Apply deterministic carrier frequency offset ``exp(j * 2*pi * df * n / fs)`` across sample index ``n``."""
    xc = _as_complex(x)
    if abs(float(cfo_hz)) <= 0.0:
        return xc
    fs = max(float(sample_rate_hz), 1.0)
    n_idx = torch.arange(xc.shape[-1], device=xc.device, dtype=xc.real.dtype)
    shape = (1,) * (xc.ndim - 1) + (xc.shape[-1],)
    angle = (2.0 * math.pi * float(cfo_hz) / fs) * n_idx.view(shape)
    return xc * torch.complex(torch.cos(angle), torch.sin(angle)).to(xc.dtype)


def apply_iq_imbalance(
    x: torch.Tensor,
    *,
    gain_imbalance_db: float = 0.0,
    phase_imbalance_deg: float = 0.0,
) -> torch.Tensor:
    """Apply power-normalized I/Q gain and quadrature phase mismatch (exact Image Rejection Ratio)."""
    xc = _as_complex(x)
    if abs(float(gain_imbalance_db)) <= 0.0 and abs(float(phase_imbalance_deg)) <= 0.0:
        return xc
    g = 10.0 ** (float(gain_imbalance_db) / 20.0)
    theta = math.radians(float(phase_imbalance_deg))
    c0 = math.sqrt(2.0 / (1.0 + g * g))
    c1 = g * c0
    half_t = 0.5 * theta
    cos_h = math.cos(half_t)
    sin_h = math.sin(half_t)
    xi = xc.real
    xq = xc.imag
    yi = c0 * (cos_h * xi - sin_h * xq)
    yq = c1 * (cos_h * xq - sin_h * xi)
    return torch.complex(yi, yq)


def apply_adc_quantization(
    x: torch.Tensor,
    *,
    num_bits: int = 8,
    clip_sigma: float = 3.0,
) -> torch.Tensor:
    """Uniform mid-rise I/Q ADC quantizer using Straight-Through Estimator (STE) for finite autograd."""
    xc = _as_complex(x)
    bits = max(1, min(16, int(num_bits)))
    levels = float(1 << bits)
    sigma_rail = torch.sqrt(
        0.5 * torch.mean((xc.real.pow(2) + xc.imag.pow(2)).detach(), dim=-1, keepdim=True).clamp(min=1e-12)
    )
    v_max = float(clip_sigma) * sigma_rail
    delta = (2.0 * v_max) / levels
    half_max = (levels * 0.5 - 0.5) * delta

    def _quant(real_part: torch.Tensor) -> torch.Tensor:
        idx = torch.floor(torch.clamp(real_part, -v_max, v_max) / delta)
        q = torch.clamp((idx + 0.5) * delta, -half_max, half_max)
        return real_part + (q - real_part).detach()

    return torch.complex(_quant(xc.real), _quant(xc.imag))


def measure_papr_db(x: torch.Tensor) -> torch.Tensor:
    """Compute per-waveform Peak-to-Average Power Ratio (dB) along the last dimension."""
    xc = _as_complex(x)
    inst_p = xc.real.pow(2) + xc.imag.pow(2)
    peak = torch.amax(inst_p, dim=-1)
    avg = torch.mean(inst_p, dim=-1).clamp(min=1e-12)
    return 10.0 * torch.log10(peak.clamp(min=1e-12) / avg)


def measure_aclr_db(x: torch.Tensor, *, occupied_fraction: float = 0.5) -> float:
    """Compute Adjacent Channel Leakage Ratio ``10*log10(P_inband / P_oob)`` (higher dB = cleaner spectrum)."""
    xc = _as_complex(x)
    n_fft = xc.shape[-1]
    if n_fft < 4:
        return 60.0
    frac = min(max(float(occupied_fraction), 0.1), 0.9)
    n_in = max(2, round(frac * n_fft))
    start = (n_fft - n_in) // 2
    end = start + n_in

    spec = torch.fft.fftshift(torch.fft.fft(xc, dim=-1), dim=-1)
    psd = spec.real.pow(2) + spec.imag.pow(2)
    in_mask = torch.zeros(n_fft, dtype=torch.bool, device=xc.device)
    in_mask[start:end] = True
    p_in = torch.mean(psd[..., in_mask]).clamp(min=1e-12)
    p_oob = torch.mean(psd[..., ~in_mask]).clamp(min=1e-12)
    return float(10.0 * torch.log10(p_in / p_oob).item())


class RFImpairmentChain(nn.Module):
    """Composable differentiable TX/RX RF front-end impairment module."""

    def __init__(self, config: RFImpairmentConfig | None = None, *, seed: int | None = None) -> None:
        super().__init__()
        self.config = config or RFImpairmentConfig()
        self.seed = seed
        self._call_count = 0

    @classmethod
    def from_config(cls, config: RFImpairmentConfig, *, seed: int | None = None) -> RFImpairmentChain:
        return cls(config=config, seed=seed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = _as_complex(x)
        cfg = self.config
        if abs(float(cfg.iq_gain_imbalance_db)) > 0.0 or abs(float(cfg.iq_phase_imbalance_deg)) > 0.0:
            y = apply_iq_imbalance(
                y,
                gain_imbalance_db=cfg.iq_gain_imbalance_db,
                phase_imbalance_deg=cfg.iq_phase_imbalance_deg,
            )
        if cfg.pa_enabled:
            y = apply_rapp_pa(
                y,
                ibo_db=cfg.pa_ibo_db,
                smoothness_p=cfg.pa_smoothness_p,
                phase_alpha=cfg.pa_phase_alpha,
                phase_beta=cfg.pa_phase_beta,
            )
        cfo = cfg.effective_cfo_hz()
        if abs(cfo) > 0.0:
            y = apply_cfo(y, cfo_hz=cfo, sample_rate_hz=cfg.sample_rate_hz)
        if cfg.phase_noise_enabled:
            call_seed = None if self.seed is None else (int(self.seed) + self._call_count)
            self._call_count += 1
            y = apply_phase_noise(
                y,
                phase_noise_dbc_hz=cfg.phase_noise_dbc_hz,
                freq_offset_hz=cfg.phase_noise_offset_hz,
                sample_rate_hz=cfg.sample_rate_hz,
                seed=call_seed,
            )
        if cfg.adc_bits is not None:
            y = apply_adc_quantization(y, num_bits=cfg.adc_bits, clip_sigma=cfg.adc_clip_sigma)
        return y

    def measure_distortion(
        self,
        x_clean: torch.Tensor,
        x_impaired: torch.Tensor | None = None,
        *,
        occupied_fraction: float = 0.5,
    ) -> dict[str, float]:
        """Compute quantitative RF distortion KPIs (EVM %, PAPR in/out dB, ACLR in/out dB, IRR dB)."""
        xc = _as_complex(x_clean)
        yi = _as_complex(x_impaired) if x_impaired is not None else self.forward(xc)
        # Optimal complex scalar gain fit (excluding phase/gain bias when measuring distortion EVM)
        num = torch.sum(xc.conj() * yi, dim=-1, keepdim=True)
        den = torch.sum(xc.real.pow(2) + xc.imag.pow(2), dim=-1, keepdim=True).clamp(min=1e-12)
        g_fit = num / den
        err_pow = torch.mean((yi - g_fit * xc).abs().pow(2)).item()
        ref_pow = max(float(torch.mean(xc.abs().pow(2)).item()), 1e-12)
        evm_pct = 100.0 * math.sqrt(max(err_pow / ref_pow, 0.0))

        # Analytical IQ Image Rejection Ratio (IRR) in dB
        g_iq = 10.0 ** (float(self.config.iq_gain_imbalance_db) / 20.0)
        theta_iq = math.radians(float(self.config.iq_phase_imbalance_deg))
        c0 = math.sqrt(2.0 / (1.0 + g_iq * g_iq))
        c1 = g_iq * c0
        num_irr = (c0 * c0 + c1 * c1) + 2.0 * c0 * c1 * math.cos(theta_iq)
        den_irr = max((c0 * c0 + c1 * c1) - 2.0 * c0 * c1 * math.cos(theta_iq), 1e-12)
        irr_db = 10.0 * math.log10(num_irr / den_irr)

        return {
            "evm_pct": float(evm_pct),
            "papr_in_db": float(torch.mean(measure_papr_db(xc)).item()),
            "papr_out_db": float(torch.mean(measure_papr_db(yi)).item()),
            "aclr_in_db": measure_aclr_db(xc, occupied_fraction=occupied_fraction),
            "aclr_out_db": measure_aclr_db(yi, occupied_fraction=occupied_fraction),
            "iq_irr_db": float(min(irr_db, 120.0)),
        }


__all__ = [
    "RFImpairmentChain",
    "RFImpairmentConfig",
    "apply_adc_quantization",
    "apply_cfo",
    "apply_iq_imbalance",
    "apply_phase_noise",
    "apply_rapp_pa",
    "measure_aclr_db",
    "measure_papr_db",
]
