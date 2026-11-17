"""Clarke/Jakes Sum-of-Sinusoids Rayleigh Fading Channel (Clarke, BSTJ 1968; Jakes, 1974).

[Lineage & Intuition]
- Family / Era / Tier : channel_classic / classic / S
- Parent              : `awgn_link` (Shannon 1948)
- Key Diff            : Introduces small-scale multipath Rayleigh amplitude fading plus temporally
                        correlated Doppler evolution matching R(dt) = J0(2*pi*f_d*dt).
- Extension Hook [AI-Prompt Hook]:
    1. Add a deterministic LoS component -> `rician_los`.
    2. Transform high-Doppler time-frequency channel into 2D Delay-Doppler -> `otfs_delay_doppler`.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Any

import torch
from scipy import special

from wannado.wce.core.provenance import MECHANISM_PROVENANCE
from wannado.wce.core.types import ParamProvenance, ProvisionResult, WCEMeta
from wannado.wce.core.units import coherence_time_sec, doppler_max_hz
from wannado.wce.core.utils import resolve_device, set_global_seed

META = WCEMeta(
    id="rayleigh_clarke",
    name="Clarke/Jakes Correlated Rayleigh Fading Channel",
    family="channel_classic",
    era="classic",
    tier="S",
    output_kind="complex_channel_tensor",
    supports_torch_complex=True,
    supports_autograd=True,
    standard_ids=("gpp_ts38211",),
    parent_id="awgn_link",
    diff_from_parent="Replaces constant channel gain with zero-mean CN(0,1) multipath Rayleigh fading & Jakes Doppler autocorrelation.",
    summary_plain="Standard NLoS small-scale Rayleigh fading with physically consistent Doppler time correlation J0(2*pi*f_d*dt).",
    when_to_use="Rich-scattering NLoS links, mobile channel prediction benchmarks, and classical fading baselines.",
    when_not_to_use="Strong direct LoS links (`rician_los`), directional mmWave (`saleh_valenzuela`), or 3GPP spatial multi-cell (`tr38901_uma_umi_rma`).",
    hyperparam_intuition={
        "speed_mps": "Mobile terminal velocity; higher speed shrinks coherence time and speeds up fading.",
        "slot_duration_ms": "Time step between adjacent channel snapshots (e.g., 1 ms for mu=0 or 0.5 ms for mu=1).",
        "num_sinusoids": "Number of isotropic plane waves superposed in Clarke's sum-of-sinusoids (larger = smoother J0 match).",
    },
    extension_hooks=(
        "Inject specular LoS steering component -> `rician_los`",
        "Map time-varying taps into Delay-Doppler -> `otfs_delay_doppler`",
    ),
    module_path="wannado.wce.blocks.channel_classic.rayleigh_clarke1968",
    default_mechanisms=("shannon_awgn", "clarke_jakes_doppler"),
    citation=MECHANISM_PROVENANCE["clarke_jakes_doppler"].citation,
)


@dataclass
class RayleighClarkeConfig:
    batch_size: int = 128
    num_steps: int = 64
    num_antennas: int = 4
    fc_ghz: float = 3.5
    speed_mps: float = 15.0
    slot_duration_ms: float = 0.5
    num_sinusoids: int = 32
    device: str = "auto"
    seed: int = 42


def provision_rayleigh_clarke(config: RayleighClarkeConfig | None = None, **overrides: Any) -> ProvisionResult:
    """Generate unit-power temporally correlated Rayleigh MIMO-like fading H[B, T, A] via Clarke SOS."""
    cfg = config or RayleighClarkeConfig()
    if overrides:
        cfg = RayleighClarkeConfig(**{**asdict(cfg), **overrides})
    t0 = time.perf_counter()
    set_global_seed(cfg.seed)
    dev = resolve_device(cfg.device)

    fc_hz = cfg.fc_ghz * 1e9
    fd = doppler_max_hz(cfg.speed_mps, fc_hz)
    tc = coherence_time_sec(fd)
    dt = cfg.slot_duration_ms * 1e-3

    # Vectorized Clarke sum-of-sinusoids with independent I/Q phases for true CN(0, 1)
    # h_i(t) = (1 / sqrt(M)) sum_m exp(j * (2*pi*f_d*t*cos(alpha_m) + phi_m))
    shape_prefix = (cfg.batch_size, cfg.num_antennas, cfg.num_sinusoids)
    alpha = torch.rand(shape_prefix, device=dev) * (2.0 * math.pi)
    phi = torch.rand(shape_prefix, device=dev) * (2.0 * math.pi)
    t_grid = torch.arange(cfg.num_steps, device=dev, dtype=torch.float32) * dt  # (T,)

    phase = 2.0 * math.pi * fd * torch.cos(alpha).unsqueeze(-1) * t_grid + phi.unsqueeze(-1)
    # Shape: (B, A, T) -> transpose to (B, T, A)
    h = torch.complex(torch.cos(phase), torch.sin(phase)).sum(dim=2) / math.sqrt(cfg.num_sinusoids)
    h = h.permute(0, 2, 1).contiguous().to(torch.complex64)

    # Empirical lag-1 temporal autocorrelation vs theoretical J0(2*pi*f_d*dt)
    if cfg.num_steps >= 2:
        corr_num = torch.mean((h[:, 1:, :] * h[:, :-1, :].conj()).real).item()
        corr_den = max(torch.mean(h.abs().pow(2)).item(), 1e-12)
        emp_lag1 = float(corr_num / corr_den)
    else:
        emp_lag1 = 1.0
    theo_lag1 = float(special.j0(2.0 * math.pi * fd * dt))

    prov = [
        ParamProvenance("speed_mps", cfg.speed_mps, "m/s", "Terminal speed governing Doppler spread", "clarke_jakes_doppler", "BSTJ 1968", "clarke1968statistical", f"Yields f_d={fd:.2f} Hz and T_c={tc*1e3:.2f} ms."),
        ParamProvenance("slot_duration_ms", cfg.slot_duration_ms, "ms", "Snapshot temporal spacing", "gpp_ts38211", "Clause 4.2", "3gpp_ts38211", "Aligns temporal sampling with 5G NR slot durations (e.g., mu=1 -> 0.5 ms)."),
        ParamProvenance("num_sinusoids", cfg.num_sinusoids, "waves", "Isotropic scatterer waves", "clarke_jakes_doppler", "BSTJ 1968", "clarke1968statistical", "Matches Clarke J0(2*pi*f_d*dt) autocorrelation."),
    ]
    return ProvisionResult(
        block_id=META.id,
        seed=cfg.seed,
        elapsed_seconds=time.perf_counter() - t0,
        config=asdict(cfg),
        tensors={"h": h, "t_sec": t_grid},
        metrics={
            "max_doppler_hz": fd,
            "coherence_time_ms": tc * 1e3 if math.isfinite(tc) else float("inf"),
            "theoretical_lag1_autocorr": theo_lag1,
            "empirical_lag1_autocorr": emp_lag1,
            "autocorr_abs_error": abs(emp_lag1 - theo_lag1),
            "mean_channel_power": float(torch.mean(h.abs().pow(2)).item()),
        },
        diagnostics={"device": str(dev), "normalized_channel_power": 1.0, "lag1_autocorr_gap": abs(emp_lag1 - theo_lag1)},
        parameter_provenance=prov,
        mechanisms_used=list(META.default_mechanisms),
        standard_ids_used=list(META.standard_ids),
    )
