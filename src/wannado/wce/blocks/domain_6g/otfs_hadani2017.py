"""OTFS Delay-Doppler Channel & High-Mobility ICI Comparison (Hadani et al., IEEE WCNC 2017).

[Lineage & Intuition]
- Family / Era / Tier : domain_6g / frontier / S (IMT-2030 High-Mobility 500-1000 km/h Pillar)
- Parent              : `rayleigh_clarke`
- Key Diff            : Transforms fast time-varying doubly-dispersive multipath channels into
                        sparse, quasi-static 2D Delay-Doppler representations via ISFFT/SFFT.
- Extension Hook [AI-Prompt Hook]:
    1. Pair with LEO orbital Doppler (`ntn_leo_satellite`) or HSR/V2X (`v2x_vehicular_sl`).
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Any

import torch

from wannado.wce.core.provenance import MECHANISM_PROVENANCE
from wannado.wce.core.types import ParamProvenance, ProvisionResult, WCEMeta
from wannado.wce.core.units import doppler_max_hz
from wannado.wce.core.utils import complex_normal, resolve_device, set_global_seed

META = WCEMeta(
    id="otfs_delay_doppler",
    name="OTFS 2D Delay-Doppler High-Mobility Channel (Hadani et al., 2017)",
    family="domain_6g",
    era="frontier",
    tier="S",
    output_kind="complex_channel_tensor",
    supports_torch_complex=True,
    supports_autograd=True,
    standard_ids=("itur_m2160", "gpp_tr38821"),
    parent_id="rayleigh_clarke",
    diff_from_parent="Represents high-Doppler multipath channels in 2D Delay-Doppler domain via SFFT/ISFFT.",
    summary_plain="High-mobility 2D Delay-Doppler OTFS channel capturing diversity and eliminating OFDM inter-carrier interference.",
    when_to_use="High-speed trains (500-1000 km/h), aviation, LEO NTN, and doubly-dispersive waveforms.",
    when_not_to_use="Low-mobility pedestrian channels where standard OFDM suffices (`tr38901_cdl_tdl`).",
    hyperparam_intuition={
        "m_delay_bins": "Number of delay grid bins (subcarriers M).",
        "n_doppler_bins": "Number of Doppler grid bins (time slots N).",
        "speed_kmh": "Mobility speed driving maximum Doppler relative to subcarrier spacing.",
    },
    extension_hooks=("Plug custom MP/MPA Delay-Doppler equalizer on top of tensors['h_dd']",),
    module_path="wannado.wce.blocks.domain_6g.otfs_hadani2017",
    default_mechanisms=("hadani_otfs_delay_doppler", "clarke_jakes_doppler"),
    citation=MECHANISM_PROVENANCE["hadani_otfs_delay_doppler"].citation,
)


@dataclass
class OTFSConfig:
    batch_size: int = 32
    m_delay_bins: int = 32
    n_doppler_bins: int = 16
    num_paths: int = 4
    fc_ghz: float = 4.0
    subcarrier_spacing_khz: float = 15.0
    speed_kmh: float = 500.0
    device: str = "auto"
    seed: int = 42


def provision_otfs_delay_doppler(config: OTFSConfig | None = None, **overrides: Any) -> ProvisionResult:
    """Construct sparse 2D Delay-Doppler channel & compare OFDM ICI energy leakage vs OTFS spread."""
    cfg = config or OTFSConfig()
    if overrides:
        cfg = OTFSConfig(**{**asdict(cfg), **overrides})
    t0 = time.perf_counter()
    set_global_seed(cfg.seed)
    dev = resolve_device(cfg.device)

    speed_mps = cfg.speed_kmh / 3.6
    fd_max = doppler_max_hz(speed_mps, cfg.fc_ghz * 1e9)
    scs_hz = cfg.subcarrier_spacing_khz * 1e3
    norm_doppler = fd_max / max(scs_hz, 1.0)

    # Construct 2D Delay-Doppler channel impulse response H_dd[B, M, N]
    h_dd = torch.zeros((cfg.batch_size, cfg.m_delay_bins, cfg.n_doppler_bins), dtype=torch.complex64, device=dev)
    delay_taps = torch.randint(0, max(1, cfg.m_delay_bins // 4), (cfg.batch_size, cfg.num_paths), device=dev)
    doppler_shifts = torch.randint(0, cfg.n_doppler_bins, (cfg.batch_size, cfg.num_paths), device=dev)
    gains = complex_normal((cfg.batch_size, cfg.num_paths), variance=1.0 / max(1, cfg.num_paths), device=dev)

    b_idx = torch.arange(cfg.batch_size, device=dev).unsqueeze(-1)
    h_dd[b_idx, delay_taps, doppler_shifts] += gains

    # ISFFT to Time-Frequency grid H_tf = FFT_M(IFFT_N(H_dd)) (scaled to preserve energy)
    h_tf = torch.fft.fft(torch.fft.ifft(h_dd, dim=-1, norm="ortho"), dim=-2, norm="ortho")

    # Estimate normalized OFDM ICI floor ~ 1 - sinc^2(fd_max / scs) vs OTFS sparse tap fraction
    x_ici = min(norm_doppler, 0.9)
    sinc_val = math.sin(math.pi * x_ici) / (math.pi * x_ici) if x_ici > 1e-6 else 1.0
    ofdm_ici_leakage = max(0.0, 1.0 - sinc_val ** 2)
    dd_sparsity = float((h_dd.abs() > 1e-8).float().mean().item())

    prov = [
        ParamProvenance("speed_kmh", cfg.speed_kmh, "km/h", "Vehicle/HSR/Aircraft speed", "itur_m2160", "Capability #6", "itur_m2160_2023", "IMT-2030 targets 500-1000 km/h extreme mobility."),
        ParamProvenance("m_delay_bins", cfg.m_delay_bins, "bins", "Delay-Doppler grid dimension M x N", "hadani_otfs_delay_doppler", "IEEE WCNC 2017", "hadani2017orthogonal", "Converts time-varying channels into compact sparse DD taps."),
    ]
    return ProvisionResult(
        block_id=META.id,
        seed=cfg.seed,
        elapsed_seconds=time.perf_counter() - t0,
        config=asdict(cfg),
        tensors={"h_dd": h_dd, "h_tf": h_tf},
        metrics={
            "max_doppler_hz": fd_max,
            "normalized_doppler_fd_over_scs": norm_doppler,
            "estimated_ofdm_ici_power_leakage": ofdm_ici_leakage,
            "dd_channel_nonzero_density": dd_sparsity,
            "mean_dd_energy": float(torch.mean(h_dd.abs().pow(2).sum(dim=(-2, -1))).item()),
        },
        diagnostics={"device": str(dev), "normalized_channel_power": 1.0},
        parameter_provenance=prov,
        mechanisms_used=list(META.default_mechanisms),
        standard_ids_used=list(META.standard_ids),
    )
