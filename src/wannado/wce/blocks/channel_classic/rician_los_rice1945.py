"""Rician LoS + Scattered NLoS Fading Channel (Rice, BSTJ 1945).

[Lineage & Intuition]
- Family / Era / Tier : channel_classic / classic / S
- Parent              : `rayleigh_clarke`
- Key Diff            : Superposes a deterministic phase-coherent specular LoS ray with scattered
                        Clarke NLoS multipath controlled by Ricean K-factor, preserving unit power.
- Extension Hook [AI-Prompt Hook]:
    1. Make K-factor a function of satellite elevation angle -> `ntn_leo_satellite`.
    2. Combine with 3D spatial elevation LoS probability -> `uav_air_to_ground`.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Any

import torch

from wannado.wce.core.provenance import MECHANISM_PROVENANCE
from wannado.wce.core.types import LitCitation, ParamProvenance, ProvisionResult, WCEMeta
from wannado.wce.core.units import db_to_linear, doppler_max_hz
from wannado.wce.core.utils import complex_normal, resolve_device, set_global_seed

_RICE_CIT = LitCitation(
    key="rice1945mathematical",
    title="Mathematical Analysis of Random Noise",
    authors="S. O. Rice",
    venue="The Bell System Technical Journal",
    year=1945,
    url="https://ieeexplore.ieee.org/document/6772729",
    bibtex=(
        "@article{rice1945mathematical,\n"
        "  title={Mathematical Analysis of Random Noise},\n"
        "  author={Rice, Stephen O.},\n"
        "  journal={The Bell System Technical Journal},\n"
        "  volume={24},\n"
        "  number={1},\n"
        "  pages={46--156},\n"
        "  year={1945},\n"
        "  url={https://ieeexplore.ieee.org/document/6772729}\n"
        "}"
    ),
    ieee='S. O. Rice, "Mathematical analysis of random noise," Bell Syst. Tech. J., vol. 24, no. 1, pp. 46-156, Jan. 1945.',
    apa="Rice, S. O. (1945). Mathematical analysis of random noise. The Bell System Technical Journal, 24(1), 46-156.",
)

META = WCEMeta(
    id="rician_los",
    name="Rician LoS + NLoS Baseband Fading Channel",
    family="channel_classic",
    era="classic",
    tier="S",
    output_kind="complex_channel_tensor",
    supports_torch_complex=True,
    supports_autograd=True,
    standard_ids=("gpp_tr38901",),
    parent_id="rayleigh_clarke",
    diff_from_parent="Adds deterministic LoS component weighted by sqrt(K/(K+1)) to scattered Rayleigh component.",
    summary_plain="Unit-power LoS + multipath Rician fading parameterized by K-factor (dB) with optional Doppler rotation.",
    when_to_use="UAV, satellite, RIS-direct, or indoor LoS links where a dominant specular ray exists.",
    when_not_to_use="Purely obstructed urban NLoS (`rayleigh_clarke`) or site-specific geometry (`tr38901_uma_umi_rma`).",
    hyperparam_intuition={
        "k_factor_db": "Ratio of LoS power to scattered power in dB (0 dB = equal split, 10 dB = strong LoS, -inf = Rayleigh).",
        "aoa_los_rad": "Angle of arrival of the direct LoS wave affecting Doppler shift f_d * cos(aoa).",
    },
    extension_hooks=(
        "Coupled elevation-dependent K-factor -> `ntn_leo_satellite`",
        "Extend to spatial multi-cluster mmWave -> `saleh_valenzuela`",
    ),
    module_path="wannado.wce.blocks.channel_classic.rician_los_rice1945",
    default_mechanisms=("shannon_awgn", "clarke_jakes_doppler"),
    citation=_RICE_CIT,
)


@dataclass
class RicianConfig:
    batch_size: int = 256
    num_steps: int = 32
    num_antennas: int = 4
    k_factor_db: float = 10.0
    fc_ghz: float = 3.5
    speed_mps: float = 5.0
    slot_duration_ms: float = 0.5
    aoa_los_rad: float = 0.4
    device: str = "auto"
    seed: int = 42


def provision_rician_los(config: RicianConfig | None = None, **overrides: Any) -> ProvisionResult:
    """Provision unit-average-power Rician fading H[B, T, A] = sqrt(K/(K+1)) H_LoS + sqrt(1/(K+1)) H_NLoS."""
    cfg = config or RicianConfig()
    if overrides:
        cfg = RicianConfig(**{**asdict(cfg), **overrides})
    t0 = time.perf_counter()
    set_global_seed(cfg.seed)
    dev = resolve_device(cfg.device)

    k_lin = float(db_to_linear(cfg.k_factor_db))
    w_los = math.sqrt(k_lin / (k_lin + 1.0))
    w_nlos = math.sqrt(1.0 / (k_lin + 1.0))

    fd = doppler_max_hz(cfg.speed_mps, cfg.fc_ghz * 1e9)
    dt = cfg.slot_duration_ms * 1e-3
    t_grid = torch.arange(cfg.num_steps, device=dev, dtype=torch.float32) * dt

    # Specular LoS ULA phase progression across antenna index and Doppler across time
    ant_idx = torch.arange(cfg.num_antennas, device=dev, dtype=torch.float32)
    los_phase = (
        2.0 * math.pi * fd * math.cos(cfg.aoa_los_rad) * t_grid.view(1, cfg.num_steps, 1)
        - math.pi * math.sin(cfg.aoa_los_rad) * ant_idx.view(1, 1, cfg.num_antennas)
    )
    h_los = torch.complex(torch.cos(los_phase), torch.sin(los_phase)).expand(cfg.batch_size, -1, -1)

    # Temporally correlated AR(1) complex Gaussian NLoS component normalized to CN(0, 1)
    rho = max(0.0, min(0.999, float(torch.special.bessel_j0(torch.tensor(2.0 * math.pi * fd * dt)).item())))
    innov_std = math.sqrt(max(1e-8, 1.0 - rho * rho))
    h_nlos_list = [complex_normal((cfg.batch_size, cfg.num_antennas), variance=1.0, device=dev)]
    for _ in range(1, cfg.num_steps):
        w = complex_normal((cfg.batch_size, cfg.num_antennas), variance=1.0, device=dev)
        h_nlos_list.append(rho * h_nlos_list[-1] + innov_std * w)
    h_nlos = torch.stack(h_nlos_list, dim=1)

    h = (w_los * h_los + w_nlos * h_nlos).to(torch.complex64)
    prov = [
        ParamProvenance("k_factor_db", cfg.k_factor_db, "dB", "Specular LoS-to-NLoS power ratio", "gpp_tr38901", "Clause 7.5", "3gpp_tr38901", "Controls fading depth via sqrt(K/(K+1)) LoS weight."),
        ParamProvenance("speed_mps", cfg.speed_mps, "m/s", "Relative velocity", "clarke_jakes_doppler", "BSTJ 1968", "clarke1968statistical", f"Drives LoS phase rotation and NLoS AR(1) correlation rho={rho:.4f}."),
    ]
    return ProvisionResult(
        block_id=META.id,
        seed=cfg.seed,
        elapsed_seconds=time.perf_counter() - t0,
        config=asdict(cfg),
        tensors={"h": h, "h_los": h_los, "h_nlos": h_nlos},
        metrics={
            "k_factor_linear": k_lin,
            "los_power_fraction": w_los ** 2,
            "mean_channel_power": float(torch.mean(h.abs().pow(2)).item()),
            "nlos_temporal_rho": rho,
        },
        diagnostics={"device": str(dev), "normalized_channel_power": 1.0},
        parameter_provenance=prov,
        mechanisms_used=list(META.default_mechanisms),
        standard_ids_used=list(META.standard_ids),
    )
