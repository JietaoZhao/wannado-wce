"""STAR-RIS Simultaneous Transmitting and Reflecting Channel (Xu, Liu, Mu & Dobre, IEEE CL 2021).

[Lineage & Intuition]
- Family / Era / Tier : arrays_and_ris / frontier / A
- Parent              : `passive_ris_cascade`
- Key Diff            : Enables 360-deg full-space coverage via simultaneous transmission (T) and reflection (R)
                        under strict energy-splitting conservation |beta_m^T|^2 + |beta_m^R|^2 <= 1.
- Extension Hook [AI-Prompt Hook]:
    1. Add time-switching (TS) or mode-switching (MS) dynamic scheduling across slots.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Any, Literal

import torch

from wannado.wce.core.provenance import MECHANISM_PROVENANCE
from wannado.wce.core.types import ParamProvenance, ProvisionResult, WCEMeta
from wannado.wce.core.utils import complex_normal, resolve_device, set_global_seed

META = WCEMeta(
    id="star_ris",
    name="STAR-RIS Full-Space Transmission & Reflection Channel",
    family="arrays_and_ris",
    era="frontier",
    tier="A",
    output_kind="complex_channel_tensor",
    supports_torch_complex=True,
    supports_autograd=True,
    standard_ids=("etsi_gr_ris_003",),
    parent_id="passive_ris_cascade",
    diff_from_parent="Supports Energy Splitting (ES) & Mode Switching (MS) serving reflection + transmission half-spaces simultaneously.",
    summary_plain="360-degree full-space STAR-RIS channel serving both front reflection users and back transmission users.",
    when_to_use="Indoor-to-outdoor windows, full-space 360-deg coverage, and NOMA/STAR-RIS beamforming.",
    when_not_to_use="Half-space reflection-only metasurfaces (`passive_ris_cascade`).",
    hyperparam_intuition={
        "protocol": "ES (continuous energy split per element) vs MS (each element purely reflects or transmits).",
        "reflection_energy_fraction": "Fraction of incident power allocated to the reflection half-space.",
    },
    extension_hooks=("Optimize coupled T/R phase shifts for NOMA/ISAC",),
    module_path="wannado.wce.blocks.arrays_and_ris.star_ris_xu2021",
    default_mechanisms=("star_ris_energy_split",),
    citation=MECHANISM_PROVENANCE["star_ris_energy_split"].citation,
)


@dataclass
class STARRISConfig:
    batch_size: int = 128
    num_bs_ant: int = 4
    num_star_elements: int = 48
    protocol: Literal["ES", "MS"] = "ES"
    reflection_energy_fraction: float = 0.5
    device: str = "auto"
    seed: int = 42


def provision_star_ris(config: STARRISConfig | None = None, **overrides: Any) -> ProvisionResult:
    """Provision STAR-RIS reflection & transmission half-space effective channels enforcing energy conservation."""
    cfg = config or STARRISConfig()
    if overrides:
        cfg = STARRISConfig(**{**asdict(cfg), **overrides})
    t0 = time.perf_counter()
    set_global_seed(cfg.seed)
    dev = resolve_device(cfg.device)

    eta_r = float(min(1.0, max(0.0, cfg.reflection_energy_fraction)))
    if cfg.protocol == "ES":
        amp_r = torch.full((cfg.batch_size, cfg.num_star_elements), math.sqrt(eta_r), device=dev)
        amp_t = torch.full((cfg.batch_size, cfg.num_star_elements), math.sqrt(1.0 - eta_r), device=dev)
    else:
        mask_r = (torch.rand((cfg.batch_size, cfg.num_star_elements), device=dev) < eta_r).to(torch.float32)
        amp_r = mask_r
        amp_t = 1.0 - mask_r

    theta_r = torch.rand((cfg.batch_size, cfg.num_star_elements), device=dev) * (2.0 * math.pi)
    theta_t = torch.rand((cfg.batch_size, cfg.num_star_elements), device=dev) * (2.0 * math.pi)
    phi_r = amp_r * torch.complex(torch.cos(theta_r), torch.sin(theta_r))
    phi_t = amp_t * torch.complex(torch.cos(theta_t), torch.sin(theta_t))

    energy_sum = (phi_r.abs().pow(2) + phi_t.abs().pow(2)).max().item()
    g_bs = complex_normal((cfg.batch_size, cfg.num_star_elements, cfg.num_bs_ant), device=dev)
    h_r = complex_normal((cfg.batch_size, cfg.num_star_elements), device=dev)
    h_t = complex_normal((cfg.batch_size, cfg.num_star_elements), device=dev)

    h_eff_r = torch.einsum("bm,bm,bmn->bn", h_r.conj(), phi_r, g_bs).to(torch.complex64)
    h_eff_t = torch.einsum("bm,bm,bmn->bn", h_t.conj(), phi_t, g_bs).to(torch.complex64)

    prov = [
        ParamProvenance("protocol", cfg.protocol, "enum", "STAR-RIS operating protocol (ES vs MS)", "star_ris_energy_split", "IEEE CL 2021", "xu2021starriss", "Enforces |beta_R|^2 + |beta_T|^2 <= 1 per element."),
    ]
    return ProvisionResult(
        block_id=META.id,
        seed=cfg.seed,
        elapsed_seconds=time.perf_counter() - t0,
        config=asdict(cfg),
        tensors={"h_eff_reflect": h_eff_r, "h_eff_transmit": h_eff_t, "phi_r": phi_r, "phi_t": phi_t},
        metrics={
            "reflect_channel_gain": float(torch.mean(h_eff_r.abs().pow(2).sum(dim=-1)).item()),
            "transmit_channel_gain": float(torch.mean(h_eff_t.abs().pow(2).sum(dim=-1)).item()),
            "max_element_energy_sum": float(energy_sum),
        },
        diagnostics={"device": str(dev), "is_passive_ris": True, "max_ris_modulus": float(math.sqrt(energy_sum))},
        parameter_provenance=prov,
        mechanisms_used=list(META.default_mechanisms),
        standard_ids_used=list(META.standard_ids),
    )
