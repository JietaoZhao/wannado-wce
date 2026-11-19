"""Passive Reconfigurable Intelligent Surface (RIS) Cascaded Channel (ETSI GR RIS 003; Wu & Zhang 2019).

[Lineage & Intuition]
- Family / Era / Tier : arrays_and_ris / frontier / S
- Parent              : `farfield_upa`
- Key Diff            : Models direct BS->UE plus two-hop cascaded BS->RIS->UE propagation with
                        strict passive unit-modulus phase matrix |phi_m| = 1 (continuous or b-bit).
- Extension Hook [AI-Prompt Hook]:
    1. Split energy into simultaneous 360-deg reflection + transmission -> `star_ris`.
    2. Upgrade passive loads to active reflection amplifiers -> `active_ris_flexible`.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Any

import torch

from wannado.wce.core.provenance import MECHANISM_PROVENANCE
from wannado.wce.core.types import ParamProvenance, ProvisionResult, WCEMeta
from wannado.wce.core.units import db_to_linear
from wannado.wce.core.utils import complex_normal, resolve_device, set_global_seed

META = WCEMeta(
    id="passive_ris_cascade",
    name="Passive RIS Two-Hop Cascaded Channel (ETSI GR RIS 003 + Wu & Zhang 2019)",
    family="arrays_and_ris",
    era="frontier",
    tier="S",
    output_kind="complex_channel_tensor",
    supports_torch_complex=True,
    supports_autograd=True,
    standard_ids=("etsi_gr_ris_003", "gpp_tr38901"),
    parent_id="farfield_upa",
    diff_from_parent="Constructs H_eff = h_d^H + h_r^H * Phi * G enforcing strict passive |phi_m|=1 conservation.",
    summary_plain="Standardized passive RIS cascaded channel supporting continuous or b-bit discrete phase shifts and co-phasing gain benchmark.",
    when_to_use="Smart radio environments, blockage bypass, joint active-passive beamforming, and RIS DRL control.",
    when_not_to_use="Full-space transmission+reflection (`star_ris`) or amplified active metasurfaces (`active_ris_flexible`).",
    hyperparam_intuition={
        "num_ris_elements": "Meta-atom count M; passive RIS SNR scales asymptotically as M^2 under coherent alignment!",
        "phase_bits": "0 = continuous [0, 2pi); >0 = discrete b-bit hardware phase quantization (e.g., 2-bit -> 4 phases).",
    },
    extension_hooks=("Optimize Phi via DRL or alternating manifold optimization",),
    module_path="wannado.wce.blocks.arrays_and_ris.passive_ris_wu2019",
    default_mechanisms=("wu_zhang_passive_ris",),
    citation=MECHANISM_PROVENANCE["wu_zhang_passive_ris"].citation,
)


@dataclass
class PassiveRISConfig:
    batch_size: int = 128
    num_bs_ant: int = 8
    num_ris_elements: int = 64
    phase_bits: int = 0
    rician_k_ris_db: float = 10.0
    direct_link_attenuation_db: float = 15.0
    ris_cascade_gain_scale: float = 1.0
    device: str = "auto"
    seed: int = 42


def _quantize_phases(theta: torch.Tensor, phase_bits: int) -> torch.Tensor:
    if phase_bits <= 0:
        return theta
    levels = 2 ** phase_bits
    step = (2.0 * math.pi) / levels
    return torch.round(theta / step) * step


def provision_passive_ris_cascade(config: PassiveRISConfig | None = None, **overrides: Any) -> ProvisionResult:
    """Provision BS-RIS-UE cascade channels & compare optimal co-phased RIS vs random phase vs no-RIS."""
    cfg = config or PassiveRISConfig()
    if overrides:
        cfg = PassiveRISConfig(**{**asdict(cfg), **overrides})
    t0 = time.perf_counter()
    set_global_seed(cfg.seed)
    dev = resolve_device(cfg.device)

    k_lin = float(db_to_linear(cfg.rician_k_ris_db))
    w_los = math.sqrt(k_lin / (k_lin + 1.0))
    w_nlos = math.sqrt(1.0 / (k_lin + 1.0))

    # G: (B, M, Nt) BS-to-RIS; h_r: (B, M) RIS-to-UE; h_d: (B, Nt) attenuated direct link
    g_mat = w_los * torch.ones((cfg.batch_size, cfg.num_ris_elements, cfg.num_bs_ant), dtype=torch.complex64, device=dev) + w_nlos * complex_normal((cfg.batch_size, cfg.num_ris_elements, cfg.num_bs_ant), device=dev)
    h_r = w_los * torch.ones((cfg.batch_size, cfg.num_ris_elements), dtype=torch.complex64, device=dev) + w_nlos * complex_normal((cfg.batch_size, cfg.num_ris_elements), device=dev)
    dir_scale = math.sqrt(float(db_to_linear(-cfg.direct_link_attenuation_db)))
    h_d = dir_scale * complex_normal((cfg.batch_size, cfg.num_bs_ant), device=dev)

    # Random passive phase baseline: |phi_m| == 1
    theta_rand = _quantize_phases(torch.rand((cfg.batch_size, cfg.num_ris_elements), device=dev) * (2.0 * math.pi), cfg.phase_bits)
    phi_rand = torch.complex(torch.cos(theta_rand), torch.sin(theta_rand)).to(torch.complex64)

    # Optimal SISO-equivalent / MRT co-phasing aligned to dominant BS beamforming vector w = h_d / ||h_d|| or leading link
    # For single-stream MRT direction u_bs = G[0]^H h_r^* or direct-link phase alignment:
    cascaded_prod = h_r.conj().unsqueeze(-1) * g_mat * cfg.ris_cascade_gain_scale  # (B, M, Nt)
    # Maximize |<h_d + sum_m phi_m * cascaded_prod_m, w_bs>| via conjugate inner product with w_bs = h_d / ||h_d||
    w_bs = h_d / torch.linalg.norm(h_d, dim=-1, keepdim=True).clamp(min=1e-8)
    proj = torch.einsum("bmn,bn->bm", cascaded_prod, w_bs.conj())
    ref_phase = torch.angle(torch.sum(h_d * w_bs.conj(), dim=-1, keepdim=True))
    theta_opt = _quantize_phases(ref_phase - torch.angle(proj), cfg.phase_bits)
    phi_opt = torch.complex(torch.cos(theta_opt), torch.sin(theta_opt)).to(torch.complex64)

    h_eff_rand = h_d + torch.einsum("bm,bmn->bn", phi_rand, cascaded_prod)
    h_eff_opt = h_d + torch.einsum("bm,bmn->bn", phi_opt, cascaded_prod)

    pow_no_ris = float(torch.mean(h_d.abs().pow(2).sum(dim=-1)).item())
    pow_rand_ris = float(torch.mean(h_eff_rand.abs().pow(2).sum(dim=-1)).item())
    pow_opt_ris = float(torch.mean(h_eff_opt.abs().pow(2).sum(dim=-1)).item())
    max_phi_mag = float(max(phi_opt.abs().max().item(), phi_rand.abs().max().item()))

    prov = [
        ParamProvenance("num_ris_elements", cfg.num_ris_elements, "elements", "Passive reflecting meta-atoms M", "wu_zhang_passive_ris", "IEEE TWC 2019", "wu2019intelligent", "Coherent alignment scales array gain quadratically with M."),
        ParamProvenance("phase_bits", cfg.phase_bits, "bits", "Phase shifter quantization resolution", "etsi_gr_ris_003", "Clause 5", "etsi_gr_ris_003", "0 = continuous; b>0 quantizes to 2^b unit-modulus phase levels."),
    ]
    return ProvisionResult(
        block_id=META.id,
        seed=cfg.seed,
        elapsed_seconds=time.perf_counter() - t0,
        config=asdict(cfg),
        tensors={"h_direct": h_d, "g_bs_ris": g_mat, "h_ris_ue": h_r, "phi_opt": phi_opt, "h_eff_opt": h_eff_opt},
        metrics={
            "gain_no_ris": pow_no_ris,
            "gain_random_ris": pow_rand_ris,
            "gain_aligned_ris": pow_opt_ris,
            "ris_gain_ratio_vs_random": pow_opt_ris / max(pow_rand_ris, 1e-12),
            "max_ris_element_modulus": max_phi_mag,
        },
        diagnostics={"device": str(dev), "is_passive_ris": True, "max_ris_modulus": max_phi_mag},
        parameter_provenance=prov,
        mechanisms_used=list(META.default_mechanisms),
        standard_ids_used=list(META.standard_ids),
    )
