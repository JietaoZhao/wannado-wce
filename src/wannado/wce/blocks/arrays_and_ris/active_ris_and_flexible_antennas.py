"""Active RIS, Fluid Antenna (FAS), & Pinching-Antenna (PASS) Flexible Physical Channels.

[Lineage & Intuition]
- Family / Era / Tier : arrays_and_ris / frontier / S (2023-2025 Beyond-Passive Physical Architectures)
- Parent              : `passive_ris_cascade`
- Key Diff            : Unifies Active RIS (amplification + dynamic amplifier noise G*Phi*n_v, Zhang 2023),
                        Fluid Antenna Spatial Port Diversity (FAS, Wong 2021), and Pinching-Antenna
                        Waveguide LoS Reconfiguration (PASS, Ding/Schober/Poor 2025).
- Extension Hook [AI-Prompt Hook]:
    1. Optimize active RIS gain matrix under finite DC/RF power constraint P_A.
    2. Optimize dielectric pinch coordinates along waveguide for multi-user NOMA.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
import torch
from scipy import special

from wannado.wce.core.provenance import MECHANISM_PROVENANCE
from wannado.wce.core.types import ParamProvenance, ProvisionResult, WCEMeta
from wannado.wce.core.units import db_to_linear, wavelength_m
from wannado.wce.core.utils import complex_normal, resolve_device, set_global_seed

META = WCEMeta(
    id="active_ris_flexible",
    name="Active RIS, Fluid Antenna (FAS) & Pinching-Antenna (PASS) Channels",
    family="arrays_and_ris",
    era="frontier",
    tier="S",
    output_kind="complex_channel_tensor",
    supports_torch_complex=True,
    supports_autograd=True,
    standard_ids=("etsi_gr_ris_003", "itur_m2160"),
    parent_id="passive_ris_cascade",
    diff_from_parent="Adds active reflection amplification + dynamic thermal noise, FAS spatial port correlation, and PASS waveguide LoS.",
    summary_plain="2023-2025 beyond-passive antenna & metasurface models: Active RIS, Fluid Antenna (FAS), and Pinching Antenna (PASS).",
    when_to_use="Overcoming RIS double-fading (Active RIS), compact port diversity (FAS), or dynamic waveguide LoS pinching (PASS).",
    when_not_to_use="Pure passive RIS (`passive_ris_cascade`) or fixed far-field massive MIMO (`farfield_upa`).",
    hyperparam_intuition={
        "architecture": "active_ris | fas (fluid antenna) | pass (pinching antenna).",
        "active_amp_gain_db": "Active RIS reflection amplification per element (must include amplified noise n_v!).",
        "fas_num_ports": "Number of densely packed switchable fluid antenna ports over length W*lambda.",
        "pass_waveguide_len_m": "Dielectric waveguide span along which pinches dynamically shorten LoS paths.",
    },
    extension_hooks=("Plug into joint DRL position/port selection via WirelessGymEnvAdapter",),
    module_path="wannado.wce.blocks.arrays_and_ris.active_ris_and_flexible_antennas",
    default_mechanisms=("active_ris_noise", "wong_fluid_antenna", "ding_pinching_antenna"),
    citation=MECHANISM_PROVENANCE["ding_pinching_antenna"].citation,
)


@dataclass
class ActiveRISFlexibleConfig:
    architecture: Literal["active_ris", "fas", "pass"] = "active_ris"
    batch_size: int = 128
    # Active RIS knobs
    num_ris_elements: int = 32
    active_amp_gain_db: float = 10.0
    ris_thermal_noise_ratio: float = 1.0
    # Fluid Antenna (FAS) knobs
    fas_num_ports: int = 16
    fas_aperture_wavelengths: float = 1.0
    # Pinching Antenna (PASS) knobs
    pass_waveguide_len_m: float = 20.0
    pass_height_m: float = 3.0
    fc_ghz: float = 28.0
    device: str = "auto"
    seed: int = 42


def provision_active_ris_flexible(config: ActiveRISFlexibleConfig | None = None, **overrides: Any) -> ProvisionResult:
    """Provision Active RIS (with amplified noise), FAS port channels, or PASS waveguide-pinched LoS channels."""
    cfg = config or ActiveRISFlexibleConfig()
    if overrides:
        cfg = ActiveRISFlexibleConfig(**{**asdict(cfg), **overrides})
    t0 = time.perf_counter()
    set_global_seed(cfg.seed)
    dev = resolve_device(cfg.device)

    if cfg.architecture == "active_ris":
        # Zhang et al. (IEEE TCOM 2023) double-fading two-hop SNR balance:
        # Both signal & passive link undergo hop1 * hop2 attenuation, whereas Active RIS dynamic thermal noise
        # only traverses hop2 -> overcome double-fading loss (beta_1 * beta_2) until amplified noise dominates.
        amp_lin = math.sqrt(float(db_to_linear(cfg.active_amp_gain_db)))
        hop1_snr_unit = 10.0  # +10 dB BS-to-RIS incident power per element over thermal floor
        hop2_loss = 1e-3      # -30 dB RIS-to-UE second-hop pathloss scale
        g_in = complex_normal((cfg.batch_size, cfg.num_ris_elements), device=dev)
        h_out = complex_normal((cfg.batch_size, cfg.num_ris_elements), device=dev)
        theta = -torch.angle(g_in * h_out)
        phi_active = amp_lin * torch.complex(torch.cos(theta), torch.sin(theta))
        cascade_vec = torch.sum(h_out * torch.complex(torch.cos(theta), torch.sin(theta)) * g_in, dim=-1)
        sig_active = amp_lin * math.sqrt(hop1_snr_unit * hop2_loss) * cascade_vec
        sig_passive = math.sqrt(hop1_snr_unit * hop2_loss) * cascade_vec
        # Dynamic noise amplified by Active RIS traverses only hop2 (factor sqrt(hop2_loss))
        n_v = complex_normal((cfg.batch_size, cfg.num_ris_elements), variance=cfg.ris_thermal_noise_ratio, device=dev)
        noise_dyn = math.sqrt(hop2_loss) * torch.sum(h_out * phi_active * n_v, dim=-1)
        snr_active = sig_active.abs().pow(2) / (1.0 + noise_dyn.abs().pow(2))
        snr_passive = sig_passive.abs().pow(2)
        metrics = {
            "active_ris_effective_snr": float(torch.mean(snr_active).item()),
            "passive_ris_equivalent_snr": float(torch.mean(snr_passive).item()),
            "dynamic_noise_power_ratio": float(torch.mean(noise_dyn.abs().pow(2)).item()),
            "active_gain_over_passive": float((torch.mean(snr_active) / torch.mean(snr_passive).clamp(min=1e-12)).item()),
        }
        tensors = {"h_eff": sig_active, "dynamic_noise": noise_dyn, "phi_active": phi_active}
        diag = {"device": str(dev), "is_passive_ris": False, "max_ris_modulus": amp_lin}
        prov = [ParamProvenance("active_amp_gain_db", cfg.active_amp_gain_db, "dB", "Active RIS element reflection amplitude gain", "active_ris_noise", "IEEE TCOM 2023", "zhang2023active", "Includes hop-2 amplified dynamic thermal noise sqrt(beta_2)*G*Phi*n_v.")]

    elif cfg.architecture == "fas":
        # Jakes spatial correlation across K tightly spaced fluid ports over W*lambda (Wong et al., 2021)
        pos = torch.linspace(0.0, cfg.fas_aperture_wavelengths, cfg.fas_num_ports, dtype=torch.float64)
        dist_lam = torch.abs(pos.unsqueeze(0) - pos.unsqueeze(1)).numpy()
        cov_np = special.j0(2.0 * math.pi * dist_lam) + 1e-6 * np.eye(cfg.fas_num_ports)
        l_chol = torch.from_numpy(np.linalg.cholesky(cov_np)).to(device=dev, dtype=torch.complex64)
        z = complex_normal((cfg.batch_size, cfg.fas_num_ports), device=dev)
        h_ports = torch.einsum("ij,bj->bi", l_chol, z)
        best_port_gain = torch.max(h_ports.abs().pow(2), dim=-1).values
        single_port_gain = h_ports[:, 0].abs().pow(2)
        metrics = {
            "fas_best_port_gain": float(torch.mean(best_port_gain).item()),
            "fixed_single_port_gain": float(torch.mean(single_port_gain).item()),
            "fas_spatial_diversity_gain": float((torch.mean(best_port_gain) / torch.mean(single_port_gain).clamp(min=1e-12)).item()),
        }
        tensors = {"h_ports": h_ports}
        diag = {"device": str(dev), "normalized_channel_power": 1.0}
        prov = [ParamProvenance("fas_num_ports", cfg.fas_num_ports, "ports", "Fluid antenna port resolution", "wong_fluid_antenna", "IEEE TWC 2021", "wong2021fluid", "J0(2*pi*d/lambda) spatial port correlation.")]

    else:
        # Pinching-Antenna (PASS): compare fixed BS at origin vs optimal waveguide pinch directly above user x_u
        lam = wavelength_m(cfg.fc_ghz * 1e9)
        x_u = (torch.rand(cfg.batch_size, device=dev) - 0.5) * cfg.pass_waveguide_len_m
        y_u = torch.rand(cfg.batch_size, device=dev) * 4.0 + 1.0
        d_fixed = torch.sqrt(x_u.pow(2) + y_u.pow(2) + cfg.pass_height_m ** 2)
        d_pinched = torch.sqrt(y_u.pow(2) + cfg.pass_height_m ** 2)  # pinch moves to x_pinch = x_u
        gain_fixed = (lam / (4.0 * math.pi * d_fixed)).pow(2)
        gain_pinched = (lam / (4.0 * math.pi * d_pinched)).pow(2)
        metrics = {
            "pass_pinched_path_gain_db": float(10.0 * torch.log10(torch.mean(gain_pinched)).item()),
            "fixed_antenna_path_gain_db": float(10.0 * torch.log10(torch.mean(gain_fixed)).item()),
            "pass_los_gain_boost_linear": float((torch.mean(gain_pinched) / torch.mean(gain_fixed).clamp(min=1e-18)).item()),
        }
        tensors = {"gain_pinched": gain_pinched, "gain_fixed": gain_fixed, "user_x_m": x_u}
        diag = {"device": str(dev), "normalized_channel_power": 1.0}
        prov = [ParamProvenance("pass_waveguide_len_m", cfg.pass_waveguide_len_m, "m", "Waveguide span for dielectric pinch activation", "ding_pinching_antenna", "arXiv 2412.02376", "ding2025flexible", "Minimizes large-scale LoS distance via waveguide pinching.")]

    return ProvisionResult(
        block_id=META.id,
        seed=cfg.seed,
        elapsed_seconds=time.perf_counter() - t0,
        config=asdict(cfg),
        tensors=tensors,
        metrics=metrics,
        diagnostics=diag,
        parameter_provenance=prov,
        mechanisms_used=list(META.default_mechanisms),
        standard_ids_used=list(META.standard_ids),
    )
