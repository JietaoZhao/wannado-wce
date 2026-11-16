"""Near-Field XL-MIMO Spherical Wave & THz Molecular Absorption (Cui & Dai, IEEE TCOM 2022; ITU-R P.676).

[Lineage & Intuition]
- Family / Era / Tier : arrays_and_ris / frontier / S (6G XL-MIMO & Sub-THz Core Physics)
- Parent              : `farfield_upa`
- Key Diff            : Replaces far-field plane wave steering with exact spherical wavefronts (distance + angle)
                        inside the Rayleigh distance Z_R = 2*D^2/lambda, plus ITU-R P.676 THz gas absorption.
- Extension Hook [AI-Prompt Hook]:
    1. Quantize near-field focal beams into polar-domain dictionaries for compressive CSI estimation.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch

from wannado.wce.core.geometry import nearfield_spherical_steering, upa_element_coordinates, upa_farfield_steering
from wannado.wce.core.provenance import MECHANISM_PROVENANCE
from wannado.wce.core.types import ParamProvenance, ProvisionResult, WCEMeta
from wannado.wce.core.units import db_to_linear, effective_rayleigh_distance_m, rayleigh_distance_m, wavelength_m
from wannado.wce.core.utils import complex_normal, resolve_device, set_global_seed
from wannado.wce.standards.etsi_ieee import itur_p676_attenuation_db_per_km, itur_p676_path_absorption_db

META = WCEMeta(
    id="nearfield_xlmimo_thz",
    name="Near-Field XL-MIMO Spherical Wave & THz ITU-R P.676 Channel",
    family="arrays_and_ris",
    era="frontier",
    tier="S",
    output_kind="complex_channel_tensor",
    supports_torch_complex=True,
    supports_autograd=True,
    standard_ids=("itur_p676", "itur_m2160"),
    parent_id="farfield_upa",
    diff_from_parent="Computes exact spherical-wave near-field steering and ITU-R P.676 O2/H2O molecular absorption.",
    summary_plain="6G Extremely Large-Scale MIMO (XL-MIMO) near-field beam-focusing & sub-THz molecular absorption channel.",
    when_to_use="ELAA/XL-MIMO, sub-THz/THz links, beamfocusing vs far-field energy-spread diagnostics.",
    when_not_to_use="Small sub-6 GHz arrays operating well beyond the Rayleigh distance (`farfield_upa`).",
    hyperparam_intuition={
        "num_tx_x": "Aperture width elements; Rayleigh distance scales quadratically with aperture D^2!",
        "user_distance_m": "Tx-to-UE distance; when d < Z_R, planar wave assumptions break down.",
        "fc_ghz": "Carrier frequency (e.g., 100 GHz / 300 GHz THz); drives both Z_R and ITU-R P.676 absorption.",
    },
    extension_hooks=("Implement polar-domain dictionary OMP estimation per Cui & Dai (2022)",),
    module_path="wannado.wce.blocks.arrays_and_ris.nearfield_xlmimo_cui2022",
    default_mechanisms=("cui_dai_nearfield_spherical", "itur_p676_thz_absorption"),
    citation=MECHANISM_PROVENANCE["cui_dai_nearfield_spherical"].citation,
)


@dataclass
class NearFieldXLMIMOConfig:
    batch_size: int = 64
    num_tx_x: int = 128
    num_tx_y: int = 1
    num_paths: int = 4
    fc_ghz: float = 100.0
    user_distance_m: float = 15.0
    broadside_angle_deg: float = 15.0
    water_vapor_g_m3: float = 7.5
    spacing_lambda: float = 0.5
    device: str = "auto"
    seed: int = 42


def provision_nearfield_xlmimo_thz(config: NearFieldXLMIMOConfig | None = None, **overrides: Any) -> ProvisionResult:
    """Provision exact near-field spherical-wave channel & compare far-field planar beamfocusing mismatch."""
    cfg = config or NearFieldXLMIMOConfig()
    if overrides:
        cfg = NearFieldXLMIMOConfig(**{**asdict(cfg), **overrides})
    t0 = time.perf_counter()
    set_global_seed(cfg.seed)
    dev = resolve_device(cfg.device)

    fc_hz = cfg.fc_ghz * 1e9
    lam = wavelength_m(fc_hz)
    d_elem = cfg.spacing_lambda * lam
    aperture_m = math.hypot((cfg.num_tx_x - 1) * d_elem, (cfg.num_tx_y - 1) * d_elem)
    zr_m = rayleigh_distance_m(aperture_m, fc_hz)
    theta_rad = math.radians(cfg.broadside_angle_deg)
    zr_eff_m = effective_rayleigh_distance_m(aperture_m, fc_hz, theta_rad)

    elem_coords = upa_element_coordinates(cfg.num_tx_x, cfg.num_tx_y, fc_hz, spacing_lambda=cfg.spacing_lambda).to(dev)

    # Scatterer / target 3D locations around user_distance_m (x-z array facing +y broadside)
    r_scatter = torch.rand((cfg.batch_size, cfg.num_paths), device=dev) * (0.3 * cfg.user_distance_m) + 0.85 * cfg.user_distance_m
    azim = (torch.rand((cfg.batch_size, cfg.num_paths), device=dev) - 0.5) * 0.4 + theta_rad
    elev = (torch.rand((cfg.batch_size, cfg.num_paths), device=dev) - 0.5) * 0.15 + 0.5 * math.pi
    target_xyz = torch.stack(
        [
            r_scatter * torch.sin(azim) * torch.sin(elev),
            r_scatter * torch.cos(azim) * torch.sin(elev),
            r_scatter * torch.cos(elev),
        ],
        dim=-1,
    )  # (B, L, 3)

    b_near = nearfield_spherical_steering(elem_coords, target_xyz, fc_hz, include_amplitude_variation=True)  # (B, L, N)
    # Matching planar-wave far-field tangent approximation for the exact element geometry: r_n - r ≈ -(u_hat · element_pos)
    u_hat = target_xyz / torch.norm(target_xyz, dim=-1, keepdim=True).clamp(min=1e-6)
    center = elem_coords.mean(dim=0, keepdim=True)
    phase_far = (2.0 * math.pi / lam) * torch.einsum("blc,nc->bln", u_hat, elem_coords - center)
    a_far = torch.complex(torch.cos(phase_far), torch.sin(phase_far)) / math.sqrt(max(1, elem_coords.shape[0]))
    a_far = a_far.to(torch.complex64)

    # Measure normalized beamforming correlation |<b_near, a_far>|^2 (quantifies far-field energy spread loss)
    mismatch_corr = torch.abs(torch.sum(b_near.conj() * a_far, dim=-1)).pow(2)
    mean_farfield_match = float(torch.mean(mismatch_corr).item())

    gains = complex_normal((cfg.batch_size, cfg.num_paths), variance=1.0, device=dev)
    n_ant = cfg.num_tx_x * cfg.num_tx_y
    h_near = math.sqrt(n_ant / max(1, cfg.num_paths)) * torch.einsum("bl,bln->bn", gains, b_near.conj())

    gamma_db_km = float(itur_p676_attenuation_db_per_km(cfg.fc_ghz, water_vapor_g_m3=cfg.water_vapor_g_m3))
    abs_loss_db = float(itur_p676_path_absorption_db(cfg.fc_ghz, cfg.user_distance_m, water_vapor_g_m3=cfg.water_vapor_g_m3))
    fspl_db = 20.0 * math.log10(4.0 * math.pi * cfg.user_distance_m / lam)
    total_pl_db = fspl_db + abs_loss_db

    prov = [
        ParamProvenance("aperture_m", round(aperture_m, 4), "m", "Physical array aperture D", "cui_dai_nearfield_spherical", "IEEE TCOM 2022", "cui2022channel", f"Classical Z_R = 2D^2/lambda = {zr_m:.2f} m; d={cfg.user_distance_m:.1f} m."),
        ParamProvenance("fc_ghz", cfg.fc_ghz, "GHz", "Carrier frequency driving Rayleigh boundary & THz gas loss", "itur_p676", "Annex 1", "itur_p676_2022", f"Specific gaseous attenuation gamma(f) = {gamma_db_km:.2f} dB/km."),
    ]
    return ProvisionResult(
        block_id=META.id,
        seed=cfg.seed,
        elapsed_seconds=time.perf_counter() - t0,
        config=asdict(cfg),
        tensors={"h_near": h_near, "steering_near": b_near, "steering_far": a_far},
        metrics={
            "aperture_m": aperture_m,
            "rayleigh_distance_m": zr_m,
            "effective_rayleigh_distance_m": zr_eff_m,
            "is_in_nearfield": float(cfg.user_distance_m < zr_m),
            "farfield_beam_alignment_efficiency": mean_farfield_match,
            "molecular_absorption_db_per_km": gamma_db_km,
            "molecular_absorption_loss_db": abs_loss_db,
            "total_pathloss_db": total_pl_db,
            "mean_channel_power_per_ant": float(torch.mean(h_near.abs().pow(2)).item()),
        },
        diagnostics={
            "device": str(dev),
            "normalized_channel_power": 1.0,
            "min_tx_rx_distance_m": cfg.user_distance_m,
            "rayleigh_distance_m": zr_m,
            "assumed_farfield": False,
        },
        parameter_provenance=prov,
        mechanisms_used=list(META.default_mechanisms),
        standard_ids_used=list(META.standard_ids),
    )
