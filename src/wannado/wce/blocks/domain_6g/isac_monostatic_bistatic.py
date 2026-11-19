"""ISAC / JSAC Dual-Functional Channel, Target RCS & Cramér-Rao Bound (TR 22.837 / TR 38.901 Rel-19; Liu 2022).

[Lineage & Intuition]
- Family / Era / Tier : domain_6g / frontier / S (IMT-2030 ISAC Pillar)
- Parent              : `tr38901_uma_umi_rma`
- Key Diff            : Splits propagation into Background Multipath + Target Echo Channel (monostatic or
                        bistatic geometry with Swerling/dBsm RCS) and computes sensing range/angle CRB.
- Extension Hook [AI-Prompt Hook]:
    1. Couple with passive/STAR RIS to extend sensing around NLoS corners -> `passive_ris_cascade`.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Any, Literal

import torch

from wannado.wce.core.geometry import ula_farfield_steering
from wannado.wce.core.provenance import MECHANISM_PROVENANCE
from wannado.wce.core.types import ParamProvenance, ProvisionResult, WCEMeta
from wannado.wce.core.units import (
    SPEED_OF_LIGHT_MPS,
    db_to_linear,
    dbm_to_watt,
    thermal_noise_power_watt,
    wavelength_m,
)
from wannado.wce.core.utils import complex_normal, resolve_device, set_global_seed

META = WCEMeta(
    id="isac_jsac_channel",
    name="ISAC Monostatic/Bistatic Sensing & Communication Channel (TR 38.901 Rel-19 + Liu 2022)",
    family="domain_6g",
    era="frontier",
    tier="S",
    output_kind="complex_channel_tensor",
    supports_torch_complex=True,
    supports_autograd=True,
    standard_ids=("gpp_tr22837", "gpp_tr38901_isac", "itur_m2160"),
    parent_id="tr38901_uma_umi_rma",
    diff_from_parent="Adds radar equation target RCS reflection, monostatic/bistatic delay/Doppler, background clutter, and sensing CRB.",
    summary_plain="IMT-2030 & 3GPP Rel-19 Integrated Sensing and Communication channel synthesizing comm links + target echoes + CRB bounds.",
    when_to_use="DFRC beamforming, range/velocity/angle CRB-rate trade-offs, and target tracking ISAC benchmarks.",
    when_not_to_use="Pure communication links without sensing targets.",
    hyperparam_intuition={
        "topology": "monostatic (co-located STX/SRX, round-trip 2*d) vs bistatic (separated STX -> target -> SRX).",
        "target_rcs_dbsm": "Radar Cross Section in dBsm (e.g., pedestrian ~ -5 dBsm, UAV ~ -10 dBsm, car ~ +10 dBsm).",
        "comm_power_split": "Power/beam split alpha between communication user and sensing target illumination.",
    },
    extension_hooks=("Optimize joint DFRC precoder trading off sum-rate vs angle/range CRB",),
    module_path="wannado.wce.blocks.domain_6g.isac_monostatic_bistatic",
    default_mechanisms=("liu_isac_crb",),
    citation=MECHANISM_PROVENANCE["liu_isac_crb"].citation,
)


@dataclass
class ISACConfig:
    topology: Literal["monostatic", "bistatic"] = "monostatic"
    batch_size: int = 64
    num_tx: int = 16
    num_rx_sensing: int = 16
    fc_ghz: float = 28.0
    bandwidth_hz: float = 100e6
    tx_power_dbm: float = 33.0
    noise_figure_db: float = 7.0
    target_range_m: float = 60.0
    bistatic_rx_range_m: float = 45.0
    target_rcs_dbsm: float = 0.0
    target_speed_mps: float = 15.0
    comm_user_distance_m: float = 80.0
    comm_power_split: float = 0.5
    num_clutter_paths: int = 4
    device: str = "auto"
    seed: int = 42


def provision_isac_jsac_channel(config: ISACConfig | None = None, **overrides: Any) -> ProvisionResult:
    """Provision dual-functional ISAC communication + target/clutter sensing channels and CRB metrics."""
    cfg = config or ISACConfig()
    if overrides:
        cfg = ISACConfig(**{**asdict(cfg), **overrides})
    t0 = time.perf_counter()
    set_global_seed(cfg.seed)
    dev = resolve_device(cfg.device)

    fc_hz = cfg.fc_ghz * 1e9
    lam = wavelength_m(fc_hz)
    tx_w = dbm_to_watt(cfg.tx_power_dbm)
    noise_w = thermal_noise_power_watt(cfg.bandwidth_hz, noise_figure_db=cfg.noise_figure_db)

    d_tx_tgt = cfg.target_range_m
    d_tgt_rx = cfg.target_range_m if cfg.topology == "monostatic" else cfg.bistatic_rx_range_m
    rcs_m2 = float(db_to_linear(cfg.target_rcs_dbsm))

    # Bistatic/Monostatic Radar Equation: Pr = Pt * Gt * Gr * lam^2 * sigma / ((4*pi)^3 * d1^2 * d2^2)
    gt_lin = float(cfg.num_tx)
    gr_lin = float(cfg.num_rx_sensing)
    p_sense = tx_w * (1.0 - cfg.comm_power_split)
    rx_echo_w = (p_sense * gt_lin * gr_lin * (lam ** 2) * rcs_m2) / (((4.0 * math.pi) ** 3) * (d_tx_tgt ** 2) * (d_tgt_rx ** 2))
    base_sensing_snr_lin = rx_echo_w / max(noise_w, 1e-30)

    # Draw stochastic target RCS fluctuation BEFORE topology-dependent theta_r so
    # monostatic vs bistatic at identical geometry & seed share identical fading draws.
    alpha_tgt = complex_normal((cfg.batch_size,), variance=1.0, device=dev)
    rcs_fluct = alpha_tgt.abs().pow(2).clamp(min=1e-4)
    sensing_snr_vec = base_sensing_snr_lin * rcs_fluct

    # Range CRB: var(tau) >= 1 / (8 * pi^2 * B_rms^2 * SNR_sense), with B_rms^2 = B^2 / 12
    # Monostatic two-way round-trip tau = 2*R/c => sigma_R = (c / 2) * sigma_tau; bistatic uses factor 1.0.
    b_rms2 = (cfg.bandwidth_hz ** 2) / 12.0
    round_trip_factor = 2.0 if cfg.topology == "monostatic" else 1.0
    range_crb_vec = SPEED_OF_LIGHT_MPS / (
        round_trip_factor * 2.0 * math.sqrt(2.0) * math.pi * torch.sqrt((b_rms2 * sensing_snr_vec).clamp(min=1e-12))
    )

    # Target Echo Channel H_target[B, Nr_s, Nt] + Background Clutter Channel H_bg[B, Nr_s, Nt]
    theta_t = (torch.rand(cfg.batch_size, device=dev) - 0.5) * 1.2
    theta_r = theta_t if cfg.topology == "monostatic" else (torch.rand(cfg.batch_size, device=dev) - 0.5) * 1.2
    a_tx = ula_farfield_steering(cfg.num_tx, theta_t)
    a_rx = ula_farfield_steering(cfg.num_rx_sensing, theta_r)
    h_target = math.sqrt(cfg.num_tx * cfg.num_rx_sensing) * torch.einsum("b,bn,bm->bnm", alpha_tgt, a_rx, a_tx.conj())

    clut_ang_t = (torch.rand((cfg.batch_size, cfg.num_clutter_paths), device=dev) - 0.5) * math.pi
    clut_ang_r = (torch.rand((cfg.batch_size, cfg.num_clutter_paths), device=dev) - 0.5) * math.pi
    c_tx = ula_farfield_steering(cfg.num_tx, clut_ang_t)
    c_rx = ula_farfield_steering(cfg.num_rx_sensing, clut_ang_r)
    c_g = complex_normal((cfg.batch_size, cfg.num_clutter_paths), variance=1.0 / max(1, cfg.num_clutter_paths), device=dev)
    h_bg = math.sqrt(cfg.num_tx * cfg.num_rx_sensing) * torch.einsum("bl,bln,blm->bnm", c_g, c_rx, c_tx.conj())

    # Communication user channel (1-ant UE, MRT beamformed effective gain ||h_comm||_2^2)
    comm_fspl_lin = (lam / (4.0 * math.pi * cfg.comm_user_distance_m)) ** 2
    h_comm = complex_normal((cfg.batch_size, cfg.num_tx), variance=1.0, device=dev)
    comm_gain = h_comm.abs().pow(2).sum(dim=-1).clamp(min=1e-6)
    comm_snr_vec = (tx_w * cfg.comm_power_split * comm_fspl_lin / max(noise_w, 1e-30)) * comm_gain
    comm_rate_vec = torch.log2(1.0 + comm_snr_vec)

    prov = [
        ParamProvenance("topology", cfg.topology, "enum", "Monostatic vs Bistatic sensing geometry", "gpp_tr38901_isac", "Rel-19 ISAC", "3gpp_tr38901_isac_rel19", "Separates background multipath clutter from target echo links."),
        ParamProvenance("target_rcs_dbsm", cfg.target_rcs_dbsm, "dBsm", "Radar Cross Section of sensing target", "liu_isac_crb", "IEEE JSAC 2022", "liu2022integrated", "Scales target reflection power and sensing range/angle CRB."),
    ]
    return ProvisionResult(
        block_id=META.id,
        seed=cfg.seed,
        elapsed_seconds=time.perf_counter() - t0,
        config=asdict(cfg),
        tensors={
            "h_target": h_target.to(torch.complex64),
            "h_background_clutter": h_bg.to(torch.complex64),
            "h_comm": h_comm.to(torch.complex64),
            "snr_db": (10.0 * torch.log10(comm_snr_vec.clamp(min=1e-12))).to(torch.float32),
            "range_crb_m": range_crb_vec.to(torch.float32),
        },
        metrics={
            "sensing_snr_db": float(10.0 * torch.log10(torch.mean(sensing_snr_vec).clamp(min=1e-12)).item()),
            "range_rmse_crb_m": float(torch.median(range_crb_vec).item()),
            "comm_snr_db": float(10.0 * torch.log10(torch.mean(comm_snr_vec).clamp(min=1e-12)).item()),
            "comm_spectral_efficiency_bps_hz": float(torch.mean(comm_rate_vec).item()),
            "mean_channel_power": float(torch.mean(h_comm.abs().pow(2)).item()),
            "two_way_delay_us": float(((d_tx_tgt + d_tgt_rx) / SPEED_OF_LIGHT_MPS) * 1e6),
            "target_doppler_hz": float(((2.0 if cfg.topology == "monostatic" else 1.5) * cfg.target_speed_mps * fc_hz) / SPEED_OF_LIGHT_MPS),
        },
        diagnostics={"device": str(dev), "normalized_channel_power": 1.0},
        parameter_provenance=prov,
        mechanisms_used=list(META.default_mechanisms),
        standard_ids_used=list(META.standard_ids),
    )
