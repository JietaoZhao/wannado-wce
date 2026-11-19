"""3GPP TR 38.901 TDL/CDL Wideband OFDM Channel with TS 38.211 Numerology.

[Lineage & Intuition]
- Family / Era / Tier : channel_3gpp / standard / S (Standard Wideband OFDM-MIMO Link Environment)
- Parent              : `tr38901_uma_umi_rma`
- Key Diff            : Produces frequency-selective, time-varying MIMO-OFDM channel grids H[B, Nr, Nt, N_sc, T]
                        from standardized TDL-A..E / CDL-A..E delay-power-angle profiles & TS 38.211 SCS.
- Extension Hook [AI-Prompt Hook]:
    1. Feed H_freq directly into LS/LMMSE CSI estimation -> `ls_lmmse_csi`.
    2. Compare OFDM against Delay-Doppler OTFS under high speed -> `otfs_delay_doppler`.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Any, Literal

import torch

from wannado.wce.core.geometry import ula_farfield_steering
from wannado.wce.core.provenance import STANDARDS_PROVENANCE
from wannado.wce.core.types import ParamProvenance, ProvisionResult, WCEMeta
from wannado.wce.core.units import doppler_max_hz
from wannado.wce.core.utils import complex_normal, resolve_device, set_global_seed
from wannado.wce.standards.gpp_38901 import CDL_PROFILES, TDL_PROFILES
from wannado.wce.standards.gpp_extensions import get_nr_numerology

META = WCEMeta(
    id="tr38901_cdl_tdl",
    name="3GPP TR 38.901 TDL/CDL Wideband MIMO-OFDM Channel",
    family="channel_3gpp",
    era="standard",
    tier="S",
    output_kind="complex_channel_tensor",
    supports_torch_complex=True,
    supports_autograd=True,
    standard_ids=("gpp_tr38901", "gpp_ts38211"),
    parent_id="tr38901_uma_umi_rma",
    diff_from_parent="Synthesizes frequency-domain OFDM subcarrier-by-slot channels from TR 38.901 Clause 7.7 delay profiles.",
    summary_plain="Frequency-selective wideband MIMO-OFDM link channel supporting TDL-A..E and spatial CDL-A..E with 5G NR numerology mu=0..4.",
    when_to_use="OFDM resource block allocation, CSI estimation/compression, MIMO precoding, and link-level BLER.",
    when_not_to_use="High-Doppler OTFS comparison (`otfs_delay_doppler`) or near-field spherical wavefronts (`nearfield_xlmimo_thz`).",
    hyperparam_intuition={
        "profile": "TDL-A/B/C & CDL-A/B/C are NLOS; TDL-D/E & CDL-D/E have a dominant LoS path.",
        "numerology_mu": "NR subcarrier spacing index 0..4 (15*2^mu kHz); higher mu handles high Doppler with shorter slots.",
        "delay_spread_ns": "RMS delay spread scaling multipath delays; larger DS creates sharper frequency selectivity.",
    },
    extension_hooks=("Evaluate pilot CSI estimation NMSE -> `ls_lmmse_csi`",),
    module_path="wannado.wce.blocks.channel_3gpp.tr38901_cdl_tdl",
    default_mechanisms=("nr_ofdm_numerology", "clarke_jakes_doppler"),
    citation=STANDARDS_PROVENANCE["gpp_ts38211"].citation,
)


@dataclass
class CDLTDLConfig:
    profile: Literal["TDL-A", "TDL-B", "TDL-C", "TDL-D", "TDL-E", "CDL-A", "CDL-B", "CDL-C", "CDL-D", "CDL-E"] = "CDL-C"
    batch_size: int = 32
    num_tx: int = 4
    num_rx: int = 2
    num_subcarriers: int = 64
    num_slots: int = 14
    numerology_mu: int = 1
    fc_ghz: float = 3.5
    delay_spread_ns: float = 100.0
    speed_mps: float = 10.0
    device: str = "auto"
    seed: int = 42


def provision_tr38901_cdl_tdl(config: CDLTDLConfig | None = None, **overrides: Any) -> ProvisionResult:
    """Synthesize unit-power wideband OFDM MIMO channel H_freq[B, Nr, Nt, Nsc, T]."""
    cfg = config or CDLTDLConfig()
    if overrides:
        cfg = CDLTDLConfig(**{**asdict(cfg), **overrides})
    t0 = time.perf_counter()
    set_global_seed(cfg.seed)
    dev = resolve_device(cfg.device)

    num_info = get_nr_numerology(cfg.numerology_mu)
    scs_hz = float(num_info["scs_khz"]) * 1e3
    slot_sec = float(num_info["slot_duration_ms"]) * 1e-3

    is_cdl = cfg.profile.startswith("CDL")
    prof = CDL_PROFILES[cfg.profile] if is_cdl else TDL_PROFILES[cfg.profile]
    delays_sec = torch.as_tensor(prof["delays_norm"], device=dev, dtype=torch.float32) * (cfg.delay_spread_ns * 1e-9)
    p_lin = torch.pow(10.0, torch.as_tensor(prof["powers_db"], device=dev, dtype=torch.float32) / 10.0)
    p_lin = p_lin / p_lin.sum()
    n_taps = delays_sec.shape[0]
    k_db = float(prof.get("k_db", float("-inf")))
    has_los_ray = math.isfinite(k_db)

    # Spatial steering (CDL uses sub-ray angular spread; TDL uses i.i.d. MIMO; D/E profiles honor k_db LoS ray)
    fd = doppler_max_hz(cfg.speed_mps, cfg.fc_ghz * 1e9)
    t_grid = torch.arange(cfg.num_slots, device=dev, dtype=torch.float32) * slot_sec
    f_grid = (torch.arange(cfg.num_subcarriers, device=dev, dtype=torch.float32) - 0.5 * cfg.num_subcarriers) * scs_hz

    tap_angles = (torch.rand((cfg.batch_size, n_taps), device=dev) * 2.0 - 1.0) * math.pi
    doppler_phase = 2.0 * math.pi * fd * torch.cos(tap_angles).unsqueeze(-1) * t_grid  # (B, L, T)
    freq_phase = -2.0 * math.pi * delays_sec.unsqueeze(-1) * f_grid  # (L, Nsc)
    tf_phase = torch.complex(
        torch.cos(freq_phase.unsqueeze(0).unsqueeze(-1) + doppler_phase.unsqueeze(2)),
        torch.sin(freq_phase.unsqueeze(0).unsqueeze(-1) + doppler_phase.unsqueeze(2)),
    )  # (B, L, Nsc, T)

    if is_cdl:
        aod_rad = torch.deg2rad(torch.as_tensor(prof["aod_deg"], device=dev, dtype=torch.float32))
        aoa_rad = torch.deg2rad(torch.as_tensor(prof["aoa_deg"], device=dev, dtype=torch.float32))
        # Expand each NLOS cluster into sub-rays across cluster angular spread (~3 deg ASD / ~8 deg ASA)
        n_sub = 8
        offsets = torch.linspace(-1.0, 1.0, n_sub, device=dev, dtype=torch.float32)
        sub_aod = aod_rad.unsqueeze(-1) + offsets * math.radians(3.0)
        sub_aoa = aoa_rad.unsqueeze(-1) + offsets * math.radians(8.0)
        a_tx = ula_farfield_steering(cfg.num_tx, sub_aod)  # (L, M, Nt)
        a_rx = ula_farfield_steering(cfg.num_rx, sub_aoa)  # (L, M, Nr)
        ray_coeff = complex_normal((cfg.batch_size, n_taps, n_sub), variance=1.0 / n_sub, device=dev)
        spatial_nlos = math.sqrt(cfg.num_tx * cfg.num_rx) * torch.einsum("blm,lmn,lmk->blnk", ray_coeff, a_rx, a_tx.conj())
        if has_los_ray:
            # TR 38.901 Tables 7.7.1-4/5: tap 0 IS the specular LoS component (power p_lin[0]) and taps 1..L-1
            # are the Rayleigh clusters (including tap 1 at delay 0.0000), so p_lin already embodies k_db exactly.
            a_tx0 = ula_farfield_steering(cfg.num_tx, aod_rad[:1])
            a_rx0 = ula_farfield_steering(cfg.num_rx, aoa_rad[:1])
            phi0 = torch.rand((cfg.batch_size, 1, 1), device=dev) * (2.0 * math.pi)
            h_los0 = math.sqrt(cfg.num_tx * cfg.num_rx) * torch.complex(torch.cos(phi0), torch.sin(phi0)) * torch.einsum(
                "ln,lm->lnm", a_rx0, a_tx0.conj()
            )
            spatial = spatial_nlos * torch.sqrt(p_lin).view(1, n_taps, 1, 1)
            spatial[:, 0] = math.sqrt(float(p_lin[0].item())) * h_los0
        else:
            spatial = spatial_nlos * torch.sqrt(p_lin).view(1, n_taps, 1, 1)
    else:
        spatial_nlos = complex_normal((cfg.batch_size, n_taps, cfg.num_rx, cfg.num_tx), variance=1.0, device=dev)
        if has_los_ray:
            # TR 38.901 Tables 7.7.2-4/5: tap 0 IS the specular LoS component (power p_lin[0]) and taps 1..L-1
            # are the Rayleigh taps (tap 1 at delay 0.0000 gives exact tap-1 Rician K = p_lin[0]/p_lin[1]).
            phi0 = torch.rand((cfg.batch_size, 1, 1), device=dev) * (2.0 * math.pi)
            spatial = spatial_nlos * torch.sqrt(p_lin).view(1, n_taps, 1, 1)
            spatial[:, 0] = math.sqrt(float(p_lin[0].item())) * torch.complex(torch.cos(phi0), torch.sin(phi0))
        else:
            spatial = spatial_nlos * torch.sqrt(p_lin).view(1, n_taps, 1, 1)

    h_freq = torch.einsum("blnm,blft->bnmft", spatial, tf_phase).to(torch.complex64)

    prov = [
        ParamProvenance("profile", cfg.profile, "enum", "TR 38.901 TDL/CDL delay-power profile", "gpp_tr38901", "Clause 7.7", "3gpp_tr38901", "Tabulated relative tap delays, powers, and AoA/AoD clusters."),
        ParamProvenance("numerology_mu", cfg.numerology_mu, "enum", f"5G NR SCS={scs_hz/1e3:.0f} kHz, slot={slot_sec*1e3:.3f} ms", "gpp_ts38211", "Clause 4.2", "3gpp_ts38211", "Governs subcarrier spacing and OFDM time-slot granularity."),
        ParamProvenance("delay_spread_ns", cfg.delay_spread_ns, "ns", "RMS delay spread scaling", "gpp_tr38901", "Table 7.7.3-1", "3gpp_tr38901", "Determines channel coherence bandwidth across subcarriers."),
    ]
    return ProvisionResult(
        block_id=META.id,
        seed=cfg.seed,
        elapsed_seconds=time.perf_counter() - t0,
        config=asdict(cfg),
        tensors={"h_freq": h_freq, "tap_delays_ns": delays_sec * 1e9, "tap_powers_lin": p_lin},
        metrics={
            "subcarrier_spacing_khz": scs_hz / 1e3,
            "slot_duration_ms": slot_sec * 1e3,
            "max_doppler_hz": fd,
            "mean_channel_power": float(torch.mean(h_freq.abs().pow(2)).item()),
            "rms_delay_spread_ns": float(torch.sqrt(torch.sum(p_lin * (delays_sec * 1e9) ** 2) - torch.sum(p_lin * (delays_sec * 1e9)) ** 2).item()),
        },
        diagnostics={"device": str(dev), "normalized_channel_power": 1.0},
        parameter_provenance=prov,
        mechanisms_used=list(META.default_mechanisms),
        standard_ids_used=list(META.standard_ids),
    )
