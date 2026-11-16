"""3GPP TR 38.901 Urban/Rural/Indoor Pathloss, LoS Probability, O2I Penetration & MIMO Channel.

[Lineage & Intuition]
- Family / Era / Tier : channel_3gpp / standard / S (Mandatory 5G/6G Macro/Micro Baseline)
- Parent              : `ppp_stochastic_geo`
- Key Diff            : Implements exact TR 38.901 Table 7.4.1-1 pathloss (UMa, UMi, RMa, InH),
                        Table 7.4.2-1 LoS probabilities, Table 7.4.3-2 O2I low/high loss, and MIMO fading.
- Extension Hook [AI-Prompt Hook]:
    1. Switch to factory clutter scenarios -> `tr38901_inf`.
    2. Expand into wideband OFDM subcarriers -> `tr38901_cdl_tdl`.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
import torch

from wannado.wce.core.provenance import STANDARDS_PROVENANCE
from wannado.wce.core.types import ParamProvenance, ProvisionResult, WCEMeta
from wannado.wce.core.units import (
    db_to_linear,
    dbm_to_watt,
    linear_to_db,
    thermal_noise_power_watt,
)
from wannado.wce.core.utils import complex_normal, resolve_device, set_global_seed
from wannado.wce.standards.gpp_38901 import (
    compute_los_probability_tr38901,
    compute_o2i_penetration_db,
    compute_pathloss_tr38901_db,
)

META = WCEMeta(
    id="tr38901_uma_umi_rma",
    name="3GPP TR 38.901 UMa / UMi / RMa / InH Spatial Channel",
    family="channel_3gpp",
    era="standard",
    tier="S",
    output_kind="complex_channel_tensor",
    supports_torch_complex=True,
    supports_autograd=True,
    standard_ids=("gpp_tr38901", "gpp_ts38211"),
    parent_id="ppp_stochastic_geo",
    diff_from_parent="Uses standardized 3GPP TR 38.901 dual-slope breakpoint pathloss, LoS probability, shadow fading, and O2I building loss.",
    summary_plain="Authoritative 3GPP 0.5-100 GHz cellular channel engine across Urban Macro, Urban Micro, Rural Macro, and Indoor Office.",
    when_to_use="Any 5G/6G cellular link-budget, resource allocation, handover, or coverage simulation requiring 3GPP compliance.",
    when_not_to_use="Industrial factories (`tr38901_inf`), LEO satellites (`ntn_leo_satellite`), or near-field XL-MIMO (`nearfield_xlmimo_thz`).",
    hyperparam_intuition={
        "scenario": "UMa (above-rooftop macro), UMi (street-canyon micro), RMa (rural wide), or InH (indoor office).",
        "fc_ghz": "Carrier frequency in GHz (0.5 - 100 GHz valid range in TR 38.901).",
        "o2i_fraction": "Fraction of indoor UEs incurring TR 38.901 building wall + depth penetration loss.",
    },
    extension_hooks=("Add multi-cell wrap-around ICI scheduling -> `multicell_interference_mac`",),
    module_path="wannado.wce.blocks.channel_3gpp.tr38901_macro_micro",
    default_mechanisms=("tr38901_pathloss_los", "tr38901_o2i_penetration"),
    citation=STANDARDS_PROVENANCE["gpp_tr38901"].citation,
)


@dataclass
class TR38901MacroMicroConfig:
    scenario: Literal["UMa", "UMi", "RMa", "InH"] = "UMa"
    num_links: int = 512
    num_tx: int = 4
    num_rx: int = 2
    fc_ghz: float = 3.5
    bandwidth_hz: float = 20e6
    tx_power_dbm: float = 43.0
    noise_figure_db: float = 7.0
    h_bs_m: float = 25.0
    h_ut_m: float = 1.5
    d_2d_min_m: float = 20.0
    d_2d_max_m: float = 500.0
    o2i_fraction: float = 0.2
    o2i_model: Literal["low_loss", "high_loss"] = "low_loss"
    device: str = "auto"
    seed: int = 42


def provision_tr38901_uma_umi_rma(config: TR38901MacroMicroConfig | None = None, **overrides: Any) -> ProvisionResult:
    """Provision 3GPP TR 38.901 large-scale pathloss + shadow fading + O2I + normalized MIMO small-scale fading."""
    cfg = config or TR38901MacroMicroConfig()
    if overrides:
        cfg = TR38901MacroMicroConfig(**{**asdict(cfg), **overrides})
    t0 = time.perf_counter()
    rng = set_global_seed(cfg.seed)
    dev = resolve_device(cfg.device)

    d_2d = np.sqrt(rng.uniform(cfg.d_2d_min_m ** 2, cfg.d_2d_max_m ** 2, size=cfg.num_links))
    p_los = compute_los_probability_tr38901(cfg.scenario, d_2d, h_ut_m=cfg.h_ut_m, h_bs_m=cfg.h_bs_m)
    is_los = rng.uniform(0.0, 1.0, size=cfg.num_links) < p_los

    pl_db, sf_std_db = compute_pathloss_tr38901_db(
        cfg.scenario,
        d_2d,
        cfg.fc_ghz,
        is_los=is_los,
        h_bs_m=cfg.h_bs_m,
        h_ut_m=cfg.h_ut_m,
    )
    sf_db = rng.normal(0.0, sf_std_db, size=cfg.num_links)

    o2i_mask = rng.uniform(0.0, 1.0, size=cfg.num_links) < cfg.o2i_fraction
    d_in = rng.uniform(0.0, 25.0 if cfg.scenario != "RMa" else 10.0, size=cfg.num_links)
    o2i_loss = np.where(
        o2i_mask,
        compute_o2i_penetration_db(cfg.fc_ghz, model=cfg.o2i_model, d_2d_in_m=d_in, rng=rng),
        0.0,
    )

    total_loss_db = pl_db + sf_db + o2i_loss
    tx_w = dbm_to_watt(cfg.tx_power_dbm)
    noise_w = thermal_noise_power_watt(cfg.bandwidth_hz, noise_figure_db=cfg.noise_figure_db)
    large_scale_gain = db_to_linear(-total_loss_db)
    avg_snr_lin = (tx_w * large_scale_gain) / max(noise_w, 1e-30)

    # Normalized small-scale MIMO matrix H_small[B, Nr, Nt] with unit mean entry power
    h_nlos = complex_normal((cfg.num_links, cfg.num_rx, cfg.num_tx), variance=1.0, device=dev)
    k_lin = torch.where(
        torch.from_numpy(is_los).to(dev),
        torch.full((cfg.num_links,), float(db_to_linear(9.0 if cfg.scenario == "UMa" else 7.0)), device=dev),
        torch.zeros((cfg.num_links,), device=dev),
    ).view(-1, 1, 1)
    phase_los = torch.rand((cfg.num_links, cfg.num_rx, cfg.num_tx), device=dev) * (2.0 * math.pi)
    h_los = torch.complex(torch.cos(phase_los), torch.sin(phase_los))
    h_small = (torch.sqrt(k_lin / (k_lin + 1.0)) * h_los + torch.sqrt(1.0 / (k_lin + 1.0)) * h_nlos).to(torch.complex64)

    # Apply large-scale channel amplitude sqrt(large_scale_gain) cleanly separated from normalized H_small
    gain_t = torch.from_numpy(np.sqrt(large_scale_gain)).to(device=dev, dtype=torch.float32).view(-1, 1, 1)
    h_full = (h_small * gain_t).to(torch.complex64)

    prov = [
        ParamProvenance("scenario", cfg.scenario, "enum", "3GPP TR 38.901 deployment environment", "gpp_tr38901", "Table 7.4.1-1", "3gpp_tr38901", "Selects standardized breakpoint pathloss, shadow fading sigma_SF, and LoS curves."),
        ParamProvenance("fc_ghz", cfg.fc_ghz, "GHz", "Center carrier frequency", "gpp_tr38901", "Clause 7.4", "3gpp_tr38901", "Valid across 0.5 to 100 GHz in TR 38.901."),
        ParamProvenance("o2i_model", cfg.o2i_model, "enum", "Building penetration material mix", "gpp_tr38901", "Table 7.4.3-2", "3gpp_tr38901", "Low-loss (glass/concrete) vs high-loss (IRR metallic coated glass)."),
    ]
    return ProvisionResult(
        block_id=META.id,
        seed=cfg.seed,
        elapsed_seconds=time.perf_counter() - t0,
        config=asdict(cfg),
        tensors={
            "h_small": h_small,
            "h_full": h_full,
            "pathloss_db": torch.from_numpy(total_loss_db).to(torch.float32),
            "snr_db": torch.from_numpy(linear_to_db(avg_snr_lin)).to(torch.float32),
            "d_2d_m": torch.from_numpy(d_2d).to(torch.float32),
            "is_los": torch.from_numpy(is_los),
        },
        metrics={
            "los_fraction": float(np.mean(is_los)),
            "mean_pathloss_db": float(np.mean(total_loss_db)),
            "median_snr_db": float(np.median(linear_to_db(avg_snr_lin))),
            "mean_spectral_efficiency_bps_hz": float(np.mean(np.log2(1.0 + avg_snr_lin))),
            "mean_small_scale_power": float(torch.mean(h_small.abs().pow(2)).item()),
        },
        diagnostics={"device": str(dev), "normalized_channel_power": 1.0, "shadow_fading_std_db": sf_std_db},
        parameter_provenance=prov,
        mechanisms_used=list(META.default_mechanisms),
        standard_ids_used=list(META.standard_ids),
    )
