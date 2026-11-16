"""AWGN Link & Complex Baseband Thermal Noise Baseline (Shannon, BSTJ 1948).

[Lineage & Intuition]
- Family / Era / Tier : channel_classic / classic / S (Information-Theoretic Baseline Ceiling)
- Parent              : None (Root of all wireless channel & link models)
- Key Physics         : Realizes thermal noise floor N = k_B * T * B * NF, additive CN(0, sigma^2)
                        baseband impairment, and exact Shannon capacity C = B * log2(1 + SNR).
- Extension Hook [AI-Prompt Hook]:
    1. Replace fixed gain with frequency-flat Rayleigh/Rician fading -> `rayleigh_clarke` / `rician_los`.
    2. Penalize capacity for finite short-packet blocklengths -> `hrllc_fbl_urllc`.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Any

import torch

from wannado.wce.core.provenance import MECHANISM_PROVENANCE
from wannado.wce.core.types import LitCitation, ParamProvenance, ProvisionResult, WCEMeta
from wannado.wce.core.units import (
    db_to_linear,
    dbm_to_watt,
    linear_to_db,
    thermal_noise_power_dbm,
    thermal_noise_power_watt,
    watt_to_dbm,
)
from wannado.wce.core.utils import complex_normal, resolve_device, set_global_seed

META = WCEMeta(
    id="awgn_link",
    name="AWGN Baseband Channel & Shannon Capacity Baseline",
    family="channel_classic",
    era="classic",
    tier="S",
    output_kind="complex_channel_tensor",
    supports_torch_complex=True,
    supports_autograd=True,
    standard_ids=("itur_m2160",),
    parent_id=None,
    diff_from_parent="Root baseline defining thermal noise power N0*B*NF and Shannon AWGN capacity.",
    summary_plain="Idealized fading-free complex Gaussian noise baseline with exact thermal noise floor and Shannon upper bound.",
    when_to_use="Sanity-checking SNR definitions, thermal noise units, or serving as the fading-free capacity ceiling.",
    when_not_to_use="Realistic mobile multipath, shadowing, or multi-antenna spatial correlation studies.",
    hyperparam_intuition={
        "bandwidth_hz": "Channel bandwidth; wider B increases capacity linearly but raises thermal noise k_B*T*B.",
        "noise_figure_db": "Receiver hardware imperfection above ideal thermal floor (typical 5-9 dB for UEs).",
        "tx_power_dbm": "Transmitter EIRP/conducted power in dBm.",
        "pathloss_db": "Deterministic link attenuation subtracted from Tx power before computing SNR.",
    },
    extension_hooks=(
        "Add time-varying multipath fading -> `rayleigh_clarke` / `rician_los`",
        "Apply finite-blocklength dispersion loss -> `hrllc_fbl_urllc`",
    ),
    module_path="wannado.wce.blocks.channel_classic.awgn_shannon1948",
    default_mechanisms=("shannon_awgn",),
    citation=MECHANISM_PROVENANCE["shannon_awgn"].citation,
)


@dataclass
class AWGNConfig:
    batch_size: int = 256
    num_symbols: int = 128
    fc_ghz: float = 3.5
    bandwidth_hz: float = 20e6
    tx_power_dbm: float = 23.0
    pathloss_db: float = 105.0
    noise_figure_db: float = 7.0
    temp_k: float = 290.0
    device: str = "auto"
    seed: int = 42


def provision_awgn_link(config: AWGNConfig | None = None, **overrides: Any) -> ProvisionResult:
    """Provision a calibrated complex64 AWGN baseband link and compute exact Shannon rate."""
    cfg = config or AWGNConfig()
    if overrides:
        cfg = AWGNConfig(**{**asdict(cfg), **overrides})
    t0 = time.perf_counter()
    set_global_seed(cfg.seed)
    dev = resolve_device(cfg.device)

    noise_w = thermal_noise_power_watt(cfg.bandwidth_hz, noise_figure_db=cfg.noise_figure_db, temp_k=cfg.temp_k)
    noise_dbm = thermal_noise_power_dbm(cfg.bandwidth_hz, noise_figure_db=cfg.noise_figure_db, temp_k=cfg.temp_k)
    rx_power_dbm = cfg.tx_power_dbm - cfg.pathloss_db
    rx_power_w = dbm_to_watt(rx_power_dbm)
    snr_lin = rx_power_w / max(noise_w, 1e-30)
    snr_db = float(linear_to_db(snr_lin))
    shannon_se = math.log2(1.0 + snr_lin)
    capacity_mbps = (cfg.bandwidth_hz * shannon_se) / 1e6

    # Unit-gain deterministic channel h = 1 + 0j and calibrated CN(0, 1/snr_lin) normalized noise
    h = torch.ones((cfg.batch_size, cfg.num_symbols), dtype=torch.complex64, device=dev)
    noise_unit = complex_normal((cfg.batch_size, cfg.num_symbols), variance=1.0 / max(snr_lin, 1e-12), device=dev)
    pilot_qpsk = torch.complex(
        torch.sign(torch.randn((cfg.batch_size, cfg.num_symbols), device=dev)) / math.sqrt(2.0),
        torch.sign(torch.randn((cfg.batch_size, cfg.num_symbols), device=dev)) / math.sqrt(2.0),
    ).to(torch.complex64)
    rx_samples = h * pilot_qpsk + noise_unit

    prov = [
        ParamProvenance("bandwidth_hz", cfg.bandwidth_hz, "Hz", "Signal bandwidth", "gpp_ts38211", "Clause 4.2", "3gpp_ts38211", "Defines noise integration bandwidth B and capacity scaling."),
        ParamProvenance("noise_figure_db", cfg.noise_figure_db, "dB", "Receiver RF front-end noise figure", "gpp_tr38913", "Clause 6", "3gpp_tr38913", "Standard UE/gNB thermal noise floor offset above k_B*T*B."),
        ParamProvenance("tx_power_dbm", cfg.tx_power_dbm, "dBm", "Transmitter power", "gpp_tr38913", "Clause 6", "3gpp_tr38913", "Typical 3GPP UE max transmit power (23 dBm)."),
        ParamProvenance("shannon_capacity", round(capacity_mbps, 4), "Mbps", "Asymptotic AWGN capacity ceiling", "shannon_awgn", "BSTJ 1948", "shannon1948mathematical", "Exact B*log2(1 + P_rx / (k_B*T*B*NF))."),
    ]
    return ProvisionResult(
        block_id=META.id,
        seed=cfg.seed,
        elapsed_seconds=time.perf_counter() - t0,
        config=asdict(cfg),
        tensors={"h": h, "noise": noise_unit, "rx_samples": rx_samples},
        metrics={
            "snr_db": snr_db,
            "noise_power_dbm": noise_dbm,
            "rx_power_dbm": float(watt_to_dbm(rx_power_w)),
            "spectral_efficiency_bps_hz": shannon_se,
            "shannon_capacity_mbps": capacity_mbps,
            "mean_channel_power": float(torch.mean(h.abs().pow(2)).item()),
            "empirical_noise_var": float(torch.mean(noise_unit.abs().pow(2)).item()),
        },
        diagnostics={"device": str(dev), "normalized_channel_power": 1.0},
        parameter_provenance=prov,
        mechanisms_used=list(META.default_mechanisms),
        standard_ids_used=list(META.standard_ids),
    )
