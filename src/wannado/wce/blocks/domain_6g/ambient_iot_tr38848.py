"""3GPP TR 38.848 Ambient IoT (A-IoT) RF Energy Harvesting & Backscatter Two-Hop Link Budget.

[Lineage & Intuition]
- Family / Era / Tier : domain_6g / standard / A (IMT-2030 Massive & Sustainable IoT)
- Parent              : `awgn_link`
- Key Diff            : Models batteryless uW-class backscatter devices powered by RF carrier waves across
                        TR 38.848 Topology 1 (direct BS-to-Device) and Topology 2 (intermediate helper).
- Extension Hook [AI-Prompt Hook]:
    1. Scale device density up to 10^6 - 10^8 / km^2 random access contention.
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
    watt_to_dbm,
    wavelength_m,
)
from wannado.wce.core.utils import set_global_seed

META = WCEMeta(
    id="ambient_iot_backscatter",
    name="3GPP TR 38.848 Ambient IoT Energy Harvesting & Backscatter Environment",
    family="domain_6g",
    era="standard",
    tier="A",
    output_kind="link_capacity_metric",
    supports_torch_complex=False,
    supports_autograd=False,
    standard_ids=("gpp_tr38848", "itur_m2160"),
    parent_id="awgn_link",
    diff_from_parent="Computes dual-leg RF harvesting power (uW) and backscatter reader SNR under TR 38.848 topologies.",
    summary_plain="3GPP Rel-18 Ambient IoT (A-IoT) batteryless backscatter two-hop link budget and energy-harvesting feasibility checker.",
    when_to_use="Zero-energy IoT, passive tags, RF backscatter networks, and IMT-2030 Massive/Sustainable Communication.",
    when_not_to_use="Active battery-powered cellular UEs (`tr38901_uma_umi_rma`).",
    hyperparam_intuition={
        "topology": "topology_1 (direct BS <-> tag) vs topology_2 (intermediate helper UE close to tag).",
        "tag_sensitivity_dbm": "Minimum harvested RF power to wake up batteryless tag (typically -20 to -10 dBm).",
    },
    extension_hooks=("Integrate non-linear RF-DC energy harvester saturation curve",),
    module_path="wannado.wce.blocks.domain_6g.ambient_iot_tr38848",
    default_mechanisms=("shannon_awgn",),
    citation=STANDARDS_PROVENANCE["gpp_tr38848"].citation,
)


@dataclass
class AmbientIoTConfig:
    topology: Literal["topology_1", "topology_2"] = "topology_2"
    num_devices: int = 256
    fc_ghz: float = 0.9
    bandwidth_hz: float = 180e3
    carrier_tx_power_dbm: float = 33.0
    helper_distance_m: float = 12.0
    direct_distance_m: float = 80.0
    backscatter_loss_db: float = 6.0
    tag_sensitivity_dbm: float = -20.0
    noise_figure_db: float = 6.0
    seed: int = 42


def provision_ambient_iot_backscatter(config: AmbientIoTConfig | None = None, **overrides: Any) -> ProvisionResult:
    """Evaluate TR 38.848 two-hop forward powering & reverse backscatter SNR across Ambient IoT devices."""
    cfg = config or AmbientIoTConfig()
    if overrides:
        cfg = AmbientIoTConfig(**{**asdict(cfg), **overrides})
    t0 = time.perf_counter()
    rng = set_global_seed(cfg.seed)

    lam = wavelength_m(cfg.fc_ghz * 1e9)
    max_d = cfg.direct_distance_m if cfg.topology == "topology_1" else cfg.helper_distance_m
    dists = rng.uniform(2.0, max(3.0, max_d), size=cfg.num_devices)
    fading_pow = rng.exponential(scale=1.0, size=cfg.num_devices)

    # Forward energizing link + backscatter modulation loss -> two-way return SNR
    path_gain_fwd = ((lam / (4.0 * math.pi * dists)) ** 2) * fading_pow
    tx_w = dbm_to_watt(cfg.carrier_tx_power_dbm)
    harvested_w = tx_w * path_gain_fwd * 0.5  # 50% RF-DC / impedance duty split
    harvested_dbm = np.array([watt_to_dbm(w) for w in harvested_w])
    active_mask = harvested_dbm >= cfg.tag_sensitivity_dbm

    bs_loss_lin = float(db_to_linear(-cfg.backscatter_loss_db))
    rx_reader_w = tx_w * (path_gain_fwd ** 2) * bs_loss_lin
    noise_w = thermal_noise_power_watt(cfg.bandwidth_hz, noise_figure_db=cfg.noise_figure_db)
    reader_snr_db = linear_to_db(rx_reader_w / max(noise_w, 1e-30))

    prov = [
        ParamProvenance("topology", cfg.topology, "enum", "TR 38.848 A-IoT deployment topology", "gpp_tr38848", "Clause 4.2", "3gpp_tr38848", "Topology 1 (direct BS) vs Topology 2 (helper-assisted short range)."),
        ParamProvenance("tag_sensitivity_dbm", cfg.tag_sensitivity_dbm, "dBm", "Minimum wake-up RF power (~10 uW at -20 dBm)", "gpp_tr38848", "Clause 5", "3gpp_tr38848", "Governs energy-harvesting activation probability."),
    ]
    return ProvisionResult(
        block_id=META.id,
        seed=cfg.seed,
        elapsed_seconds=time.perf_counter() - t0,
        config=asdict(cfg),
        tensors={"harvested_dbm": torch.from_numpy(harvested_dbm).to(torch.float32), "reader_snr_db": torch.from_numpy(reader_snr_db).to(torch.float32)},
        metrics={
            "device_activation_rate": float(np.mean(active_mask)),
            "mean_harvested_uw": float(np.mean(harvested_w) * 1e6),
            "median_reader_snr_db": float(np.median(reader_snr_db)),
        },
        diagnostics={"normalized_channel_power": 1.0},
        parameter_provenance=prov,
        mechanisms_used=list(META.default_mechanisms),
        standard_ids_used=list(META.standard_ids),
    )
