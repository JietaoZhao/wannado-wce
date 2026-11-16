"""IEEE 802.11bf WLAN Sensing CSI Channel & IEEE 802.11bn (Wi-Fi 8 UHR) Multi-AP Co-SR.

[Lineage & Intuition]
- Family / Era / Tier : domain_6g / standard / A
- Parent              : `isac_jsac_channel`
- Key Diff            : Models IEEE 802.11bf static environment vs dynamic human micro-Doppler CSI
                        paths across sub-7 GHz / 60 GHz, plus 802.11bn Multi-AP Co-SR SNR boost.
- Extension Hook [AI-Prompt Hook]:
    1. Extract CSI amplitude/phase difference time series for breathing/gesture detection ML models.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Any

import torch

from wannado.wce.core.provenance import STANDARDS_PROVENANCE
from wannado.wce.core.types import ParamProvenance, ProvisionResult, WCEMeta
from wannado.wce.core.units import SPEED_OF_LIGHT_MPS, db_to_linear, linear_to_db
from wannado.wce.core.utils import complex_normal, resolve_device, set_global_seed

META = WCEMeta(
    id="wifi_sensing_80211bf",
    name="IEEE 802.11bf WLAN Sensing & 802.11bn Multi-AP Co-SR Environment",
    family="domain_6g",
    era="standard",
    tier="A",
    output_kind="complex_channel_tensor",
    supports_torch_complex=True,
    supports_autograd=True,
    standard_ids=("ieee_80211bf", "ieee_80211bn"),
    parent_id="isac_jsac_channel",
    diff_from_parent="Decomposes Wi-Fi CSI into static wall/furniture paths plus dynamic human micro-Doppler reflections.",
    summary_plain="WLAN sensing CSI stream generator (802.11bf) paired with Wi-Fi 8 Multi-AP Coordinated Spatial Reuse (802.11bn).",
    when_to_use="Indoor human breathing/motion/gesture sensing over Wi-Fi CSI and Multi-AP dense WLAN studies.",
    when_not_to_use="Outdoor cellular 5G/6G (`tr38901_uma_umi_rma` / `isac_jsac_channel`).",
    hyperparam_intuition={
        "motion_hz": "Dynamic target oscillation frequency (e.g. 0.3 Hz human breathing or higher gesture Doppler).",
        "multi_ap_cosr_gain_db": "Interference suppression gain achieved by 802.11bn Multi-AP coordinated spatial reuse.",
    },
    extension_hooks=("Train gesture/respiration classifier directly on tensors['csi_stream']",),
    module_path="wannado.wce.blocks.domain_6g.wifi_sensing_80211bf",
    default_mechanisms=(),
    citation=STANDARDS_PROVENANCE["ieee_80211bf"].citation,
)


@dataclass
class WiFiSensingConfig:
    batch_size: int = 32
    num_subcarriers: int = 56
    num_packets: int = 128
    packet_interval_ms: float = 10.0
    fc_ghz: float = 5.2
    bandwidth_hz: float = 20e6
    motion_hz: float = 0.35
    motion_displacement_m: float = 0.015
    dynamic_to_static_ratio_db: float = -12.0
    multi_ap_cosr_gain_db: float = 4.5
    device: str = "auto"
    seed: int = 42


def provision_wifi_sensing_80211bf(config: WiFiSensingConfig | None = None, **overrides: Any) -> ProvisionResult:
    """Synthesize IEEE 802.11bf static+dynamic CSI sensing tensors and 802.11bn Multi-AP Co-SR gains."""
    cfg = config or WiFiSensingConfig()
    if overrides:
        cfg = WiFiSensingConfig(**{**asdict(cfg), **overrides})
    t0 = time.perf_counter()
    set_global_seed(cfg.seed)
    dev = resolve_device(cfg.device)

    lam = SPEED_OF_LIGHT_MPS / (cfg.fc_ghz * 1e9)
    t_sec = torch.arange(cfg.num_packets, device=dev, dtype=torch.float32) * (cfg.packet_interval_ms * 1e-3)
    f_offsets = torch.linspace(-0.5, 0.5, cfg.num_subcarriers, device=dev) * cfg.bandwidth_hz

    # Static multipath component constant across packets
    tau_static = torch.rand((cfg.batch_size, 1), device=dev) * 80e-9
    h_stat_base = complex_normal((cfg.batch_size, 1), variance=1.0, device=dev)
    static_phase = -2.0 * math.pi * tau_static * f_offsets.unsqueeze(0)
    h_static = h_stat_base.unsqueeze(-1) * torch.complex(torch.cos(static_phase), torch.sin(static_phase)).unsqueeze(1)

    # Dynamic human chest/target reflection modulating path delay by displacement(t)
    dyn_amp = math.sqrt(float(db_to_linear(cfg.dynamic_to_static_ratio_db)))
    disp_t = cfg.motion_displacement_m * torch.sin(2.0 * math.pi * cfg.motion_hz * t_sec)  # (T,)
    dyn_phase = (-4.0 * math.pi / lam) * disp_t.view(1, cfg.num_packets, 1) - 2.0 * math.pi * (40e-9) * f_offsets.view(1, 1, cfg.num_subcarriers)
    h_dyn = dyn_amp * complex_normal((cfg.batch_size, 1, 1), variance=1.0, device=dev) * torch.complex(torch.cos(dyn_phase), torch.sin(dyn_phase))

    csi = (h_static + h_dyn).to(torch.complex64)
    # 802.11bn Multi-AP Co-SR baseline SINR uplift check
    uncoord_sinr_db = 8.0
    cosr_sinr_db = uncoord_sinr_db + cfg.multi_ap_cosr_gain_db

    prov = [
        ParamProvenance("motion_hz", cfg.motion_hz, "Hz", "Target micro-Doppler oscillation rate", "ieee_80211bf", "TGbf Channel Models", "ieee_80211bf", "Modulates dynamic reflection phase across consecutive CSI packets."),
        ParamProvenance("multi_ap_cosr_gain_db", cfg.multi_ap_cosr_gain_db, "dB", "Wi-Fi 8 Multi-AP Co-SR SINR uplift", "ieee_80211bn", "TGbn UHR", "ieee_80211bn", "Captures coordinated spatial reuse interference mitigation."),
    ]
    return ProvisionResult(
        block_id=META.id,
        seed=cfg.seed,
        elapsed_seconds=time.perf_counter() - t0,
        config=asdict(cfg),
        tensors={"csi_stream": csi, "h_static": h_static, "h_dynamic": h_dyn},
        metrics={
            "dynamic_to_static_power_db": cfg.dynamic_to_static_ratio_db,
            "csi_temporal_variance": float(torch.mean(torch.var(csi.abs(), dim=1)).item()),
            "cosr_effective_sinr_db": cosr_sinr_db,
            "mean_channel_power": float(torch.mean(csi.abs().pow(2)).item()),
        },
        diagnostics={"device": str(dev), "normalized_channel_power": 1.0},
        parameter_provenance=prov,
        mechanisms_used=[],
        standard_ids_used=list(META.standard_ids),
    )
