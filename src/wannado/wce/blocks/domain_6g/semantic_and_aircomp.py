"""AIAC Differentiable Semantic Baseband & AirComp Cell-Free Aggregation (Xie 2021; Zhu 2020; Ngo 2017).

[Lineage & Intuition]
- Family / Era / Tier : domain_6g / frontier / S (IMT-2030 AIAC Native AI & Communication Pillar)
- Parent              : `rayleigh_clarke`
- Key Diff            : Provides both (1) differentiable PyTorch semantic-coding baseband channel layers
                        for DeepSC-style JSCC and (2) over-the-air analog superposition AirComp over
                        distributed Cell-Free Massive MIMO access points.
- Extension Hook [AI-Prompt Hook]:
    1. Wrap `result.extra['channel_module']` directly inside end-to-end semantic autoencoders or PyTorch FL.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Any

import torch

from wannado.wce.core.adapters import ChannelModuleAdapter
from wannado.wce.core.provenance import MECHANISM_PROVENANCE
from wannado.wce.core.types import ParamProvenance, ProvisionResult, WCEMeta
from wannado.wce.core.units import db_to_linear
from wannado.wce.core.utils import complex_normal, resolve_device, set_global_seed

META = WCEMeta(
    id="semantic_aircomp_channel",
    name="Semantic Baseband & AirComp Cell-Free Edge AI Channel",
    family="domain_6g",
    era="frontier",
    tier="S",
    output_kind="complex_channel_tensor",
    supports_torch_complex=True,
    supports_autograd=True,
    standard_ids=("gpp_tr38843", "itur_m2160"),
    parent_id="rayleigh_clarke",
    diff_from_parent="Exposes differentiable PyTorch ChannelModuleAdapter plus multi-device analog AirComp aggregation MSE.",
    summary_plain="IMT-2030 AI-and-Communication (AIAC) environment supporting differentiable DeepSC semantic links and AirComp federated aggregation.",
    when_to_use="End-to-end semantic JSCC training, over-the-air federated edge learning (AirComp), and Cell-Free macro-diversity.",
    when_not_to_use="Standard bit-level cellular traffic queueing (`multicell_interference_mac`).",
    hyperparam_intuition={
        "snr_db": "Channel SNR in dB passed through differentiable semantic channel layer.",
        "num_edge_devices": "Devices simultaneously transmitting gradients/embeddings in AirComp.",
        "num_cellfree_aps": "Distributed Cell-Free APs cooperatively receiving AirComp waveforms.",
    },
    extension_hooks=("Attach PyTorch Transformer semantic encoder/decoder to extra['channel_module']",),
    module_path="wannado.wce.blocks.domain_6g.semantic_and_aircomp",
    default_mechanisms=("xie_deepsc_semantic", "zhu_aircomp_feel", "ngo_cell_free_mimo"),
    citation=MECHANISM_PROVENANCE["zhu_aircomp_feel"].citation,
)


@dataclass
class SemanticAirCompConfig:
    batch_size: int = 64
    latent_dim: int = 32
    snr_db: float = 12.0
    rician_k_db: float = 3.0
    num_edge_devices: int = 12
    num_cellfree_aps: int = 8
    aircomp_cutoff_gamma_th: float = 0.2
    device: str = "auto"
    seed: int = 42


def provision_semantic_aircomp_channel(config: SemanticAirCompConfig | None = None, **overrides: Any) -> ProvisionResult:
    """Provision differentiable semantic channel module & compute Cell-Free AirComp gradient aggregation MSE."""
    cfg = config or SemanticAirCompConfig()
    if overrides:
        cfg = SemanticAirCompConfig(**{**asdict(cfg), **overrides})
    t0 = time.perf_counter()
    set_global_seed(cfg.seed)
    dev = resolve_device(cfg.device)

    k_lin = float(db_to_linear(cfg.rician_k_db))
    w_los = math.sqrt(k_lin / (k_lin + 1.0))
    w_nlos = math.sqrt(1.0 / (k_lin + 1.0))

    def _sample_semantic_h(bs: int) -> torch.Tensor:
        return (w_los + w_nlos * complex_normal((bs, cfg.latent_dim), device=dev)).to(torch.complex64)

    chan_mod = ChannelModuleAdapter(_sample_semantic_h, snr_db=cfg.snr_db, add_noise=True)

    # Differentiability verification on semantic latent vector x
    x_latent = complex_normal((cfg.batch_size, cfg.latent_dim), device=dev).requires_grad_(True)
    y_rx = chan_mod(x_latent)
    loss = torch.mean((y_rx - x_latent).abs().pow(2))
    loss.backward()
    grad_ok = x_latent.grad is not None and torch.isfinite(x_latent.grad).all().item()

    # Cell-Free AirComp: K devices, M distributed APs with pathloss beta_{m,k}
    ap_xy = torch.rand((cfg.num_cellfree_aps, 2), device=dev) * 200.0
    dev_xy = torch.rand((cfg.num_edge_devices, 2), device=dev) * 200.0
    dists = torch.cdist(ap_xy, dev_xy).clamp(min=10.0)
    beta = (dists / 10.0).pow(-3.0)  # (M, K)
    h_cf = complex_normal((cfg.batch_size, cfg.num_cellfree_aps, cfg.num_edge_devices), device=dev) * torch.sqrt(beta).unsqueeze(0)
    # Coherent MRC combining across APs -> effective scalar channel gain per device
    eff_gain = torch.sum(h_cf.abs().pow(2), dim=1)  # (B, K)
    # Truncated channel inversion AirComp MSE (Zhu et al., IEEE TWC 2020):
    # Active devices (g_k >= gamma_th) invert to common alignment threshold sqrt(gamma_th), so receiver
    # thermal noise variance scales inversely with gamma_th (not mean(eff_gain)).
    snr_lin = float(db_to_linear(cfg.snr_db))
    participating = (eff_gain >= cfg.aircomp_cutoff_gamma_th).float()
    k_active = torch.clamp(torch.sum(participating, dim=-1), min=1.0)
    agg_noise_mse = 1.0 / (snr_lin * max(float(cfg.aircomp_cutoff_gamma_th), 1e-6) * k_active)
    dropout_mse = torch.mean(1.0 - participating, dim=-1)
    aircomp_mse = float(torch.mean(agg_noise_mse + dropout_mse).item())

    prov = [
        ParamProvenance("snr_db", cfg.snr_db, "dB", "Baseband semantic/AirComp SNR", "xie_deepsc_semantic", "IEEE TSP 2021", "xie2021deep", "Drives differentiable latent distortion & AirComp aggregation MSE."),
        ParamProvenance("num_cellfree_aps", cfg.num_cellfree_aps, "APs", "Distributed Cell-Free APs", "ngo_cell_free_mimo", "IEEE TWC 2017", "ngo2017cellfree", "Macro-diversity overcomes deep single-cell AirComp channel truncation."),
    ]
    return ProvisionResult(
        block_id=META.id,
        seed=cfg.seed,
        elapsed_seconds=time.perf_counter() - t0,
        config=asdict(cfg),
        tensors={"y_semantic_sample": y_rx.detach(), "h_cellfree": h_cf},
        metrics={
            "semantic_channel_mse": float(loss.item()),
            "autograd_gradient_valid": float(grad_ok),
            "aircomp_aggregation_mse": aircomp_mse,
            "aircomp_participation_rate": float(torch.mean(participating).item()),
        },
        diagnostics={"device": str(dev), "normalized_channel_power": 1.0, "autograd_connected": bool(grad_ok)},
        parameter_provenance=prov,
        mechanisms_used=list(META.default_mechanisms),
        standard_ids_used=list(META.standard_ids),
        extra={"channel_module": chan_mod},
    )
