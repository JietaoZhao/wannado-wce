"""Saleh-Valenzuela Clustered Sparse mmWave/THz Geometric MIMO Channel (Saleh & Valenzuela, IEEE JSAC 1987).

[Lineage & Intuition]
- Family / Era / Tier : channel_classic / classic / S (Foundation for mmWave/THz & RIS geometry)
- Parent              : `rician_los`
- Key Diff            : Organizes multipath arrivals into sparse angular clusters with ULA/UPA array steering
                        rather than i.i.d. rich-scattering MIMO matrices.
- Extension Hook [AI-Prompt Hook]:
    1. Replace far-field steering with near-field spherical steering -> `nearfield_xlmimo_thz`.
    2. Insert two-hop RIS reflection matrix -> `passive_ris_cascade`.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Any

import torch

from wannado.wce.core.geometry import ula_farfield_steering
from wannado.wce.core.provenance import MECHANISM_PROVENANCE
from wannado.wce.core.types import ParamProvenance, ProvisionResult, WCEMeta
from wannado.wce.core.utils import complex_normal, resolve_device, set_global_seed

META = WCEMeta(
    id="saleh_valenzuela",
    name="Saleh-Valenzuela Clustered Sparse MIMO Channel",
    family="channel_classic",
    era="classic",
    tier="S",
    output_kind="complex_channel_tensor",
    supports_torch_complex=True,
    supports_autograd=True,
    standard_ids=("gpp_tr38901",),
    parent_id="rician_los",
    diff_from_parent="Constructs spatially sparse multi-cluster MIMO H = sqrt(Nt*Nr/L) sum alpha * a_r(theta) a_t^H(phi).",
    summary_plain="Canonical mmWave/THz sparse clustered geometric MIMO channel with normalized Frobenius power E[||H||_F^2] = Nt * Nr.",
    when_to_use="mmWave hybrid beamforming, beam alignment, compressed CSI estimation, and sparse MIMO baselines.",
    when_not_to_use="Near-field XL-MIMO where distance < Rayleigh distance (`nearfield_xlmimo_thz`).",
    hyperparam_intuition={
        "num_clusters": "Number of dominant environmental scattering clusters (typically 2-6 at mmWave).",
        "rays_per_cluster": "Sub-paths around each cluster mean angle producing angular spread.",
        "angle_spread_deg": "Laplacian/Gaussian angular spread within each cluster.",
    },
    extension_hooks=(
        "Swap ula_farfield_steering for nearfield_spherical_steering -> `nearfield_xlmimo_thz`",
    ),
    module_path="wannado.wce.blocks.channel_classic.saleh_valenzuela1987",
    default_mechanisms=("saleh_valenzuela_cluster",),
    citation=MECHANISM_PROVENANCE["saleh_valenzuela_cluster"].citation,
)


@dataclass
class SalehValenzuelaConfig:
    batch_size: int = 128
    num_tx: int = 16
    num_rx: int = 4
    num_clusters: int = 4
    rays_per_cluster: int = 5
    angle_spread_deg: float = 7.5
    spacing_lambda: float = 0.5
    device: str = "auto"
    seed: int = 42


def provision_saleh_valenzuela(config: SalehValenzuelaConfig | None = None, **overrides: Any) -> ProvisionResult:
    """Provision normalized clustered MIMO channel H[B, Nr, Nt] with E[||H||_F^2] = Nr * Nt."""
    cfg = config or SalehValenzuelaConfig()
    if overrides:
        cfg = SalehValenzuelaConfig(**{**asdict(cfg), **overrides})
    t0 = time.perf_counter()
    set_global_seed(cfg.seed)
    dev = resolve_device(cfg.device)

    total_paths = cfg.num_clusters * cfg.rays_per_cluster
    spread_rad = math.radians(cfg.angle_spread_deg)

    # Cluster center angles in [-pi/3, pi/3], plus ray offsets
    mean_aod = (torch.rand((cfg.batch_size, cfg.num_clusters, 1), device=dev) - 0.5) * (2.0 * math.pi / 3.0)
    mean_aoa = (torch.rand((cfg.batch_size, cfg.num_clusters, 1), device=dev) - 0.5) * (2.0 * math.pi / 3.0)
    aod = (mean_aod + torch.randn((cfg.batch_size, cfg.num_clusters, cfg.rays_per_cluster), device=dev) * spread_rad).reshape(cfg.batch_size, total_paths)
    aoa = (mean_aoa + torch.randn((cfg.batch_size, cfg.num_clusters, cfg.rays_per_cluster), device=dev) * spread_rad).reshape(cfg.batch_size, total_paths)

    alpha = complex_normal((cfg.batch_size, total_paths), variance=1.0, device=dev)
    a_tx = ula_farfield_steering(cfg.num_tx, aod, spacing_lambda=cfg.spacing_lambda)  # (B, L, Nt)
    a_rx = ula_farfield_steering(cfg.num_rx, aoa, spacing_lambda=cfg.spacing_lambda)  # (B, L, Nr)

    # Steering vectors are unit-norm, so outer product ||a_rx a_tx^H||_F = 1; scale by sqrt(Nt*Nr / L)
    scale = math.sqrt((cfg.num_tx * cfg.num_rx) / max(1, total_paths))
    weighted_rx = a_rx * alpha.unsqueeze(-1)
    h = scale * torch.einsum("bln,blm->bnm", weighted_rx, a_tx.conj()).to(torch.complex64)

    mean_frob_sq = float(torch.mean(h.abs().pow(2).sum(dim=(-2, -1))).item())
    mean_entry_pow = mean_frob_sq / max(1, cfg.num_tx * cfg.num_rx)

    prov = [
        ParamProvenance("num_clusters", cfg.num_clusters, "clusters", "Sparse multipath scattering clusters", "saleh_valenzuela_cluster", "IEEE JSAC 1987", "saleh1987statistical", "Models limited mmWave/THz angular scattering."),
        ParamProvenance("angle_spread_deg", cfg.angle_spread_deg, "deg", "Intra-cluster angular spread", "gpp_tr38901", "Clause 7.5", "3gpp_tr38901", "Determines rank and spatial correlation of MIMO channel."),
    ]
    return ProvisionResult(
        block_id=META.id,
        seed=cfg.seed,
        elapsed_seconds=time.perf_counter() - t0,
        config=asdict(cfg),
        tensors={"h": h, "aod_rad": aod, "aoa_rad": aoa},
        metrics={
            "mean_frobenius_sq": mean_frob_sq,
            "target_frobenius_sq": float(cfg.num_tx * cfg.num_rx),
            "mean_channel_power": mean_entry_pow,
            "total_multipath_rays": float(total_paths),
        },
        diagnostics={"device": str(dev), "normalized_channel_power": 1.0},
        parameter_provenance=prov,
        mechanisms_used=list(META.default_mechanisms),
        standard_ids_used=list(META.standard_ids),
    )
