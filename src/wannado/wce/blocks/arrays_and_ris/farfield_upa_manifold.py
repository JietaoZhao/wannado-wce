"""Far-Field 2D Uniform Planar Array (UPA) Steering & Kronecker Spatial Correlation.

[Lineage & Intuition]
- Family / Era / Tier : arrays_and_ris / classic / S
- Parent              : `saleh_valenzuela`
- Key Diff            : Extends 1D ULA to 2D azimuth-elevation UPA manifold and exponential Kronecker correlation.
- Extension Hook [AI-Prompt Hook]:
    1. Switch to spherical wavefront when distance < Rayleigh distance -> `nearfield_xlmimo_thz`.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Any

import torch

from wannado.wce.core.geometry import upa_farfield_steering
from wannado.wce.core.provenance import MECHANISM_PROVENANCE
from wannado.wce.core.types import ParamProvenance, ProvisionResult, WCEMeta
from wannado.wce.core.units import rayleigh_distance_m, wavelength_m
from wannado.wce.core.utils import complex_normal, resolve_device, set_global_seed

META = WCEMeta(
    id="farfield_upa",
    name="Far-Field 2D UPA Massive MIMO Channel",
    family="arrays_and_ris",
    era="classic",
    tier="S",
    output_kind="complex_channel_tensor",
    supports_torch_complex=True,
    supports_autograd=True,
    standard_ids=("gpp_tr38901",),
    parent_id="saleh_valenzuela",
    diff_from_parent="Adds 2D azimuth-elevation planar array steering vectors and Kronecker correlation.",
    summary_plain="Standard 2D Uniform Planar Array (UPA) far-field massive MIMO channel with controllable spatial correlation.",
    when_to_use="3D beamforming (azimuth + elevation), FD-MIMO, and standard far-field massive MIMO benchmarks.",
    when_not_to_use="Close-range XL-MIMO / THz apertures operating inside the Rayleigh distance (`nearfield_xlmimo_thz`).",
    hyperparam_intuition={
        "tx_nx_ny": "Horizontal and vertical antenna counts at BS (e.g. (8, 8) = 64 elements).",
        "corr_rho": "Adjacent antenna spatial correlation coefficient [0, 0.99].",
    },
    extension_hooks=("Switch to nearfield_spherical_steering for XL-MIMO apertures",),
    module_path="wannado.wce.blocks.arrays_and_ris.farfield_upa_manifold",
    default_mechanisms=("saleh_valenzuela_cluster",),
    citation=MECHANISM_PROVENANCE["saleh_valenzuela_cluster"].citation,
)


@dataclass
class FarFieldUPAConfig:
    batch_size: int = 64
    tx_nx: int = 8
    tx_ny: int = 4
    num_rx: int = 4
    num_paths: int = 6
    fc_ghz: float = 28.0
    spacing_lambda: float = 0.5
    corr_rho: float = 0.5
    device: str = "auto"
    seed: int = 42


def _exp_corr_sqrt(n: int, rho: float, device: torch.device) -> torch.Tensor:
    idx = torch.arange(n, device=device, dtype=torch.float32)
    mat = torch.pow(max(0.0, min(0.99, rho)), torch.abs(idx.unsqueeze(0) - idx.unsqueeze(1)))
    eigvals, eigvecs = torch.linalg.eigh(mat)
    eigvals = torch.clamp(eigvals, min=1e-6)
    return (eigvecs * torch.sqrt(eigvals)) @ eigvecs.T


def provision_farfield_upa(config: FarFieldUPAConfig | None = None, **overrides: Any) -> ProvisionResult:
    """Provision far-field 2D UPA MIMO channel H[B, Nr, Nt_x*Nt_y] with Kronecker spatial correlation."""
    cfg = config or FarFieldUPAConfig()
    if overrides:
        cfg = FarFieldUPAConfig(**{**asdict(cfg), **overrides})
    t0 = time.perf_counter()
    set_global_seed(cfg.seed)
    dev = resolve_device(cfg.device)

    nt = cfg.tx_nx * cfg.tx_ny
    azim = (torch.rand((cfg.batch_size, cfg.num_paths), device=dev) - 0.5) * math.pi
    elev = torch.rand((cfg.batch_size, cfg.num_paths), device=dev) * (0.45 * math.pi) + 0.05 * math.pi
    a_tx = upa_farfield_steering(cfg.tx_nx, cfg.tx_ny, azim, elev, spacing_lambda=cfg.spacing_lambda)  # (B, L, Nt)

    gains = complex_normal((cfg.batch_size, cfg.num_rx, cfg.num_paths), variance=1.0, device=dev)
    h_raw = math.sqrt(nt / max(1, cfg.num_paths)) * torch.einsum("brl,bln->brn", gains, a_tx.conj())
    r_rx_sqrt = _exp_corr_sqrt(cfg.num_rx, cfg.corr_rho, dev).to(torch.complex64)
    h = torch.einsum("ij,bjn->bin", r_rx_sqrt, h_raw).to(torch.complex64)

    lam = wavelength_m(cfg.fc_ghz * 1e9)
    aperture_m = math.hypot((cfg.tx_nx - 1) * cfg.spacing_lambda * lam, (cfg.tx_ny - 1) * cfg.spacing_lambda * lam)
    zr_m = rayleigh_distance_m(aperture_m, cfg.fc_ghz * 1e9)

    prov = [
        ParamProvenance("tx_nx", cfg.tx_nx, "elements", "UPA horizontal antenna count", "gpp_tr38901", "Clause 7.3", "3gpp_tr38901", f"Array aperture D={aperture_m:.3f} m -> Z_R={zr_m:.2f} m."),
        ParamProvenance("corr_rho", cfg.corr_rho, "ratio", "Receive spatial correlation", "saleh_valenzuela_cluster", "IEEE JSAC 1987", "saleh1987statistical", "Exponential Kronecker spatial correlation across Rx elements."),
    ]
    return ProvisionResult(
        block_id=META.id,
        seed=cfg.seed,
        elapsed_seconds=time.perf_counter() - t0,
        config=asdict(cfg),
        tensors={"h": h},
        metrics={
            "num_tx": float(nt),
            "aperture_m": aperture_m,
            "rayleigh_distance_m": zr_m,
            "mean_frobenius_sq": float(torch.mean(h.abs().pow(2).sum(dim=(-2, -1))).item()),
            "mean_channel_power": float(torch.mean(h.abs().pow(2)).item()) / max(1, 1),
        },
        diagnostics={"device": str(dev), "normalized_channel_power": 1.0, "rayleigh_distance_m": zr_m},
        parameter_provenance=prov,
        mechanisms_used=list(META.default_mechanisms),
        standard_ids_used=list(META.standard_ids),
    )
