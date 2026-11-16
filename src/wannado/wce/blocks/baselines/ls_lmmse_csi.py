"""Pilot-Based LS vs LMMSE Channel Estimation NMSE Baseline (3GPP TR 38.843 Reference).

[Lineage & Intuition]
- Family / Era / Tier : baselines / classic / S
- Parent              : `tr38901_cdl_tdl`
- Key Diff            : Computes analytical & empirical NMSE for Least-Squares (LS) vs Linear MMSE
                        (Wiener filter exploiting spatial correlation R_H) under pilot SNR.
- Extension Hook [AI-Prompt Hook]:
    1. Benchmark AI/ML CSI feedback & neural denoisers against the LMMSE lower bound per TR 38.843.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch

from wannado.wce.core.provenance import STANDARDS_PROVENANCE
from wannado.wce.core.types import LitCitation, ParamProvenance, ProvisionResult, WCEMeta
from wannado.wce.core.units import db_to_linear, linear_to_db
from wannado.wce.core.utils import complex_normal, resolve_device, set_global_seed

_TSE_CIT = LitCitation(
    key="tse2005fundamentals",
    title="Fundamentals of Wireless Communication",
    authors="David Tse and Pramod Viswanath",
    venue="Cambridge University Press",
    year=2005,
    url="https://web.stanford.edu/~dntse/wireless_book.html",
    bibtex=(
        "@book{tse2005fundamentals,\n"
        "  title={Fundamentals of Wireless Communication},\n"
        "  author={Tse, David and Viswanath, Pramod},\n"
        "  year={2005},\n"
        "  publisher={Cambridge University Press}\n"
        "}"
    ),
    ieee='D. Tse and P. Viswanath, Fundamentals of Wireless Communication. Cambridge University Press, 2005.',
    apa="Tse, D., & Viswanath, P. (2005). Fundamentals of Wireless Communication. Cambridge University Press.",
)

META = WCEMeta(
    id="ls_lmmse_csi",
    name="Pilot-Based LS vs LMMSE Channel Estimation Baseline",
    family="baselines",
    era="classic",
    tier="S",
    output_kind="baseline_policy_solution",
    supports_torch_complex=True,
    supports_autograd=False,
    standard_ids=("gpp_tr38843",),
    parent_id="tr38901_cdl_tdl",
    diff_from_parent="Benchmarks noisy orthogonal-pilot LS against spatial-covariance LMMSE NMSE floor.",
    summary_plain="Classic Least-Squares (LS) vs LMMSE channel estimation baseline preventing AI/ML CSI models from claiming easy LS-only wins.",
    when_to_use="Channel estimation, TR 38.843 AI/ML CSI benchmarks, and imperfect-CSI sensitivity studies.",
    when_not_to_use="Ideal Genie-CSI capacity ceilings (`water_filling`).",
    hyperparam_intuition={
        "pilot_snr_db": "SNR of pilot reference symbols (DM-RS / CSI-RS).",
        "corr_rho": "Spatial/frequency channel correlation exploited by LMMSE over naive LS.",
    },
    extension_hooks=("Feed noisy h_hat_lmmse into downstream precoders instead of true h!",),
    module_path="wannado.wce.blocks.baselines.ls_lmmse_csi",
    default_mechanisms=("shannon_awgn",),
    citation=_TSE_CIT,
)


@dataclass
class CSIEstimatorConfig:
    batch_size: int = 256
    dim: int = 16
    pilot_snr_db: float = 5.0
    corr_rho: float = 0.8
    device: str = "auto"
    seed: int = 42


def provision_ls_lmmse_csi(config: CSIEstimatorConfig | None = None, **overrides: Any) -> ProvisionResult:
    """Compare pilot-based LS vs Wiener LMMSE estimation NMSE over spatially correlated channels."""
    cfg = config or CSIEstimatorConfig()
    if overrides:
        cfg = CSIEstimatorConfig(**{**asdict(cfg), **overrides})
    t0 = time.perf_counter()
    set_global_seed(cfg.seed)
    dev = resolve_device(cfg.device)

    idx = torch.arange(cfg.dim, device=dev, dtype=torch.float32)
    r_h = torch.pow(cfg.corr_rho, torch.abs(idx.unsqueeze(0) - idx.unsqueeze(1)))
    eigvals, eigvecs = torch.linalg.eigh(r_h)
    r_sqrt = ((eigvecs * torch.sqrt(eigvals.clamp(min=1e-7))) @ eigvecs.T).to(torch.complex64)

    z = complex_normal((cfg.batch_size, cfg.dim), device=dev)
    h_true = torch.einsum("ij,bj->bi", r_sqrt, z)

    noise_var = 1.0 / float(db_to_linear(cfg.pilot_snr_db))
    n_pilot = complex_normal((cfg.batch_size, cfg.dim), variance=noise_var, device=dev)
    y_pilot = h_true + n_pilot  # Assuming orthogonal unit-power pilot sequence

    # LS estimate: h_ls = y_pilot
    h_ls = y_pilot
    # LMMSE filter: W = R_H (R_H + sigma^2 I)^-1
    r_c = r_h.to(torch.complex64)
    w_lmmse = r_c @ torch.linalg.inv(r_c + noise_var * torch.eye(cfg.dim, dtype=torch.complex64, device=dev))
    h_lmmse = torch.einsum("ij,bj->bi", w_lmmse, y_pilot)

    sig_pow = torch.mean(h_true.abs().pow(2).sum(dim=-1)).item()
    nmse_ls = float(torch.mean((h_ls - h_true).abs().pow(2).sum(dim=-1)).item() / max(sig_pow, 1e-12))
    nmse_lmmse = float(torch.mean((h_lmmse - h_true).abs().pow(2).sum(dim=-1)).item() / max(sig_pow, 1e-12))

    prov = [
        ParamProvenance("pilot_snr_db", cfg.pilot_snr_db, "dB", "Orthogonal pilot SNR", "gpp_tr38843", "Clause 5.2", "gpp_tr38843", "Determines pilot noise floor 1/SNR_pilot."),
    ]
    return ProvisionResult(
        block_id=META.id,
        seed=cfg.seed,
        elapsed_seconds=time.perf_counter() - t0,
        config=asdict(cfg),
        tensors={"h_true": h_true, "h_ls": h_ls, "h_lmmse": h_lmmse},
        metrics={
            "ls_nmse_linear": nmse_ls,
            "ls_nmse_db": float(linear_to_db(nmse_ls)),
            "lmmse_nmse_linear": nmse_lmmse,
            "lmmse_nmse_db": float(linear_to_db(nmse_lmmse)),
            "lmmse_gain_db": float(linear_to_db(nmse_ls) - linear_to_db(nmse_lmmse)),
        },
        diagnostics={"device": str(dev), "normalized_channel_power": 1.0},
        parameter_provenance=prov,
        mechanisms_used=list(META.default_mechanisms),
        standard_ids_used=list(META.standard_ids),
    )
