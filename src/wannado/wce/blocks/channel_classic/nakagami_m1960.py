"""Nakagami-m Generalized Multipath Fading Channel (Nakagami, 1960).

[Lineage & Intuition]
- Family / Era / Tier : channel_classic / classic / A
- Parent              : `rayleigh_clarke`
- Key Diff            : Parameterizes fading severity continuously via shape m >= 0.5
                        (m=1 reduces to Rayleigh; m>1 approximates Rician LoS; m<1 captures severe fading).
- Extension Hook [AI-Prompt Hook]:
    1. Pair with eavesdropper fading links for physical-layer secrecy outage analysis -> `pls_covert_antijamming`.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch

from wannado.wce.core.types import LitCitation, ParamProvenance, ProvisionResult, WCEMeta
from wannado.wce.core.utils import resolve_device, set_global_seed

_NAKAGAMI_CIT = LitCitation(
    key="nakagami1960mdistribution",
    title="The m-Distribution: A General Formula of Intensity Distribution of Rapid Fading",
    authors="Minoru Nakagami",
    venue="Statistical Methods in Radio Wave Propagation (Pergamon Press)",
    year=1960,
    url="https://doi.org/10.1016/B978-0-08-009306-2.50005-4",
    bibtex=(
        "@incollection{nakagami1960mdistribution,\n"
        "  title={The $m$-Distribution---A General Formula of Intensity Distribution of Rapid Fading},\n"
        "  author={Nakagami, Minoru},\n"
        "  booktitle={Statistical Methods in Radio Wave Propagation},\n"
        "  publisher={Pergamon Press},\n"
        "  pages={3--36},\n"
        "  year={1960},\n"
        "  doi={10.1016/B978-0-08-009306-2.50005-4}\n"
        "}"
    ),
    ieee='M. Nakagami, "The m-distribution—A general formula of intensity distribution of rapid fading," in Statistical Methods in Radio Wave Propagation, Pergamon Press, 1960, pp. 3-36.',
    apa="Nakagami, M. (1960). The m-distribution—A general formula of intensity distribution of rapid fading. Statistical Methods in Radio Wave Propagation, 3-36.",
)

META = WCEMeta(
    id="nakagami_m",
    name="Nakagami-m Generalized Fading Channel",
    family="channel_classic",
    era="classic",
    tier="A",
    output_kind="complex_channel_tensor",
    supports_torch_complex=True,
    supports_autograd=False,
    standard_ids=(),
    parent_id="rayleigh_clarke",
    diff_from_parent="Generalizes fading severity via Gamma-distributed power |h|^2 ~ Gamma(m, Omega/m).",
    summary_plain="Flexible fading channel where parameter m >= 0.5 smoothly sweeps from severe fading (m<1) through Rayleigh (m=1) to LoS (m>1).",
    when_to_use="Analytical fading-severity sweeps, vehicular/energy-harvesting/PLS theoretical validation.",
    when_not_to_use="Frequency-selective wideband MIMO spatial channel simulation (`tr38901_cdl_tdl`).",
    hyperparam_intuition={
        "m_shape": "Fading severity parameter (>= 0.5); larger m means milder fading and more stable SNR.",
        "omega": "Mean squared fading power E[|h|^2], normally 1.0.",
    },
    extension_hooks=("Use m=(K+1)^2/(2K+1) to cross-check Rician approximations",),
    module_path="wannado.wce.blocks.channel_classic.nakagami_m1960",
    default_mechanisms=("shannon_awgn",),
    citation=_NAKAGAMI_CIT,
)


@dataclass
class NakagamiConfig:
    batch_size: int = 1024
    num_links: int = 8
    m_shape: float = 2.5
    omega: float = 1.0
    device: str = "auto"
    seed: int = 42


def provision_nakagami_m(config: NakagamiConfig | None = None, **overrides: Any) -> ProvisionResult:
    """Sample exact Nakagami-m complex baseband coefficients h = sqrt(X) * exp(j*phi) where X ~ Gamma(m, omega/m)."""
    cfg = config or NakagamiConfig()
    if overrides:
        cfg = NakagamiConfig(**{**asdict(cfg), **overrides})
    if cfg.m_shape < 0.5:
        raise ValueError(f"Nakagami m_shape must be >= 0.5, got {cfg.m_shape}.")
    t0 = time.perf_counter()
    rng = set_global_seed(cfg.seed)
    dev = resolve_device(cfg.device)

    power_np = rng.gamma(shape=cfg.m_shape, scale=cfg.omega / cfg.m_shape, size=(cfg.batch_size, cfg.num_links))
    amp = torch.from_numpy(np.sqrt(power_np)).to(device=dev, dtype=torch.float32)
    phase = torch.rand((cfg.batch_size, cfg.num_links), device=dev) * (2.0 * math.pi)
    h = torch.complex(amp * torch.cos(phase), amp * torch.sin(phase)).to(torch.complex64)

    # Amount of fading AF = Var(|h|^2) / (E[|h|^2])^2; theoretical AF = 1 / m
    p_abs = h.abs().pow(2)
    emp_af = float((torch.var(p_abs, unbiased=False) / (torch.mean(p_abs).pow(2).clamp(min=1e-12))).item())

    prov = [
        ParamProvenance("m_shape", cfg.m_shape, "dimensionless", "Nakagami fading severity m >= 0.5", "shannon_awgn", "Nakagami 1960", "nakagami1960mdistribution", f"Theoretical Amount of Fading AF = 1/m = {1.0/cfg.m_shape:.3f}."),
    ]
    return ProvisionResult(
        block_id=META.id,
        seed=cfg.seed,
        elapsed_seconds=time.perf_counter() - t0,
        config=asdict(cfg),
        tensors={"h": h},
        metrics={
            "mean_channel_power": float(torch.mean(p_abs).item()),
            "empirical_amount_of_fading": emp_af,
            "theoretical_amount_of_fading": 1.0 / cfg.m_shape,
        },
        diagnostics={"device": str(dev), "normalized_channel_power": cfg.omega},
        parameter_provenance=prov,
        mechanisms_used=list(META.default_mechanisms),
        standard_ids_used=[],
    )
