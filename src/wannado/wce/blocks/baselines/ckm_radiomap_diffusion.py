"""Environment-Aware Channel Knowledge Map / Diffusion Radio Map Bridge (Zeng 2021; Zhao 2023; Fu 2025).

[Lineage & Intuition]
- Family / Era / Tier : baselines / frontier / S (2021-2025 Environment-Aware CKM & GenAI Reconstruction)
- Parent              : `tr38901_uma_umi_rma`
- Key Diff            : Synthesizes a spatially correlated 2D shadowing/blockage urban radio map ground truth,
                        samples sparse noisy measurements, and benchmarks location-aware CKM/conditional
                        score-denoising inpainting against blind location-unaware statistical averaging.
- Extension Hook [AI-Prompt Hook]:
    1. Replace built-in RBF/score denoiser with full UNet diffusion (`CKMDiff`) or `DeepMIMO`/`Sionna RT` grids!
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch

from wannado.wce.core.provenance import MECHANISM_PROVENANCE
from wannado.wce.core.types import ParamProvenance, ProvisionResult, WCEMeta
from wannado.wce.core.utils import set_global_seed

META = WCEMeta(
    id="ckm_radiomap_bridge",
    name="Environment-Aware CKM & Conditional Score Radio Map Reconstruction",
    family="baselines",
    era="frontier",
    tier="S",
    output_kind="spatial_channel_map",
    supports_torch_complex=False,
    supports_autograd=False,
    standard_ids=("itur_m2160", "gpp_tr38901"),
    parent_id="tr38901_uma_umi_rma",
    diff_from_parent="Reconstructs site-specific building-shadowed Channel Knowledge Maps from sparse noisy field pilots.",
    summary_plain="Environment-aware Channel Knowledge Map (CKM) & Radio Map benchmark bridging CKM, NeRF2, and CKMDiff concepts.",
    when_to_use="Environment-aware 6G beamforming, radio map inpainting, and DeepMIMO/Sionna-RT surrogate testing.",
    when_not_to_use="Pure stochastic single-link fading (`rayleigh_clarke`).",
    hyperparam_intuition={
        "grid_size": "2D urban map resolution (grid_size x grid_size).",
        "sample_fraction": "Fraction of map grid locations measured by sparse drive-test pilots.",
    },
    extension_hooks=("Connect to DeepMIMO / Sionna RT exported path-gain grids via ./wce_lab/interface.py",),
    module_path="wannado.wce.blocks.baselines.ckm_radiomap_diffusion",
    default_mechanisms=("zeng_ckm_prior", "fu_ckmdiff_diffusion", "zhao_nerf2_rf"),
    citation=MECHANISM_PROVENANCE["fu_ckmdiff_diffusion"].citation,
)


@dataclass
class CKMRadioMapConfig:
    grid_size: int = 32
    area_width_m: float = 200.0
    fc_ghz: float = 3.5
    sample_fraction: float = 0.08
    measurement_noise_db: float = 2.0
    diffusion_refine_steps: int = 15
    seed: int = 42


def provision_ckm_radiomap_bridge(config: CKMRadioMapConfig | None = None, **overrides: Any) -> ProvisionResult:
    """Synthesize building-shadowed 2D urban radio map & benchmark sparse CKM inpainting vs statistical blind baseline."""
    cfg = config or CKMRadioMapConfig()
    if overrides:
        cfg = CKMRadioMapConfig(**{**asdict(cfg), **overrides})
    t0 = time.perf_counter()
    rng = set_global_seed(cfg.seed)

    g = cfg.grid_size
    xs = np.linspace(-0.5 * cfg.area_width_m, 0.5 * cfg.area_width_m, g)
    ys = np.linspace(-0.5 * cfg.area_width_m, 0.5 * cfg.area_width_m, g)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    dists = np.maximum(np.hypot(gx, gy), 5.0)

    # Base statistical TR 38.901 UMi-like path gain (dB) + localized deterministic building shadow blocks
    base_gain_db = -(32.4 + 21.0 * np.log10(dists) + 20.0 * math.log10(cfg.fc_ghz))
    shadow_map = np.zeros_like(base_gain_db)
    for _ in range(4):
        cx, cy = rng.uniform(-0.35 * cfg.area_width_m, 0.35 * cfg.area_width_m, size=2)
        bw = rng.uniform(15.0, 35.0)
        # Directional shadow behind building away from origin BS
        in_box = (np.abs(gx - cx) < bw) & (np.abs(gy - cy) < bw)
        shadow_map[in_box] -= 18.0
    # Add spatially correlated Gaussian random field smoothing
    noise_field = rng.normal(0.0, 4.0, size=(g, g))
    true_map = base_gain_db + shadow_map + 0.5 * (
        noise_field + np.roll(noise_field, 1, axis=0) + np.roll(noise_field, 1, axis=1)
    ) / 1.5

    # Sparse noisy field measurements
    n_tot = g * g
    n_obs = max(4, int(round(cfg.sample_fraction * n_tot)))
    obs_idx = rng.choice(n_tot, size=n_obs, replace=False)
    obs_mask = np.zeros(n_tot, dtype=bool)
    obs_mask[obs_idx] = True
    obs_mask_2d = obs_mask.reshape(g, g)
    obs_vals = true_map.reshape(-1)[obs_idx] + rng.normal(0.0, cfg.measurement_noise_db, size=n_obs)

    # 1. Blind statistical baseline (ignores local building geometry, uses only distance-based pathloss)
    rmse_blind = float(np.sqrt(np.mean((base_gain_db - true_map) ** 2)))

    # 2. Physics-Informed CKM + Conditional Score/Diffusion Prior Inpainting:
    # Interpolate residual (obs - base_gain_db) via RBF kernel + iterative score-prior smoothing
    obs_coords = np.stack([gx.reshape(-1)[obs_idx], gy.reshape(-1)[obs_idx]], axis=-1)
    all_coords = np.stack([gx.reshape(-1), gy.reshape(-1)], axis=-1)
    res_obs = obs_vals - base_gain_db.reshape(-1)[obs_idx]
    ell = max(12.0, cfg.area_width_m / 8.0)
    d2 = np.sum((all_coords[:, None, :] - obs_coords[None, :, :]) ** 2, axis=-1)
    w_rbf = np.exp(-d2 / (2.0 * ell * ell))
    w_sum = np.maximum(np.sum(w_rbf, axis=1), 1e-12)
    res_pred = (w_rbf @ res_obs) / w_sum
    ckm_map = (base_gain_db.reshape(-1) + res_pred).reshape(g, g)

    # Conditional score/diffusion-inspired local smoothness + measurement consistency projection
    obs_grid = np.zeros_like(true_map)
    obs_grid.reshape(-1)[obs_idx] = obs_vals
    for _ in range(cfg.diffusion_refine_steps):
        lap = 0.25 * (
            np.roll(ckm_map, 1, axis=0)
            + np.roll(ckm_map, -1, axis=0)
            + np.roll(ckm_map, 1, axis=1)
            + np.roll(ckm_map, -1, axis=1)
        )
        ckm_map = 0.7 * ckm_map + 0.3 * lap
        # Hard measurement data-consistency anchor (CKMDiff style inverse conditioning)
        ckm_map[obs_mask_2d] = 0.8 * obs_grid[obs_mask_2d] + 0.2 * ckm_map[obs_mask_2d]

    rmse_ckm = float(np.sqrt(np.mean((ckm_map - true_map) ** 2)))

    prov = [
        ParamProvenance("sample_fraction", cfg.sample_fraction, "ratio", "Sparse measurement sampling ratio", "fu_ckmdiff_diffusion", "arXiv 2504.17323", "fu2025ckmdiff", "Tests inverse radio-map reconstruction from sparse noisy measurements."),
    ]
    return ProvisionResult(
        block_id=META.id,
        seed=cfg.seed,
        elapsed_seconds=time.perf_counter() - t0,
        config=asdict(cfg),
        tensors={"true_radio_map_db": torch.from_numpy(true_map).to(torch.float32), "reconstructed_ckm_db": torch.from_numpy(ckm_map).to(torch.float32)},
        metrics={
            "ckm_reconstruction_rmse_db": rmse_ckm,
            "blind_statistical_rmse_db": rmse_blind,
            "ckm_rmse_improvement_db": rmse_blind - rmse_ckm,
            "observed_grid_points": float(n_obs),
        },
        diagnostics={"normalized_channel_power": 1.0},
        parameter_provenance=prov,
        mechanisms_used=list(META.default_mechanisms),
        standard_ids_used=list(META.standard_ids),
    )
