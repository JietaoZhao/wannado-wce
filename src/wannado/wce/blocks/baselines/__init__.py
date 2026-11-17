"""Analytic wireless baselines & environment-aware CKM/RadioMap bridges."""

from __future__ import annotations

from typing import Any

from wannado.wce.blocks import BLOCK_EXPORTS, lazy_resolve

_FAMILY_EXPORTS = {k: v for k, v in BLOCK_EXPORTS.items() if v[0].startswith("wannado.wce.blocks.baselines.")}
__all__ = [
    "CKMRadioMapConfig",
    "CSIEstimatorConfig",
    "PFSchedulerConfig",
    "WMMSEConfig",
    "WaterFillingConfig",
    "provision_ckm_radiomap_bridge",
    "provision_ls_lmmse_csi",
    "provision_proportional_fair",
    "provision_water_filling",
    "provision_wmmse_precoder",
]


def __getattr__(name: str) -> Any:
    return lazy_resolve(name, _FAMILY_EXPORTS, globals(), __name__)


def __dir__() -> list[str]:
    return __all__
