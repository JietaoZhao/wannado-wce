"""Antenna array, near-field XL-MIMO, RIS/STAR-RIS, and flexible antenna blocks."""

from __future__ import annotations

from typing import Any

from wannado.wce.blocks import BLOCK_EXPORTS, lazy_resolve

_FAMILY_EXPORTS = {k: v for k, v in BLOCK_EXPORTS.items() if v[0].startswith("wannado.wce.blocks.arrays_and_ris.")}
__all__ = [
    "ActiveRISFlexibleConfig",
    "FarFieldUPAConfig",
    "NearFieldXLMIMOConfig",
    "PassiveRISConfig",
    "STARRISConfig",
    "provision_active_ris_flexible",
    "provision_farfield_upa",
    "provision_nearfield_xlmimo_thz",
    "provision_passive_ris_cascade",
    "provision_star_ris",
]


def __getattr__(name: str) -> Any:
    return lazy_resolve(name, _FAMILY_EXPORTS, globals(), __name__)


def __dir__() -> list[str]:
    return __all__
