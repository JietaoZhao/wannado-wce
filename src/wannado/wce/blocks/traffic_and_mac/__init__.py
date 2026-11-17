"""3GPP traffic generators & multi-cell MAC queue environments."""

from __future__ import annotations

from typing import Any

from wannado.wce.blocks import BLOCK_EXPORTS, lazy_resolve

_FAMILY_EXPORTS = {k: v for k, v in BLOCK_EXPORTS.items() if v[0].startswith("wannado.wce.blocks.traffic_and_mac.")}
__all__ = [
    "FTPTrafficConfig",
    "MixedTrafficConfig",
    "MultiCellMACConfig",
    "XRTrafficConfig",
    "provision_multicell_interference_mac",
    "provision_traffic_ftp_models",
    "provision_traffic_poisson_pareto_voip",
    "provision_traffic_xr_immersive",
]


def __getattr__(name: str) -> Any:
    return lazy_resolve(name, _FAMILY_EXPORTS, globals(), __name__)


def __dir__() -> list[str]:
    return __all__
