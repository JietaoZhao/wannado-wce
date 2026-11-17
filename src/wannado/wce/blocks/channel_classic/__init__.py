"""Classic fading & stochastic geometry blocks."""

from __future__ import annotations

from typing import Any

from wannado.wce.blocks import BLOCK_EXPORTS, lazy_resolve

_FAMILY_EXPORTS = {k: v for k, v in BLOCK_EXPORTS.items() if v[0].startswith("wannado.wce.blocks.channel_classic.")}
__all__ = [
    "AWGNConfig",
    "NakagamiConfig",
    "PPPCellularConfig",
    "RayleighClarkeConfig",
    "RicianConfig",
    "SalehValenzuelaConfig",
    "provision_awgn_link",
    "provision_nakagami_m",
    "provision_ppp_stochastic_geo",
    "provision_rayleigh_clarke",
    "provision_rician_los",
    "provision_saleh_valenzuela",
]


def __getattr__(name: str) -> Any:
    return lazy_resolve(name, _FAMILY_EXPORTS, globals(), __name__)


def __dir__() -> list[str]:
    return __all__
