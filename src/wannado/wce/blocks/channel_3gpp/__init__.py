"""3GPP TR 38.901 / TR 38.811 / TR 36.777 / TR 37.885 standardized channel blocks."""

from __future__ import annotations

from typing import Any

from wannado.wce.blocks import BLOCK_EXPORTS, lazy_resolve

_FAMILY_EXPORTS = {k: v for k, v in BLOCK_EXPORTS.items() if v[0].startswith("wannado.wce.blocks.channel_3gpp.")}
__all__ = sorted(_FAMILY_EXPORTS.keys())


def __getattr__(name: str) -> Any:
    return lazy_resolve(name, _FAMILY_EXPORTS, globals(), __name__)


def __dir__() -> list[str]:
    return __all__
