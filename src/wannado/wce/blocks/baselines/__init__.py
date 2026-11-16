"""Analytic wireless baselines & environment-aware CKM/RadioMap bridges."""

from __future__ import annotations

from typing import Any

from wannado.wce.blocks import BLOCK_EXPORTS, lazy_resolve

_FAMILY_EXPORTS = {k: v for k, v in BLOCK_EXPORTS.items() if v[0].startswith("wannado.wce.blocks.baselines.")}
__all__ = sorted(_FAMILY_EXPORTS.keys())


def __getattr__(name: str) -> Any:
    return lazy_resolve(name, _FAMILY_EXPORTS, globals(), __name__)


def __dir__() -> list[str]:
    return __all__
