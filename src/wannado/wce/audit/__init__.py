"""Stage 1 Preflight, Stage 3 Wireless Physical Rigor Auditing, and One-Command Verify Gate."""

from __future__ import annotations

import importlib
from typing import Any

from wannado.wce.audit.preflight import PreflightReport, preflight_wce
from wannado.wce.audit.rigor_auditor import RigorAuditReport, audit_wce_rigor

_LAZY_AUDIT_EXPORTS: dict[str, tuple[str, str]] = {
    "WCEVerifyReport": ("wannado.wce.audit.verify_gate", "WCEVerifyReport"),
    "verify_wce_target": ("wannado.wce.audit.verify_gate", "verify_wce_target"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_AUDIT_EXPORTS:
        mod_path, attr = _LAZY_AUDIT_EXPORTS[name]
        val = getattr(importlib.import_module(mod_path), attr)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "PreflightReport",
    "RigorAuditReport",
    "WCEVerifyReport",
    "audit_wce_rigor",
    "preflight_wce",
    "verify_wce_target",
]
