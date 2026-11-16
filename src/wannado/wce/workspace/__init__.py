"""Non-intrusive ./wce_lab/ workspace scaffolding & single-file block ejection."""

from wannado.wce.workspace.lab_manager import (
    WirelessEnvInterface,
    eject_block,
    scaffold_wce_lab,
)

__all__ = [
    "WirelessEnvInterface",
    "eject_block",
    "scaffold_wce_lab",
]
