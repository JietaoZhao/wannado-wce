"""IMT-2030 / 6G frontier ISAC, OTFS, HRLLC-FBL, Ambient IoT, Semantic/AirComp, and Security blocks."""

from __future__ import annotations

from typing import Any

from wannado.wce.blocks import BLOCK_EXPORTS, lazy_resolve

_FAMILY_EXPORTS = {k: v for k, v in BLOCK_EXPORTS.items() if v[0].startswith("wannado.wce.blocks.domain_6g.")}
__all__ = [
    "AmbientIoTConfig",
    "HRLLCFBLConfig",
    "ISACConfig",
    "OTFSConfig",
    "SemanticAirCompConfig",
    "WiFiSensingConfig",
    "WirelessSecurityConfig",
    "provision_ambient_iot_backscatter",
    "provision_hrllc_fbl_urllc",
    "provision_isac_jsac_channel",
    "provision_otfs_delay_doppler",
    "provision_pls_covert_antijamming",
    "provision_semantic_aircomp_channel",
    "provision_wifi_sensing_80211bf",
]


def __getattr__(name: str) -> Any:
    return lazy_resolve(name, _FAMILY_EXPORTS, globals(), __name__)


def __dir__() -> list[str]:
    return __all__
