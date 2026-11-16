"""CleanRL-style single-file wireless environment blocks across 6 authoritative families.

Supports PEP 562 lazy imports such as:
    from wannado.wce.blocks import provision_tr38901_uma_umi_rma, provision_isac_jsac_channel
"""

from __future__ import annotations

import importlib
from typing import Any

BLOCK_EXPORTS: dict[str, tuple[str, str]] = {
    # 1. Classic physics & stochastic geometry
    "provision_awgn_link": ("wannado.wce.blocks.channel_classic.awgn_shannon1948", "provision_awgn_link"),
    "AWGNConfig": ("wannado.wce.blocks.channel_classic.awgn_shannon1948", "AWGNConfig"),
    "provision_rayleigh_clarke": ("wannado.wce.blocks.channel_classic.rayleigh_clarke1968", "provision_rayleigh_clarke"),
    "RayleighClarkeConfig": ("wannado.wce.blocks.channel_classic.rayleigh_clarke1968", "RayleighClarkeConfig"),
    "provision_rician_los": ("wannado.wce.blocks.channel_classic.rician_los_rice1945", "provision_rician_los"),
    "RicianConfig": ("wannado.wce.blocks.channel_classic.rician_los_rice1945", "RicianConfig"),
    "provision_nakagami_m": ("wannado.wce.blocks.channel_classic.nakagami_m1960", "provision_nakagami_m"),
    "NakagamiConfig": ("wannado.wce.blocks.channel_classic.nakagami_m1960", "NakagamiConfig"),
    "provision_saleh_valenzuela": ("wannado.wce.blocks.channel_classic.saleh_valenzuela1987", "provision_saleh_valenzuela"),
    "SalehValenzuelaConfig": ("wannado.wce.blocks.channel_classic.saleh_valenzuela1987", "SalehValenzuelaConfig"),
    "provision_ppp_stochastic_geo": ("wannado.wce.blocks.channel_classic.ppp_cellular_andrews2011", "provision_ppp_stochastic_geo"),
    "PPPCellularConfig": ("wannado.wce.blocks.channel_classic.ppp_cellular_andrews2011", "PPPCellularConfig"),
    # 2. 3GPP NR / NTN / UAV / V2X standardized channels
    "provision_tr38901_uma_umi_rma": ("wannado.wce.blocks.channel_3gpp.tr38901_macro_micro", "provision_tr38901_uma_umi_rma"),
    "TR38901MacroMicroConfig": ("wannado.wce.blocks.channel_3gpp.tr38901_macro_micro", "TR38901MacroMicroConfig"),
    "provision_tr38901_inf": ("wannado.wce.blocks.channel_3gpp.tr38901_indoor_factory", "provision_tr38901_inf"),
    "TR38901InFConfig": ("wannado.wce.blocks.channel_3gpp.tr38901_indoor_factory", "TR38901InFConfig"),
    "provision_tr38901_cdl_tdl": ("wannado.wce.blocks.channel_3gpp.tr38901_cdl_tdl", "provision_tr38901_cdl_tdl"),
    "CDLTDLConfig": ("wannado.wce.blocks.channel_3gpp.tr38901_cdl_tdl", "CDLTDLConfig"),
    "provision_ntn_leo_satellite": ("wannado.wce.blocks.channel_3gpp.tr38811_ntn_leo", "provision_ntn_leo_satellite"),
    "NTNLEOConfig": ("wannado.wce.blocks.channel_3gpp.tr38811_ntn_leo", "NTNLEOConfig"),
    "provision_uav_air_to_ground": ("wannado.wce.blocks.channel_3gpp.tr36777_uav_a2g", "provision_uav_air_to_ground"),
    "UAVA2GConfig": ("wannado.wce.blocks.channel_3gpp.tr36777_uav_a2g", "UAVA2GConfig"),
    "provision_v2x_vehicular_sl": ("wannado.wce.blocks.channel_3gpp.tr37885_v2x_sl", "provision_v2x_vehicular_sl"),
    "V2XConfig": ("wannado.wce.blocks.channel_3gpp.tr37885_v2x_sl", "V2XConfig"),
    # 3. Arrays, Near-Field XL-MIMO, RIS, & Flexible Antennas
    "provision_farfield_upa": ("wannado.wce.blocks.arrays_and_ris.farfield_upa_manifold", "provision_farfield_upa"),
    "FarFieldUPAConfig": ("wannado.wce.blocks.arrays_and_ris.farfield_upa_manifold", "FarFieldUPAConfig"),
    "provision_nearfield_xlmimo_thz": ("wannado.wce.blocks.arrays_and_ris.nearfield_xlmimo_cui2022", "provision_nearfield_xlmimo_thz"),
    "NearFieldXLMIMOConfig": ("wannado.wce.blocks.arrays_and_ris.nearfield_xlmimo_cui2022", "NearFieldXLMIMOConfig"),
    "provision_passive_ris_cascade": ("wannado.wce.blocks.arrays_and_ris.passive_ris_wu2019", "provision_passive_ris_cascade"),
    "PassiveRISConfig": ("wannado.wce.blocks.arrays_and_ris.passive_ris_wu2019", "PassiveRISConfig"),
    "provision_star_ris": ("wannado.wce.blocks.arrays_and_ris.star_ris_xu2021", "provision_star_ris"),
    "STARRISConfig": ("wannado.wce.blocks.arrays_and_ris.star_ris_xu2021", "STARRISConfig"),
    "provision_active_ris_flexible": ("wannado.wce.blocks.arrays_and_ris.active_ris_and_flexible_antennas", "provision_active_ris_flexible"),
    "ActiveRISFlexibleConfig": ("wannado.wce.blocks.arrays_and_ris.active_ris_and_flexible_antennas", "ActiveRISFlexibleConfig"),
    # 4. Traffic models & Multi-Cell MAC queues
    "provision_traffic_ftp_models": ("wannado.wce.blocks.traffic_and_mac.ftp_traffic_3gpp", "provision_traffic_ftp_models"),
    "FTPTrafficConfig": ("wannado.wce.blocks.traffic_and_mac.ftp_traffic_3gpp", "FTPTrafficConfig"),
    "provision_traffic_poisson_pareto_voip": ("wannado.wce.blocks.traffic_and_mac.poisson_pareto_voip", "provision_traffic_poisson_pareto_voip"),
    "MixedTrafficConfig": ("wannado.wce.blocks.traffic_and_mac.poisson_pareto_voip", "MixedTrafficConfig"),
    "provision_traffic_xr_immersive": ("wannado.wce.blocks.traffic_and_mac.xr_immersive_tr38838", "provision_traffic_xr_immersive"),
    "XRTrafficConfig": ("wannado.wce.blocks.traffic_and_mac.xr_immersive_tr38838", "XRTrafficConfig"),
    "provision_multicell_interference_mac": ("wannado.wce.blocks.traffic_and_mac.multicell_mac_queue", "provision_multicell_interference_mac"),
    "MultiCellMACConfig": ("wannado.wce.blocks.traffic_and_mac.multicell_mac_queue", "MultiCellMACConfig"),
    # 5. IMT-2030 / 6G frontier scenarios
    "provision_isac_jsac_channel": ("wannado.wce.blocks.domain_6g.isac_monostatic_bistatic", "provision_isac_jsac_channel"),
    "ISACConfig": ("wannado.wce.blocks.domain_6g.isac_monostatic_bistatic", "ISACConfig"),
    "provision_wifi_sensing_80211bf": ("wannado.wce.blocks.domain_6g.wifi_sensing_80211bf", "provision_wifi_sensing_80211bf"),
    "WiFiSensingConfig": ("wannado.wce.blocks.domain_6g.wifi_sensing_80211bf", "WiFiSensingConfig"),
    "provision_otfs_delay_doppler": ("wannado.wce.blocks.domain_6g.otfs_hadani2017", "provision_otfs_delay_doppler"),
    "OTFSConfig": ("wannado.wce.blocks.domain_6g.otfs_hadani2017", "OTFSConfig"),
    "provision_hrllc_fbl_urllc": ("wannado.wce.blocks.domain_6g.hrllc_fbl_polyanskiy2010", "provision_hrllc_fbl_urllc"),
    "HRLLCFBLConfig": ("wannado.wce.blocks.domain_6g.hrllc_fbl_polyanskiy2010", "HRLLCFBLConfig"),
    "provision_ambient_iot_backscatter": ("wannado.wce.blocks.domain_6g.ambient_iot_tr38848", "provision_ambient_iot_backscatter"),
    "AmbientIoTConfig": ("wannado.wce.blocks.domain_6g.ambient_iot_tr38848", "AmbientIoTConfig"),
    "provision_semantic_aircomp_channel": ("wannado.wce.blocks.domain_6g.semantic_and_aircomp", "provision_semantic_aircomp_channel"),
    "SemanticAirCompConfig": ("wannado.wce.blocks.domain_6g.semantic_and_aircomp", "SemanticAirCompConfig"),
    "provision_pls_covert_antijamming": ("wannado.wce.blocks.domain_6g.pls_covert_antijamming", "provision_pls_covert_antijamming"),
    "WirelessSecurityConfig": ("wannado.wce.blocks.domain_6g.pls_covert_antijamming", "WirelessSecurityConfig"),
    # 6. Analytic baselines & frontier RadioMap/CKM bridge
    "provision_water_filling": ("wannado.wce.blocks.baselines.water_filling_shannon", "provision_water_filling"),
    "WaterFillingConfig": ("wannado.wce.blocks.baselines.water_filling_shannon", "WaterFillingConfig"),
    "provision_wmmse_precoder": ("wannado.wce.blocks.baselines.wmmse_shi2011", "provision_wmmse_precoder"),
    "WMMSEConfig": ("wannado.wce.blocks.baselines.wmmse_shi2011", "WMMSEConfig"),
    "provision_ls_lmmse_csi": ("wannado.wce.blocks.baselines.ls_lmmse_csi", "provision_ls_lmmse_csi"),
    "CSIEstimatorConfig": ("wannado.wce.blocks.baselines.ls_lmmse_csi", "CSIEstimatorConfig"),
    "provision_proportional_fair": ("wannado.wce.blocks.baselines.proportional_fair_jalali2000", "provision_proportional_fair"),
    "PFSchedulerConfig": ("wannado.wce.blocks.baselines.proportional_fair_jalali2000", "PFSchedulerConfig"),
    "provision_ckm_radiomap_bridge": ("wannado.wce.blocks.baselines.ckm_radiomap_diffusion", "provision_ckm_radiomap_bridge"),
    "CKMRadioMapConfig": ("wannado.wce.blocks.baselines.ckm_radiomap_diffusion", "CKMRadioMapConfig"),
}

BLOCK_ID_TO_RUNNER: dict[str, tuple[str, str]] = {
    k.removeprefix("provision_"): v for k, v in BLOCK_EXPORTS.items() if k.startswith("provision_")
}

__all__ = sorted(BLOCK_EXPORTS.keys())


def lazy_resolve(
    name: str,
    exports: dict[str, tuple[str, str]],
    namespace: dict[str, Any],
    package_name: str,
) -> Any:
    if name not in exports:
        raise AttributeError(f"module {package_name!r} has no attribute {name!r}")
    module_name, attr_name = exports[name]
    module = importlib.import_module(module_name)
    value = getattr(module, attr_name)
    namespace[name] = value
    return value


def __getattr__(name: str) -> Any:
    return lazy_resolve(name, BLOCK_EXPORTS, globals(), __name__)


def __dir__() -> list[str]:
    return sorted(__all__)
