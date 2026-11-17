"""ITU-R Recommendation M.2160-0 (IMT-2030 / 6G) usage scenarios, 15 capabilities, and matching helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class IMT2030ScenarioSpec:
    """Structured IMT-2030 (6G) usage scenario blueprint anchored to Recommendation ITU-R M.2160-0."""

    id: str
    name: str
    evolved_from_5g: str | None
    summary_plain: str
    key_capabilities: tuple[str, ...]
    kpi_targets: dict[str, str]
    recommended_wce_blocks: tuple[str, ...]
    standard_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["key_capabilities"] = list(self.key_capabilities)
        data["recommended_wce_blocks"] = list(self.recommended_wce_blocks)
        data["standard_ids"] = list(self.standard_ids)
        return data


IMT2030_USAGE_SCENARIOS: dict[str, IMT2030ScenarioSpec] = {
    "IC": IMT2030ScenarioSpec(
        id="IC",
        name="Immersive Communication",
        evolved_from_5g="eMBB (IMT-2020)",
        summary_plain="Ultra-high-rate interactive XR, holographic telepresence, and multi-user spatial video requiring consistent throughput and bounded jitter.",
        key_capabilities=("peak_data_rate", "user_experienced_data_rate", "spectrum_efficiency", "area_traffic_capacity", "mobility", "sustainability"),
        kpi_targets={
            "peak_data_rate": "50 - 200 Gbps",
            "user_experienced_rate": "300 - 500 Mbps+",
            "area_traffic_capacity": "30 - 50 Mbps/m^2+",
            "spectrum_efficiency": "1.5x - 3x IMT-2020",
            "xr_pdb_and_fps": "10 - 20 ms PDB @ 60 - 120 fps (3GPP TR 38.838)",
        },
        recommended_wce_blocks=("tr38901_uma_umi_rma", "tr38901_cdl_tdl", "nearfield_xlmimo_thz", "traffic_xr_immersive", "wmmse_precoder"),
        standard_ids=("itur_m2160", "gpp_tr38901", "gpp_tr38838", "gpp_ts38211"),
    ),
    "HRLLC": IMT2030ScenarioSpec(
        id="HRLLC",
        name="Hyper-Reliable and Low-Latency Communication",
        evolved_from_5g="URLLC (IMT-2020)",
        summary_plain="Deterministic sub-millisecond industrial control, robotic surgery, smart grid protection, and cooperative autonomous swarms.",
        key_capabilities=("user_plane_latency", "reliability", "mobility", "security_and_resilience", "positioning"),
        kpi_targets={
            "user_plane_latency": "0.1 - 1 ms",
            "reliability": "1 - 10^-5 to 1 - 10^-7",
            "mobility": "500 - 1000 km/h",
        },
        recommended_wce_blocks=("hrllc_fbl_urllc", "tr38901_inf", "v2x_vehicular_sl", "otfs_delay_doppler", "tr38901_cdl_tdl"),
        standard_ids=("itur_m2160", "gpp_tr38824", "gpp_tr38901", "gpp_tr37885", "gpp_ts38211"),
    ),
    "MC": IMT2030ScenarioSpec(
        id="MC",
        name="Massive Communication",
        evolved_from_5g="mMTC (IMT-2020)",
        summary_plain="Ultra-dense smart city sensors, pervasive digital twins, and batteryless / ambient energy-harvesting IoT.",
        key_capabilities=("connection_density", "sustainability", "coverage", "security_and_resilience"),
        kpi_targets={
            "connection_density": "10^6 - 10^8 devices/km^2",
            "energy_sustainability": "uW-class ambient backscatter / net-zero duty cycle",
        },
        recommended_wce_blocks=("ambient_iot_backscatter", "ppp_stochastic_geo", "traffic_poisson_pareto_voip", "proportional_fair"),
        standard_ids=("itur_m2160", "gpp_tr38913", "gpp_tr38848"),
    ),
    "UC": IMT2030ScenarioSpec(
        id="UC",
        name="Ubiquitous Connectivity",
        evolved_from_5g=None,
        summary_plain="Seamless coverage across rural, oceanic, aerial, and orbital layers via terrestrial-NTN (LEO/UAV/HAPS) convergence.",
        key_capabilities=("coverage", "interoperability", "mobility", "sustainability"),
        kpi_targets={
            "coverage": "Global terrestrial + aerial + orbital reach",
            "leo_altitude": "500 - 1200 km (min elevation 10 - 30 deg)",
            "uav_ceiling": "Up to 300 m AGL (3GPP TR 36.777)",
        },
        recommended_wce_blocks=("ntn_leo_satellite", "uav_air_to_ground", "rician_los", "otfs_delay_doppler"),
        standard_ids=("itur_m2160", "gpp_tr38811", "gpp_tr38821", "gpp_tr36777", "gpp_tr38914"),
    ),
    "AIAC": IMT2030ScenarioSpec(
        id="AIAC",
        name="Artificial Intelligence and Communication",
        evolved_from_5g=None,
        summary_plain="Native air-interface AI inference, CSI feedback compression, AirComp federated learning, semantic communication, and generative channel maps.",
        key_capabilities=("ai_related_capabilities", "spectrum_efficiency", "user_plane_latency", "security_and_resilience"),
        kpi_targets={
            "ai_air_interface": "One-sided / two-sided CSI & beam ML interoperability (TR 38.843)",
            "semantic_and_aircomp": "Task-oriented semantic distortion & analog aggregation MSE",
        },
        recommended_wce_blocks=("semantic_aircomp_channel", "ls_lmmse_csi", "ckm_radiomap_bridge", "pls_covert_antijamming"),
        standard_ids=("itur_m2160", "gpp_tr38843", "gpp_tr38914"),
    ),
    "ISAC": IMT2030ScenarioSpec(
        id="ISAC",
        name="Integrated Sensing and Communication",
        evolved_from_5g=None,
        summary_plain="Dual-functional wireless networks delivering data while sensing range, Doppler velocity, angle, imaging, and cm-level positioning.",
        key_capabilities=("sensing", "positioning", "spectrum_efficiency", "security_and_resilience"),
        kpi_targets={
            "positioning_accuracy": "1 - 10 cm",
            "sensing_modes": "Monostatic & Bistatic with Target RCS + CRB",
        },
        recommended_wce_blocks=("isac_jsac_channel", "wifi_sensing_80211bf", "passive_ris_cascade", "nearfield_xlmimo_thz"),
        standard_ids=("itur_m2160", "gpp_tr22837", "gpp_tr38901_isac", "gpp_tr38855", "ieee_80211bf"),
    ),
}


IMT2030_CAPABILITIES: dict[str, dict[str, str]] = {
    "peak_data_rate": {"category": "enhanced", "target": "50 - 200 Gbps"},
    "user_experienced_data_rate": {"category": "enhanced", "target": "300 - 500 Mbps+"},
    "spectrum_efficiency": {"category": "enhanced", "target": "1.5x - 3x IMT-2020"},
    "area_traffic_capacity": {"category": "enhanced", "target": "30 - 50 Mbps/m^2+"},
    "connection_density": {"category": "enhanced", "target": "10^6 - 10^8 devices/km^2"},
    "mobility": {"category": "enhanced", "target": "500 - 1000 km/h"},
    "user_plane_latency": {"category": "enhanced", "target": "0.1 - 1 ms"},
    "reliability": {"category": "enhanced", "target": "1 - 10^-5 to 1 - 10^-7"},
    "security_and_resilience": {"category": "enhanced", "target": "Zero-trust + physical/jamming/PQC resilience"},
    "coverage": {"category": "new", "target": "Deep-indoor, maritime, aerial & satellite reach"},
    "positioning": {"category": "new", "target": "1 - 10 cm indoor/outdoor accuracy"},
    "sensing": {"category": "new", "target": "Native RF detection, tracking, imaging & mapping"},
    "ai_related_capabilities": {"category": "new", "target": "Native distributed training, inference & air-interface ML"},
    "sustainability": {"category": "new", "target": "Net-zero energy scaling & uW ambient IoT"},
    "interoperability": {"category": "new", "target": "Multi-layer cellular, Wi-Fi, satellite & short-range convergence"},
}


def match_imt2030_scenario(query: str) -> list[IMT2030ScenarioSpec]:
    """Match free-text research description to ranked ITU-R M.2160 usage scenarios."""
    q = query.lower()
    keywords: dict[str, tuple[str, ...]] = {
        "IC": ("xr", "vr", "ar", "immersive", "embb", "hologra", "video", "broadband", "throughput"),
        "HRLLC": ("urllc", "hrllc", "latency", "reliable", "factory", "industrial", "fbl", "v2x", "short packet"),
        "MC": ("iot", "mmtc", "massive", "ambient", "backscatter", "sensor", "density"),
        "UC": ("ntn", "satellite", "leo", "geo", "uav", "drone", "aerial", "ubiquitous", "haps"),
        "AIAC": ("semantic", "aircomp", "federated", "csi feedback", "diffusion", "ckm", "nerf", "ai", "ml"),
        "ISAC": ("isac", "jsac", "sensing", "radar", "rcs", "crb", "positioning", "localization", "802.11bf"),
    }
    scored: list[tuple[int, IMT2030ScenarioSpec]] = []
    for sid, spec in IMT2030_USAGE_SCENARIOS.items():
        hits = sum(1 for kw in keywords[sid] if kw in q)
        if hits > 0:
            scored.append((hits, spec))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [s for _, s in scored] or [IMT2030_USAGE_SCENARIOS["IC"]]
