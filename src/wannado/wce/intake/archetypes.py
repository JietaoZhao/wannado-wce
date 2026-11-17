"""Stage 0 Standards-Anchored Scenario Archetypes & Pre-Provisioning Contract Audit."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from wannado.wce.core.types import ParamProvenance
from wannado.wce.core.units import rayleigh_distance_m
from wannado.wce.intake.spec import WCESpec


def _p(
    name: str,
    val: Any,
    unit: str,
    meaning: str,
    auth: str,
    sec: str,
    ckey: str,
    why: str,
) -> ParamProvenance:
    return ParamProvenance(
        parameter=name,
        value=val,
        unit=unit,
        plain_meaning=meaning,
        authority_id=auth,
        authority_section=sec,
        citation_key=ckey,
        rationale=why,
    )


def build_wce_archetype(archetype: str = "imt2030_immersive_xr", *, problem_name: str | None = None) -> WCESpec:
    """Construct a standards-anchored starting WCESpec with complete per-parameter provenance."""
    key = archetype.lower()

    if key == "imt2030_isac_monostatic":
        return WCESpec(
            problem_name=problem_name or "IMT2030_ISAC_Monostatic_Environment",
            archetype=key,
            imt2030_scenario="ISAC",
            goal_plain="Joint monostatic/bistatic radar sensing (target RCS + CRB) and multi-user communication over 3GPP Rel-19 ISAC channels.",
            carrier_fc_ghz=28.0,
            bandwidth_hz=100e6,
            numerology_mu=3,
            tx_power_dbm=33.0,
            noise_figure_db=7.0,
            num_tx=16,
            num_rx=16,
            aperture_m=0.08,
            min_tx_rx_distance_m=40.0,
            max_speed_mps=15.0,
            slot_duration_ms=0.125,
            csi_assumption="realistic_estimated",
            traffic_model="ftp_model_3",
            topology_type="UMi_ISAC",
            blocks_selected=["isac_jsac_channel", "passive_ris_cascade", "tr38901_cdl_tdl"],
            baseline_blocks=["wmmse_precoder", "awgn_link"],
            standard_ids=["itur_m2160", "gpp_tr22837", "gpp_tr38901_isac", "gpp_ts38211"],
            parameter_provenance=[
                _p("carrier_fc_ghz", 28.0, "GHz", "FR2 mmWave carrier frequency", "gpp_tr38901_isac", "Rel-19 ISAC", "3gpp_tr38901_isac_rel19", "Provides wide 100 MHz bandwidth and angular resolution for sensing."),
                _p("bandwidth_hz", 100, "MHz", "ISAC RF bandwidth", "gpp_tr22837", "Clause 5", "3gpp_tr22837", "Controls sensing range resolution c/(2B) = 1.5 m and thermal noise floor."),
                _p("numerology_mu", 3, "enum", "NR FR2 SCS=120 kHz, slot=0.125 ms", "gpp_ts38211", "Clause 4.2", "3gpp_ts38211", "Standardized FR2 mmWave numerology avoiding excessive phase noise / ICI."),
                _p("target_rcs_dbsm", 0.0, "dBsm", "Target Radar Cross Section", "liu_isac_crb", "IEEE JSAC 2022", "liu2022integrated", "Canonical pedestrian/drone-scale RCS benchmark."),
                _p("csi_assumption", "realistic_estimated", "enum", "Noisy/causal CSI feedback", "gpp_tr38843", "Clause 5", "3gpp_tr38843", "Prevents genie-aided sensing/beamforming leakage."),
            ],
        )

    if key == "imt2030_hrllc_factory":
        return WCESpec(
            problem_name=problem_name or "IMT2030_HRLLC_SmartFactory",
            archetype=key,
            imt2030_scenario="HRLLC",
            goal_plain="Sub-millisecond factory automation wireless environment honoring TR 38.901 InF clutter and Polyanskiy FBL short-packet bounds.",
            carrier_fc_ghz=3.5,
            bandwidth_hz=100e6,
            numerology_mu=2,
            tx_power_dbm=24.0,
            noise_figure_db=7.0,
            num_tx=4,
            num_rx=2,
            aperture_m=0.13,
            min_tx_rx_distance_m=15.0,
            max_speed_mps=3.0,
            slot_duration_ms=0.25,
            csi_assumption="realistic_estimated",
            traffic_model="poisson",
            topology_type="InF-DL",
            blocks_selected=["tr38901_inf", "hrllc_fbl_urllc", "tr38901_cdl_tdl"],
            baseline_blocks=["awgn_link", "ls_lmmse_csi"],
            standard_ids=["itur_m2160", "gpp_tr38901", "gpp_tr38824", "gpp_ts38211"],
            parameter_provenance=[
                _p("topology_type", "InF-DL", "enum", "Indoor Factory Dense Clutter Low BS", "gpp_tr38901", "Table 7.4.1-1", "3gpp_tr38901", "Captures dense metallic robot/machinery shadowing (sigma_SF=7.2 dB)."),
                _p("numerology_mu", 2, "enum", "SCS=60 kHz, slot=0.25 ms", "gpp_ts38211", "Clause 4.2", "3gpp_ts38211", "Matches sub-millisecond URLLC/HRLLC slot budget."),
                _p("blocklength_n", 200, "uses", "Short-packet FBL coding length", "polyanskiy_fbl_dispersion", "IEEE TIT 2010", "polyanskiy2010channel", "Prevents Shannon infinite-blocklength rate overestimation."),
                _p("target_bler", "1e-5", "prob", "Ultra-reliable error target", "gpp_tr38824", "Clause 5", "3gpp_tr38824", "Baseline 5G URLLC / 6G HRLLC link error requirement."),
            ],
        )

    if key == "imt2030_massive_ambient_iot":
        return WCESpec(
            problem_name=problem_name or "IMT2030_Massive_AmbientIoT",
            archetype=key,
            imt2030_scenario="MC",
            goal_plain="Ultra-dense sustainable IoT / batteryless backscatter environment calibrated to TR 38.848 and PPP spatial geometry.",
            carrier_fc_ghz=0.9,
            bandwidth_hz=180e3,
            numerology_mu=0,
            tx_power_dbm=33.0,
            noise_figure_db=6.0,
            num_tx=1,
            num_rx=1,
            aperture_m=0.16,
            min_tx_rx_distance_m=5.0,
            max_speed_mps=1.5,
            slot_duration_ms=1.0,
            csi_assumption="realistic_estimated",
            traffic_model="poisson",
            topology_type="AmbientIoT_Topology2",
            blocks_selected=["ambient_iot_backscatter", "ppp_stochastic_geo", "traffic_poisson_pareto_voip"],
            baseline_blocks=["proportional_fair", "awgn_link"],
            standard_ids=["itur_m2160", "gpp_tr38848", "gpp_tr38913"],
            parameter_provenance=[
                _p("carrier_fc_ghz", 0.9, "GHz", "Sub-1 GHz wide-coverage IoT band", "gpp_tr38848", "Clause 5", "3gpp_tr38848", "Favorable link budget for uW RF energy harvesting."),
                _p("topology_type", "Topology 2", "enum", "Helper-assisted backscatter two-hop", "gpp_tr38848", "Clause 4.2", "3gpp_tr38848", "Overcomes double pathloss of batteryless tags."),
                _p("traffic_model", "Poisson", "enum", "Memoryless sporadic mMTC arrivals", "gpp_tr36814", "Annex A", "3gpp_tr36814", "Standard mMTC sensor packet arrival process."),
            ],
        )

    if key == "imt2030_ubiquitous_leo_ntn":
        return WCESpec(
            problem_name=problem_name or "IMT2030_Ubiquitous_LEO_NTN",
            archetype=key,
            imt2030_scenario="UC",
            goal_plain="600 km LEO satellite-to-ground propagation with elevation-dependent clutter/K-factor, orbital delay, and Doppler/OTFS.",
            carrier_fc_ghz=2.0,
            bandwidth_hz=30e6,
            numerology_mu=0,
            tx_power_dbm=78.0,
            noise_figure_db=7.0,
            num_tx=1,
            num_rx=1,
            aperture_m=0.5,
            min_tx_rx_distance_m=600_000.0,
            max_speed_mps=7560.0,
            slot_duration_ms=1.0,
            csi_assumption="delayed_feedback",
            traffic_model="ftp_model_3",
            topology_type="NTN_LEO_600km",
            blocks_selected=["ntn_leo_satellite", "otfs_delay_doppler", "rician_los"],
            baseline_blocks=["awgn_link"],
            standard_ids=["itur_m2160", "gpp_tr38811", "gpp_tr38821"],
            parameter_provenance=[
                _p("orbit_altitude_km", 600.0, "km", "LEO satellite altitude", "gpp_tr38821", "Clause 5.1", "3gpp_tr38821", "Canonical 3GPP TR 38.821 LEO constellation height."),
                _p("carrier_fc_ghz", 2.0, "GHz", "S-band NTN handheld carrier", "gpp_tr38811", "Clause 6", "3gpp_tr38811", "Standardized S-band (2 GHz) / Ka-band (20-30 GHz) NTN spectrum."),
                _p("elevation_deg", 30.0, "deg", "Minimum/nominal UE look angle", "gpp_tr38811", "Table 6.6.2-1", "3gpp_tr38811", "Drives LoS prob, clutter loss, and Ricean K interpolation."),
            ],
        )

    if key == "imt2030_aiac_semantic_aircomp":
        return WCESpec(
            problem_name=problem_name or "IMT2030_AIAC_Semantic_AirComp",
            archetype=key,
            imt2030_scenario="AIAC",
            goal_plain="Differentiable baseband semantic communication, AirComp federated aggregation over Cell-Free APs, and CKM priors.",
            carrier_fc_ghz=3.5,
            bandwidth_hz=20e6,
            numerology_mu=1,
            tx_power_dbm=23.0,
            noise_figure_db=7.0,
            num_tx=8,
            num_rx=4,
            aperture_m=0.30,
            min_tx_rx_distance_m=20.0,
            max_speed_mps=3.0,
            slot_duration_ms=0.5,
            csi_assumption="realistic_estimated",
            traffic_model="ftp_model_3",
            topology_type="CellFree_AirComp",
            blocks_selected=["semantic_aircomp_channel", "ls_lmmse_csi", "ckm_radiomap_bridge"],
            baseline_blocks=["water_filling", "wmmse_precoder"],
            standard_ids=["itur_m2160", "gpp_tr38843", "gpp_tr38914"],
            parameter_provenance=[
                _p("csi_framework", "TR 38.843", "enum", "AI/ML air interface evaluation", "gpp_tr38843", "Clause 5", "3gpp_tr38843", "Defines CSI compression & beam ML training/eval boundaries."),
                _p("aircomp_cellfree", "12 devices / 8 APs", "nodes", "Over-the-air FL topology", "zhu_aircomp_feel", "IEEE TWC 2020", "zhu2020broadband", "Analog waveform superposition + Cell-Free macro-diversity."),
            ],
        )

    if key == "uav_ris_relay":
        return WCESpec(
            problem_name=problem_name or "UAV_RIS_Relay_Environment",
            archetype=key,
            imt2030_scenario="UC",
            goal_plain="3D UAV Air-to-Ground probabilistic LoS link combined with passive/STAR/active RIS metasurface relay.",
            carrier_fc_ghz=3.5,
            bandwidth_hz=20e6,
            numerology_mu=1,
            tx_power_dbm=30.0,
            noise_figure_db=7.0,
            num_tx=8,
            num_rx=1,
            aperture_m=0.30,
            min_tx_rx_distance_m=50.0,
            max_speed_mps=18.0,
            slot_duration_ms=0.5,
            csi_assumption="realistic_estimated",
            traffic_model="ftp_model_3",
            topology_type="UAV_RIS",
            blocks_selected=["uav_air_to_ground", "passive_ris_cascade", "star_ris", "active_ris_flexible"],
            baseline_blocks=["rician_los", "water_filling"],
            standard_ids=["gpp_tr36777", "etsi_gr_ris_003", "itur_m2160"],
            parameter_provenance=[
                _p("uav_altitude_m", 100.0, "m", "UAV operational altitude", "gpp_tr36777", "Annex B", "3gpp_tr36777", "Balances elevation LoS probability (Al-Hourani 2014) vs 3D pathloss."),
                _p("ris_elements", 64, "elements", "RIS unit-cell count", "wu_zhang_passive_ris", "IEEE TWC 2019", "wu2019intelligent", "Provides quadratic array gain under |phi_m|=1 passive conservation."),
            ],
        )

    if key == "xlmimo_nearfield_thz":
        return WCESpec(
            problem_name=problem_name or "XLMIMO_NearField_SubTHz",
            archetype=key,
            imt2030_scenario="IC",
            goal_plain="Sub-THz / mmWave Extremely Large-Scale MIMO operating in the radiative near-field with ITU-R P.676 gas absorption.",
            carrier_fc_ghz=100.0,
            bandwidth_hz=1e9,
            numerology_mu=4,
            tx_power_dbm=30.0,
            noise_figure_db=8.0,
            num_tx=128,
            num_rx=4,
            aperture_m=0.19,
            min_tx_rx_distance_m=15.0,
            max_speed_mps=3.0,
            slot_duration_ms=0.0625,
            csi_assumption="realistic_estimated",
            traffic_model="xr_60fps",
            topology_type="NearField_THz",
            blocks_selected=["nearfield_xlmimo_thz", "farfield_upa", "saleh_valenzuela"],
            baseline_blocks=["ls_lmmse_csi", "water_filling"],
            standard_ids=["itur_p676", "itur_m2160"],
            parameter_provenance=[
                _p("carrier_fc_ghz", 100.0, "GHz", "Sub-THz carrier frequency", "itur_p676", "Annex 1", "itur_p676_2022", "Triggers molecular O2/H2O absorption and wide Rayleigh distance Z_R."),
                _p("num_tx", 128, "elements", "ELAA / XL-MIMO aperture width", "cui_dai_nearfield_spherical", "IEEE TCOM 2022", "cui2022channel", f"Yields Z_R = {rayleigh_distance_m(0.19, 100e9):.1f} m > user distance 15 m."),
            ],
        )

    if key == "v2x_highway_urllc":
        return WCESpec(
            problem_name=problem_name or "NR_V2X_Highway_URLLC",
            archetype=key,
            imt2030_scenario="HRLLC",
            goal_plain="5.9 GHz high-speed highway V2V sidelink with dual-mobility Doppler, NLOSv truck blockage, and OTFS/FBL comparison.",
            carrier_fc_ghz=5.9,
            bandwidth_hz=20e6,
            numerology_mu=1,
            tx_power_dbm=23.0,
            noise_figure_db=9.0,
            num_tx=2,
            num_rx=2,
            aperture_m=0.05,
            min_tx_rx_distance_m=15.0,
            max_speed_mps=61.1,
            slot_duration_ms=0.5,
            csi_assumption="realistic_estimated",
            traffic_model="poisson",
            topology_type="V2X_Highway",
            blocks_selected=["v2x_vehicular_sl", "otfs_delay_doppler", "hrllc_fbl_urllc"],
            baseline_blocks=["awgn_link"],
            standard_ids=["gpp_tr37885", "gpp_tr38824", "itur_m2160"],
            parameter_provenance=[
                _p("carrier_fc_ghz", 5.9, "GHz", "ITS V2X sidelink band", "gpp_tr37885", "Clause 6", "3gpp_tr37885", "Standardized 5.9 GHz V2V frequency with 80 m highway breakpoint."),
                _p("nlosv_blockage", "15 dB", "dB", "Heavy-vehicle NLOSv attenuation", "gpp_tr37885", "Clause 6.2", "3gpp_tr37885", "Models intermediate truck/bus shadowing on highway lanes."),
            ],
        )

    if key == "multicell_ofdma_mac":
        return WCESpec(
            problem_name=problem_name or "MultiCell_Dynamic_Scheduling_MAC",
            archetype=key,
            imt2030_scenario="IC",
            goal_plain="Multi-cell wrap-around downlink interference & non-full-buffer queues with Gym interface for DRL vs PF scheduling.",
            carrier_fc_ghz=3.5,
            bandwidth_hz=10e6,
            numerology_mu=1,
            tx_power_dbm=35.0,
            noise_figure_db=7.0,
            num_tx=4,
            num_rx=1,
            aperture_m=0.13,
            min_tx_rx_distance_m=20.0,
            max_speed_mps=8.3,
            slot_duration_ms=1.0,
            csi_assumption="realistic_estimated",
            traffic_model="ftp_model_3",
            topology_type="UMi_WrapAround",
            blocks_selected=["multicell_interference_mac", "tr38901_uma_umi_rma", "traffic_ftp_models"],
            baseline_blocks=["proportional_fair", "wmmse_precoder"],
            export_targets=["gym_env", "wannado_drl", "markdown_report", "latex_table"],
            standard_ids=["gpp_tr38901", "gpp_ts38211", "gpp_tr36814"],
            parameter_provenance=[
                _p("topology_type", "UMi Wrap-Around", "enum", "4-cell periodic torus layout", "gpp_tr38901", "Table 7.4.1-1", "3gpp_tr38901", "Eliminates edge-cell interference underestimation."),
                _p("traffic_model", "FTP Model 3", "enum", "Poisson bursty traffic queues", "gpp_tr36814", "Annex A.2.1.3", "3gpp_tr36814", "Avoids unrealistic infinite full-buffer traffic."),
            ],
        )

    if key == "jamming_contested_rf":
        return WCESpec(
            problem_name=problem_name or "Jamming_Contested_RF_PLS_Environment",
            archetype=key,
            imt2030_scenario="HRLLC",
            goal_plain="Complex64 baseband security environment covering Wyner secrecy, covert KL limits, and JSR-dB jamming aligned with wannado-adv.",
            carrier_fc_ghz=2.4,
            bandwidth_hz=20e6,
            numerology_mu=0,
            tx_power_dbm=20.0,
            noise_figure_db=7.0,
            num_tx=1,
            num_rx=1,
            aperture_m=0.06,
            min_tx_rx_distance_m=25.0,
            max_speed_mps=5.0,
            slot_duration_ms=1.0,
            csi_assumption="realistic_estimated",
            traffic_model="poisson",
            topology_type="Contested_RF",
            blocks_selected=["pls_covert_antijamming", "rayleigh_clarke", "nakagami_m"],
            baseline_blocks=["awgn_link"],
            export_targets=["pytorch_complex", "wannado_adv", "markdown_report", "latex_table"],
            standard_ids=["itur_m2160"],
            parameter_provenance=[
                _p("jsr_db", -10.0, "dB", "Jamming-to-Signal Ratio", "sadeghi_rf_adversarial", "IEEE WCL 2019", "sadeghi2019adversarial", "Matches wannado-adv complex64 norm='jsr_db' evaluation budget."),
            ],
        )

    if key == "uav_gis":
        return WCESpec(
            problem_name=problem_name or "UAV_Aerial_GIS_Telemetry",
            archetype=key,
            imt2030_scenario="UC",
            goal_plain="3GPP TR 36.777 aerial UAV telemetry & GIS swarm delivery channel with altitude-dependent LoS probability and short-packet reliability.",
            carrier_fc_ghz=2.4,
            bandwidth_hz=20e6,
            numerology_mu=1,
            tx_power_dbm=30.0,
            noise_figure_db=7.0,
            num_tx=4,
            num_rx=2,
            aperture_m=0.19,
            min_tx_rx_distance_m=30.0,
            max_speed_mps=18.0,
            slot_duration_ms=0.5,
            csi_assumption="realistic_estimated",
            traffic_model="ftp_model_3",
            topology_type="UMa-AV",
            blocks_selected=["uav_air_to_ground", "farfield_upa", "hrllc_fbl_urllc"],
            baseline_blocks=["awgn_link", "water_filling"],
            export_targets=["pytorch_complex", "gym_env", "wannado_drl", "markdown_report"],
            standard_ids=["itur_m2160", "gpp_tr36777", "gpp_ts38211"],
            parameter_provenance=[
                _p("carrier_fc_ghz", 2.4, "GHz", "ISM/UAV command & GIS telemetry band", "gpp_tr36777", "Clause 4", "3gpp_tr36777", "Long-range UAV A2G link budget."),
                _p("bandwidth_hz", 20, "MHz", "UAV telemetry channel bandwidth", "gpp_ts38211", "Clause 4.2", "3gpp_ts38211", "Governs thermal floor and airtime latency."),
            ],
        )

    if key == "federated_edge":
        return WCESpec(
            problem_name=problem_name or "Federated_AI_Edge_Wireless_FL",
            archetype=key,
            imt2030_scenario="AIAC",
            goal_plain="Federated Learning / distributed gradient aggregation over realistic 5G/6G fading, straggler deadlines, and causal CSI.",
            carrier_fc_ghz=3.5,
            bandwidth_hz=20e6,
            numerology_mu=1,
            tx_power_dbm=23.0,
            noise_figure_db=7.0,
            num_tx=8,
            num_rx=2,
            aperture_m=0.30,
            min_tx_rx_distance_m=20.0,
            max_speed_mps=3.0,
            slot_duration_ms=0.5,
            csi_assumption="realistic_estimated",
            traffic_model="ftp_model_3",
            topology_type="UMa_FL",
            blocks_selected=["tr38901_uma_umi_rma", "semantic_aircomp_channel", "multicell_interference_mac"],
            baseline_blocks=["ls_lmmse_csi", "wmmse_precoder", "proportional_fair"],
            standard_ids=["itur_m2160", "gpp_tr38901", "gpp_ts38211"],
            parameter_provenance=[
                _p("carrier_fc_ghz", 3.5, "GHz", "Mid-band edge AI aggregation carrier", "gpp_tr38901", "Table 7.4.1-1", "3gpp_tr38901", "Standard urban macro FL uplink spectrum."),
                _p("tx_power_dbm", 23.0, "dBm", "Edge UE transmit power budget", "gpp_ts38211", "UE Class 3", "3gpp_ts38211", "Determines client upload duration and straggler dropout."),
            ],
        )

    if key == "v2v_control":
        return WCESpec(
            problem_name=problem_name or "Autonomous_V2V_Platooning_Control",
            archetype=key,
            imt2030_scenario="HRLLC",
            goal_plain="High-mobility 5.9 GHz V2V sidelink control channel with TR 37.885 shadowing, Polyanskiy FBL latency/error bounds, and Doppler OTFS.",
            carrier_fc_ghz=5.9,
            bandwidth_hz=20e6,
            numerology_mu=2,
            tx_power_dbm=23.0,
            noise_figure_db=9.0,
            num_tx=4,
            num_rx=4,
            aperture_m=0.08,
            min_tx_rx_distance_m=15.0,
            max_speed_mps=33.3,
            slot_duration_ms=0.25,
            csi_assumption="realistic_estimated",
            traffic_model="ftp_model_3",
            topology_type="Highway_V2V",
            blocks_selected=["v2x_vehicular_sl", "hrllc_fbl_urllc", "otfs_delay_doppler"],
            baseline_blocks=["awgn_link", "ls_lmmse_csi"],
            export_targets=["pytorch_complex", "gym_env", "wannado_drl", "markdown_report"],
            standard_ids=["itur_m2160", "gpp_tr37885", "gpp_ts38211"],
            parameter_provenance=[
                _p("carrier_fc_ghz", 5.9, "GHz", "ITS C-V2X sidelink spectrum", "gpp_tr37885", "Section 6", "3gpp_tr37885", "Dedicated safety-critical autonomous driving band."),
                _p("max_speed_mps", 33.3, "m/s", "Highway relative platoon speed (~120 km/h)", "gpp_tr37885", "Section 6.1", "3gpp_tr37885", "Governs Doppler shift and mu=2 short slot selection."),
            ],
        )

    # Default: imt2030_immersive_xr
    return WCESpec(
        problem_name=problem_name or "IMT2030_Immersive_XR_5G6G",
        archetype="imt2030_immersive_xr",
        imt2030_scenario="IC",
        goal_plain="Standardized 3GPP TR 38.901 UMa/CDL-C wireless environment serving 60 fps XR immersive traffic with 10 ms PDB.",
        carrier_fc_ghz=3.5,
        bandwidth_hz=20e6,
        numerology_mu=1,
        tx_power_dbm=43.0,
        noise_figure_db=7.0,
        num_tx=8,
        num_rx=2,
        aperture_m=0.30,
        min_tx_rx_distance_m=25.0,
        max_speed_mps=10.0,
        slot_duration_ms=0.5,
        csi_assumption="realistic_estimated",
        traffic_model="xr_60fps",
        topology_type="UMa",
        blocks_selected=["tr38901_uma_umi_rma", "tr38901_cdl_tdl", "traffic_xr_immersive"],
        baseline_blocks=["wmmse_precoder", "proportional_fair"],
        standard_ids=["itur_m2160", "gpp_tr38901", "gpp_tr38838", "gpp_ts38211"],
        parameter_provenance=[
            _p("carrier_fc_ghz", 3.5, "GHz", "Mid-band n78 5G/6G carrier", "gpp_tr38901", "Table 7.4.1-1", "3gpp_tr38901", "Primary FR1 urban macro benchmark frequency."),
            _p("numerology_mu", 1, "enum", "30 kHz SCS, 0.5 ms slot", "gpp_ts38211", "Clause 4.2", "3gpp_ts38211", "Standard FR1 TDD mid-band numerology."),
            _p("traffic_model", "XR 60 fps (10 ms PDB)", "enum", "Periodic immersive video stream", "gpp_tr38838", "Clause 4-6", "3gpp_tr38838", "Enforces >=99% packet PDB satisfaction metric."),
        ],
    )


@dataclass
class SpecAuditReport:
    """Result of Stage 0 physical contract verification on a WCESpec."""

    ok: bool
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "warnings": self.warnings, "errors": self.errors, "recommendations": self.recommendations}


def audit_wce_spec(spec: WCESpec) -> SpecAuditReport:
    """Validate Stage 0 physical & standards consistency of a WCESpec before provisioning."""
    errors: list[str] = []
    warnings: list[str] = []
    recs: list[str] = []

    if spec.carrier_fc_ghz <= 0.0:
        errors.append("carrier_fc_ghz must be strictly positive.")
    if spec.bandwidth_hz <= 0.0:
        errors.append("bandwidth_hz must be strictly positive.")
    if spec.min_tx_rx_distance_m <= 0.0:
        errors.append("min_tx_rx_distance_m must be strictly positive (> 0 m).")
    if spec.num_tx < 1 or spec.num_rx < 1:
        errors.append(f"num_tx ({spec.num_tx}) and num_rx ({spec.num_rx}) must both be >= 1.")
    if spec.numerology_mu not in (0, 1, 2, 3, 4):
        errors.append(f"numerology_mu must be in {{0, 1, 2, 3, 4}}, got {spec.numerology_mu}.")
    if spec.noise_figure_db < 0.0:
        warnings.append(f"noise_figure_db={spec.noise_figure_db} dB < 0 is non-physical for passive receivers.")
    if spec.tx_power_dbm < -30.0 or (spec.tx_power_dbm > 80.0 and "ntn_leo_satellite" not in spec.blocks_selected):
        warnings.append(f"Extreme tx_power_dbm={spec.tx_power_dbm} dBm for terrestrial deployment; verify units (dBm vs dBW).")
    if not spec.blocks_selected:
        errors.append("blocks_selected cannot be empty.")
    if not spec.parameter_provenance:
        warnings.append("No parameter_provenance entries recorded; add authority citations for every knob.")

    # TR 38.901 frequency range guard [0.5, 100] GHz
    if "tr38901_uma_umi_rma" in spec.blocks_selected and not (0.5 <= spec.carrier_fc_ghz <= 100.0):
        warnings.append(f"fc={spec.carrier_fc_ghz} GHz lies outside 3GPP TR 38.901 validated range [0.5, 100] GHz.")

    # Near-field Rayleigh distance mismatch check
    physics = spec.derived_physics_summary()
    if physics["is_nearfield_regime"] > 0.5 and "nearfield_xlmimo_thz" not in spec.blocks_selected:
        warnings.append(
            f"min_tx_rx_distance_m ({spec.min_tx_rx_distance_m:.1f} m) < Rayleigh distance ({physics['rayleigh_distance_m']:.1f} m): "
            "far-field plane-wave assumption is inaccurate! Include `nearfield_xlmimo_thz`."
        )

    # Coherence time vs slot duration check
    tc_ms = physics["coherence_time_ms"]
    if tc_ms < spec.slot_duration_ms:
        warnings.append(
            f"Channel coherence time ({tc_ms:.3f} ms) is shorter than 1 slot ({spec.slot_duration_ms} ms); "
            "increase numerology_mu or evaluate `otfs_delay_doppler`."
        )

    # Genie CSI warning
    if spec.csi_assumption == "ideal_genie_baseline":
        warnings.append("csi_assumption='ideal_genie_baseline' should only be reported as an analytical upper bound alongside realistic noisy CSI.")

    if "ambient_iot_backscatter" in spec.blocks_selected and "traffic_xr_immersive" in spec.blocks_selected:
        warnings.append("Feasibility conflict: Ambient IoT uW backscatter cannot directly serve >=15 Mbps XR streaming (`traffic_xr_immersive`); separate IoT control from broadband links.")

    if not spec.baseline_blocks:
        recs.append("Include at least one classic analytical baseline (e.g., `awgn_link`, `wmmse_precoder`, `proportional_fair`).")

    return SpecAuditReport(ok=len(errors) == 0, warnings=warnings, errors=errors, recommendations=recs)
