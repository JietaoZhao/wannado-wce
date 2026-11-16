"""Hardwired zero-hallucination provenance registry for 3GPP/ITU/ETSI/IEEE standards, papers, and sub-mechanisms."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from wannado.wce.core.types import LitCitation, StandardRef


@dataclass(frozen=True)
class MechanismProvenance:
    """Verified academic or standard provenance card for a physical/MAC/signal-processing sub-mechanism."""

    id: str
    name: str
    role_plain: str
    citation: LitCitation
    standard_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "role_plain": self.role_plain,
            "standard_id": self.standard_id,
            "citation": asdict(self.citation),
        }


def _std_ref(
    *,
    sid: str,
    org: str,
    doc_id: str,
    section: str,
    title: str,
    release: str,
    year: int,
    url: str,
    summary_plain: str,
    key_parameters: tuple[str, ...],
    bib_key: str,
    authors: str,
    venue: str,
) -> StandardRef:
    bibtex = (
        f"@techreport{{{bib_key},\n"
        f"  title={{{doc_id}: {title}}},\n"
        f"  author={{{authors}}},\n"
        f"  institution={{{org}}},\n"
        f"  number={{{doc_id}}},\n"
        f"  year={{{year}}},\n"
        f"  url={{{url}}}\n"
        f"}}"
    )
    ieee = f'{authors}, "{doc_id}: {title}," {venue}, {year}.'
    apa = f"{authors} ({year}). {doc_id}: {title}. {venue}."
    return StandardRef(
        id=sid,
        org=org,
        doc_id=doc_id,
        section=section,
        title=title,
        release=release,
        year=year,
        url=url,
        summary_plain=summary_plain,
        key_parameters=key_parameters,
        citation=LitCitation(
            key=bib_key,
            title=f"{doc_id}: {title}",
            authors=authors,
            venue=venue,
            year=year,
            url=url,
            bibtex=bibtex,
            ieee=ieee,
            apa=apa,
        ),
    )


def _article_cit(
    *,
    key: str,
    title: str,
    authors: str,
    journal: str,
    year: int,
    url: str,
    volume: str = "",
    number: str = "",
    pages: str = "",
    doi: str = "",
) -> LitCitation:
    lines = [f"@article{{{key},", f"  title={{{title}}},", f"  author={{{authors}}},", f"  journal={{{journal}}},"]
    if volume:
        lines.append(f"  volume={{{volume}}},")
    if number:
        lines.append(f"  number={{{number}}},")
    if pages:
        lines.append(f"  pages={{{pages}}},")
    lines.append(f"  year={{{year}}},")
    if doi:
        lines.append(f"  doi={{{doi}}},")
    lines.append(f"  url={{{url}}}")
    lines.append("}")
    vol_str = f", vol. {volume}" if volume else ""
    num_str = f", no. {number}" if number else ""
    pg_str = f", pp. {pages}" if pages else ""
    return LitCitation(
        key=key,
        title=title,
        authors=authors,
        venue=journal,
        year=year,
        url=url,
        bibtex="\n".join(lines),
        ieee=f'{authors}, "{title}," {journal}{vol_str}{num_str}{pg_str}, {year}.',
        apa=f"{authors} ({year}). {title}. {journal}" + (f", {volume}" if volume else "") + (f"({number})" if number else "") + (f", {pages}" if pages else "") + ".",
    )


def _inproceedings_cit(
    *,
    key: str,
    title: str,
    authors: str,
    booktitle: str,
    year: int,
    url: str,
    pages: str = "",
    doi: str = "",
) -> LitCitation:
    lines = [f"@inproceedings{{{key},", f"  title={{{title}}},", f"  author={{{authors}}},", f"  booktitle={{{booktitle}}},"]
    if pages:
        lines.append(f"  pages={{{pages}}},")
    lines.append(f"  year={{{year}}},")
    if doi:
        lines.append(f"  doi={{{doi}}},")
    lines.append(f"  url={{{url}}}")
    lines.append("}")
    pg_str = f", pp. {pages}" if pages else ""
    return LitCitation(
        key=key,
        title=title,
        authors=authors,
        venue=booktitle,
        year=year,
        url=url,
        bibtex="\n".join(lines),
        ieee=f'{authors}, "{title}," in {booktitle}{pg_str}, {year}.',
        apa=f"{authors} ({year}). {title}. In {booktitle}" + (f" (pp. {pages})" if pages else "") + ".",
    )


STANDARDS_PROVENANCE: dict[str, StandardRef] = {
    "itur_m2160": _std_ref(
        sid="itur_m2160",
        org="ITU-R",
        doc_id="Recommendation ITU-R M.2160-0",
        section="Usage Scenarios & 15 Capabilities",
        title="Framework and overall objectives of the future development of IMT for 2030 and beyond",
        release="IMT-2030 (6G)",
        year=2023,
        url="https://www.itu.int/rec/R-REC-M.2160-0-202311-I/en",
        summary_plain="Defines the 6 IMT-2030 (6G) usage scenarios (IC, HRLLC, MC, UC, AIAC, ISAC) and 15 capability targets.",
        key_parameters=("peak_rate_50_200_gbps", "latency_0.1_1_ms", "reliability_1e-5_1e-7", "positioning_1_10_cm"),
        bib_key="itur_m2160_2023",
        authors="International Telecommunication Union (ITU-R)",
        venue="ITU-R Recommendation M.2160-0",
    ),
    "itur_p676": _std_ref(
        sid="itur_p676",
        org="ITU-R",
        doc_id="Recommendation ITU-R P.676-13",
        section="Annex 1 Line-by-line Gaseous Attenuation",
        title="Attenuation by atmospheric gases and related effects",
        release="P.676-13",
        year=2022,
        url="https://www.itu.int/rec/R-REC-P.676-13-202208-I/en",
        summary_plain="Standardizes oxygen (O2) and water-vapor (H2O) molecular absorption attenuation gamma(f) up to 1000 GHz (THz).",
        key_parameters=("gamma_d_db_per_km", "gamma_w_db_per_km", "water_vapor_density_g_m3"),
        bib_key="itur_p676_2022",
        authors="International Telecommunication Union (ITU-R)",
        venue="ITU-R Recommendation P.676-13",
    ),
    "gpp_tr38901": _std_ref(
        sid="gpp_tr38901",
        org="3GPP",
        doc_id="3GPP TR 38.901",
        section="Clause 7.4 (Table 7.4.1-1 / 7.4.2-1 / 7.4.3-2) & Clause 7.7",
        title="Study on channel model for frequencies from 0.5 to 100 GHz",
        release="Rel-15..Rel-18",
        year=2024,
        url="https://www.3gpp.org/ftp/Specs/archive/38_series/38.901/",
        summary_plain="Canonical 3D spatial channel model across UMa, UMi, RMa, InH, InF (SL/DL/SH/DH/HH), O2I penetration, and CDL/TDL taps.",
        key_parameters=("pathloss_table_7_4_1_1", "los_prob_table_7_4_2_1", "o2i_low_high_loss", "cdl_tdl_profiles"),
        bib_key="3gpp_tr38901",
        authors="3rd Generation Partnership Project (3GPP)",
        venue="3GPP Technical Report TR 38.901",
    ),
    "gpp_tr38901_isac": _std_ref(
        sid="gpp_tr38901_isac",
        org="3GPP",
        doc_id="3GPP TR 38.901 (Rel-19 ISAC)",
        section="ISAC Background & Target Channel Extension",
        title="Study on channel modelling for Integrated Sensing and Communication (ISAC)",
        release="Rel-19",
        year=2025,
        url="https://www.3gpp.org/ftp/Specs/archive/38_series/38.901/",
        summary_plain="Extends TR 38.901 GSCM with background clutter + target RCS reflection channels for monostatic and bistatic sensing.",
        key_parameters=("monostatic_bistatic_geometry", "target_rcs_dbsm", "bistatic_angle", "background_target_split"),
        bib_key="3gpp_tr38901_isac_rel19",
        authors="3rd Generation Partnership Project (3GPP)",
        venue="3GPP Technical Report TR 38.901 (Rel-19)",
    ),
    "gpp_ts38211": _std_ref(
        sid="gpp_ts38211",
        org="3GPP",
        doc_id="3GPP TS 38.211",
        section="Clause 4.2–4.3 (Numerologies & Frame Structure)",
        title="NR; Physical channels and modulation",
        release="Rel-15..Rel-18",
        year=2024,
        url="https://www.3gpp.org/ftp/Specs/archive/38_series/38.211/",
        summary_plain="Defines scalable NR OFDM numerologies mu=0..4 (15–240 kHz SCS), 10 ms frame, 1 ms subframe, and 1/2^mu ms slot durations.",
        key_parameters=("numerology_mu", "subcarrier_spacing_khz", "slot_duration_ms", "symbols_per_slot_14"),
        bib_key="3gpp_ts38211",
        authors="3rd Generation Partnership Project (3GPP)",
        venue="3GPP Technical Specification TS 38.211",
    ),
    "gpp_tr38913": _std_ref(
        sid="gpp_tr38913",
        org="3GPP",
        doc_id="3GPP TR 38.913",
        section="Clause 6–7 (Scenarios & KPIs)",
        title="Study on scenarios and requirements for next generation access technologies",
        release="Rel-14..Rel-16",
        year=2020,
        url="https://www.3gpp.org/ftp/Specs/archive/38_series/38.913/",
        summary_plain="Baseline 5G NR requirements and evaluation KPIs for eMBB, URLLC, and mMTC.",
        key_parameters=("embb_kpi", "urllc_1ms_1e-5", "mmtc_1e6_per_km2"),
        bib_key="3gpp_tr38913",
        authors="3rd Generation Partnership Project (3GPP)",
        venue="3GPP Technical Report TR 38.913",
    ),
    "gpp_tr38914": _std_ref(
        sid="gpp_tr38914",
        org="3GPP",
        doc_id="3GPP TR 38.914",
        section="Rel-20 6G Feasibility Studies",
        title="Study on 6G Scenarios and Requirements",
        release="Rel-20",
        year=2025,
        url="https://www.3gpp.org/ftp/Specs/archive/38_series/38.914/",
        summary_plain="Maps ITU-R IMT-2030 scenarios (IC, HRLLC, MC, UC, AIAC, ISAC) and FR3 upper mid-band into 3GPP 6G study baselines.",
        key_parameters=("fr3_upper_midband", "6g_scenarios", "imt2030_tpr_alignment"),
        bib_key="3gpp_tr38914_rel20",
        authors="3rd Generation Partnership Project (3GPP)",
        venue="3GPP Technical Report TR 38.914 (Rel-20)",
    ),
    "gpp_tr38811": _std_ref(
        sid="gpp_tr38811",
        org="3GPP",
        doc_id="3GPP TR 38.811",
        section="Clause 6 (NTN Channel Models)",
        title="Study on New Radio (NR) to support non-terrestrial networks",
        release="Rel-15",
        year=2020,
        url="https://www.3gpp.org/ftp/Specs/archive/38_series/38.811/",
        summary_plain="Elevation-angle-dependent NTN channel models (S-band 2 GHz & Ka-band 20–30 GHz) with clutter loss, Ricean K, and LEO Doppler.",
        key_parameters=("elevation_deg", "clutter_loss_db", "leo_altitude_km", "ntn_doppler_hz"),
        bib_key="3gpp_tr38811",
        authors="3rd Generation Partnership Project (3GPP)",
        venue="3GPP Technical Report TR 38.811",
    ),
    "gpp_tr38821": _std_ref(
        sid="gpp_tr38821",
        org="3GPP",
        doc_id="3GPP TR 38.821",
        section="Clause 5–6 (LEO/GEO System Simulation Parameters)",
        title="Solutions for NR to support non-terrestrial networks (NTN)",
        release="Rel-16",
        year=2021,
        url="https://www.3gpp.org/ftp/Specs/archive/38_series/38.821/",
        summary_plain="Reference LEO (600/1200 km) & GEO system parameters, minimum 10–30 deg elevation cutoffs, and orbital Doppler compensation.",
        key_parameters=("leo_600km_1200km", "min_elevation_10deg", "propagation_delay_ms"),
        bib_key="3gpp_tr38821",
        authors="3rd Generation Partnership Project (3GPP)",
        venue="3GPP Technical Report TR 38.821",
    ),
    "gpp_tr36777": _std_ref(
        sid="gpp_tr36777",
        org="3GPP",
        doc_id="3GPP TR 36.777",
        section="Annex B (RMa-AV, UMa-AV, UMi-AV Channel Models)",
        title="Study on enhanced LTE support for aerial vehicles",
        release="Rel-15",
        year=2018,
        url="https://www.3gpp.org/ftp/Specs/archive/36_series/36.777/",
        summary_plain="Standardized aerial/UAV propagation up to 300 m height (RMa-AV, UMa-AV, UMi-AV) with height-dependent LoS probability & shadowing.",
        key_parameters=("uav_height_m", "rma_uma_umi_av_pl", "aerial_los_prob"),
        bib_key="3gpp_tr36777",
        authors="3rd Generation Partnership Project (3GPP)",
        venue="3GPP Technical Report TR 36.777",
    ),
    "gpp_tr37885": _std_ref(
        sid="gpp_tr37885",
        org="3GPP",
        doc_id="3GPP TR 37.885",
        section="Clause 6 (Urban Grid & Highway V2X Channels)",
        title="Study on evaluation methodology of new Vehicle-to-Everything (V2X) use cases for LTE and NR",
        release="Rel-15",
        year=2019,
        url="https://www.3gpp.org/ftp/Specs/archive/37_series/37.885/",
        summary_plain="V2V/V2I channels for Urban Grid (433x250m blocks, NLOSv 15 dB vehicle blockage) and 6-lane Highway (80m breakpoint, dual-mobility).",
        key_parameters=("urban_grid_433x250m", "highway_bp_80m", "nlosv_blockage_15db", "dual_mobility_doppler"),
        bib_key="3gpp_tr37885",
        authors="3rd Generation Partnership Project (3GPP)",
        venue="3GPP Technical Report TR 37.885",
    ),
    "gpp_tr38824": _std_ref(
        sid="gpp_tr38824",
        org="3GPP",
        doc_id="3GPP TR 38.824",
        section="Clause 5–7 (NR URLLC PHY & Multiplexing)",
        title="Study on physical layer enhancements for NR ultra-reliable and low latency case (URLLC)",
        release="Rel-16",
        year=2019,
        url="https://www.3gpp.org/ftp/Specs/archive/38_series/38.824/",
        summary_plain="URLLC physical layer enhancements, mini-slot puncturing over eMBB FTP traffic, and grant-free/HARQ latency reliability bounds.",
        key_parameters=("urllc_bler_1e-5", "minislot_latency", "embb_urllc_multiplexing"),
        bib_key="3gpp_tr38824",
        authors="3rd Generation Partnership Project (3GPP)",
        venue="3GPP Technical Report TR 38.824",
    ),
    "gpp_tr38838": _std_ref(
        sid="gpp_tr38838",
        org="3GPP",
        doc_id="3GPP TR 38.838",
        section="Clause 4–6 (XR Traffic Models & Satisfaction KPIs)",
        title="Study on XR (Extended Reality) evaluations for NR",
        release="Rel-17",
        year=2022,
        url="https://www.3gpp.org/ftp/Specs/archive/38_series/38.838/",
        summary_plain="Periodic bursty XR video frame traffic (30/60/90/120 fps), 10–20 ms PDB, >=99% packet and >=90% user satisfaction KPIs.",
        key_parameters=("xr_fps_60_120", "pdb_10_20_ms", "packet_satisfaction_99pct"),
        bib_key="3gpp_tr38838",
        authors="3rd Generation Partnership Project (3GPP)",
        venue="3GPP Technical Report TR 38.838",
    ),
    "gpp_tr38843": _std_ref(
        sid="gpp_tr38843",
        org="3GPP",
        doc_id="3GPP TR 38.843",
        section="Clause 5–7 (CSI Feedback, Beam Management & Positioning AI/ML)",
        title="Study on Artificial Intelligence (AI)/Machine Learning (ML) for NR air interface",
        release="Rel-18",
        year=2024,
        url="https://www.3gpp.org/ftp/Specs/archive/38_series/38.843/",
        summary_plain="Defines one-sided/two-sided AI/ML evaluation frameworks for CSI compression feedback, FR2 beam prediction, and positioning.",
        key_parameters=("csi_feedback_compression", "beam_management", "ai_positioning"),
        bib_key="3gpp_tr38843",
        authors="3rd Generation Partnership Project (3GPP)",
        venue="3GPP Technical Report TR 38.843",
    ),
    "gpp_tr38848": _std_ref(
        sid="gpp_tr38848",
        org="3GPP",
        doc_id="3GPP TR 38.848",
        section="Clause 4–6 (Ambient IoT Topologies & Link Budget)",
        title="Study on Ambient IoT (Internet of Things) in RAN",
        release="Rel-18",
        year=2023,
        url="https://www.3gpp.org/ftp/Specs/archive/38_series/38.848/",
        summary_plain="Batteryless / uW-scale RF energy-harvesting backscatter topologies (BS-to-Device and intermediate helper two-hop budgets).",
        key_parameters=("topology_1_direct", "topology_2_helper", "backscatter_two_hop_budget_uw"),
        bib_key="3gpp_tr38848",
        authors="3rd Generation Partnership Project (3GPP)",
        venue="3GPP Technical Report TR 38.848",
    ),
    "gpp_tr38855": _std_ref(
        sid="gpp_tr38855",
        org="3GPP",
        doc_id="3GPP TR 38.855",
        section="Clause 5–7 (NR Positioning Accuracy & PRS)",
        title="Study on NR positioning support",
        release="Rel-16",
        year=2019,
        url="https://www.3gpp.org/ftp/Specs/archive/38_series/38.855/",
        summary_plain="OTDOA, UTDOA, Multi-RTT, and AoA/AoD positioning procedures and ToA resolution metrics.",
        key_parameters=("otdoa_utdoa_multi_rtt", "prs_bandwidth", "toa_resolution"),
        bib_key="3gpp_tr38855",
        authors="3rd Generation Partnership Project (3GPP)",
        venue="3GPP Technical Report TR 38.855",
    ),
    "gpp_tr22837": _std_ref(
        sid="gpp_tr22837",
        org="3GPP",
        doc_id="3GPP TR 22.837",
        section="Clause 5–6 (ISAC Use Cases & Sensing KPIs)",
        title="Feasibility study on Integrated Sensing and Communication (ISAC)",
        release="Rel-19",
        year=2024,
        url="https://www.3gpp.org/ftp/Specs/archive/22_series/22.837/",
        summary_plain="Stage 1 sensing service requirements (range/velocity/angle accuracy, detection/false-alarm rates) for cellular ISAC.",
        key_parameters=("sensing_range_res", "velocity_res", "detection_prob", "false_alarm_rate"),
        bib_key="3gpp_tr22837",
        authors="3rd Generation Partnership Project (3GPP)",
        venue="3GPP Technical Report TR 22.837",
    ),
    "gpp_tr36814": _std_ref(
        sid="gpp_tr36814",
        org="3GPP",
        doc_id="3GPP TR 36.814",
        section="Annex A.2.1.3 (FTP Model 1/2/3 & VoIP Traffic Models)",
        title="Evolved Universal Terrestrial Radio Access (E-UTRA); Further advancements for E-UTRA physical layer aspects",
        release="Rel-9",
        year=2010,
        url="https://www.3gpp.org/ftp/Specs/archive/36_series/36.814/",
        summary_plain="Defines FTP Models 1 (Poisson sessions, 2MB), 2 (0.5MB + reading time), 3 (Poisson packets) and On-Off VoIP (20ms/160ms SID).",
        key_parameters=("ftp_model_1_2_3", "file_size_0.5mb_2mb", "voip_20ms_vaf50"),
        bib_key="3gpp_tr36814",
        authors="3rd Generation Partnership Project (3GPP)",
        venue="3GPP Technical Report TR 36.814",
    ),
    "etsi_gr_ris_003": _std_ref(
        sid="etsi_gr_ris_003",
        org="ETSI",
        doc_id="ETSI GR RIS 003",
        section="Clauses 4–6 (RIS Cascaded & Near-Field Channel Models)",
        title="Reconfigurable Intelligent Surfaces (RIS); Communication Models, Channel Models, Channel Estimation and Evaluation Methodology",
        release="V1.1.1",
        year=2023,
        url="https://www.etsi.org/deliver/etsi_gr/RIS/001_099/003/",
        summary_plain="Adapts TR 38.901 for BS-RIS-UE two-hop cascaded channels, near-field spherical wavefronts, and unit-cell reflection laws.",
        key_parameters=("cascaded_two_hop", "unit_cell_amplitude_phase", "nearfield_spherical_ris"),
        bib_key="etsi_gr_ris_003",
        authors="ETSI Industry Specification Group on RIS (ISG RIS)",
        venue="ETSI Group Report GR RIS 003",
    ),
    "ieee_80211bf": _std_ref(
        sid="ieee_80211bf",
        org="IEEE",
        doc_id="IEEE P802.11bf (TGbf)",
        section="TGbf Channel Models (11-21/0782)",
        title="Wireless LAN Medium Access Control (MAC) and Physical Layer (PHY) Specifications Amendment: Enhancements for Wireless LAN Sensing",
        release="802.11bf",
        year=2024,
        url="https://standards.ieee.org/ieee/802.11bf/10557/",
        summary_plain="Standardizes WLAN CSI sensing measurement exchange and dynamic multipath scattering separation across sub-7 GHz & 60 GHz.",
        key_parameters=("static_vs_dynamic_paths", "micro_doppler_csi", "sub7ghz_and_60ghz"),
        bib_key="ieee_80211bf",
        authors="IEEE 802.11 Working Group (TGbf)",
        venue="IEEE Std 802.11bf",
    ),
    "ieee_80211bn": _std_ref(
        sid="ieee_80211bn",
        org="IEEE",
        doc_id="IEEE P802.11bn (UHR / Wi-Fi 8)",
        section="Multi-AP Coordination & Reliability Targets",
        title="Wireless LAN Medium Access Control (MAC) and Physical Layer (PHY) Specifications Amendment: Enhancements for Ultra High Reliability",
        release="802.11bn",
        year=2025,
        url="https://www.ieee802.org/11/Reports/tgbn_update.htm",
        summary_plain="Ultra-High Reliability Wi-Fi 8 targeting 25% lower p95 tail latency & packet loss via Multi-AP Coordinated Spatial Reuse (Co-SR).",
        key_parameters=("multi_ap_cosr", "tail_latency_95pct_reduction", "low_sinr_resilience"),
        bib_key="ieee_80211bn",
        authors="IEEE 802.11 Working Group (TGbn)",
        venue="IEEE P802.11bn Draft",
    ),
}


MECHANISM_PROVENANCE: dict[str, MechanismProvenance] = {
    "shannon_awgn": MechanismProvenance(
        id="shannon_awgn",
        name="Shannon AWGN Capacity & Complex Baseband Thermal Noise",
        role_plain="Establishes ideal capacity C = B * log2(1 + SNR) and thermal noise floor N0*B*NF.",
        citation=_article_cit(
            key="shannon1948mathematical",
            title="A Mathematical Theory of Communication",
            authors="Claude E. Shannon",
            journal="The Bell System Technical Journal",
            volume="27",
            pages="379-423, 623-656",
            year=1948,
            url="https://ieeexplore.ieee.org/document/6773024",
        ),
    ),
    "clarke_jakes_doppler": MechanismProvenance(
        id="clarke_jakes_doppler",
        name="Clarke/Jakes Sum-of-Sinusoids Temporal Doppler Autocorrelation",
        role_plain="Generates temporally consistent mobile fading honoring J0(2*pi*f_d*dt) autocorrelation.",
        citation=_article_cit(
            key="clarke1968statistical",
            title="A Statistical Theory of Mobile-Radio Reception",
            authors="R. H. Clarke",
            journal="The Bell System Technical Journal",
            volume="47",
            number="6",
            pages="957-1000",
            year=1968,
            url="https://ieeexplore.ieee.org/document/6771005",
        ),
    ),
    "saleh_valenzuela_cluster": MechanismProvenance(
        id="saleh_valenzuela_cluster",
        name="Saleh-Valenzuela Multi-Cluster Sparse Multipath",
        role_plain="Models clustered geometric multipath arrivals/departures foundational for mmWave/THz MIMO.",
        citation=_article_cit(
            key="saleh1987statistical",
            title="A Statistical Model for Indoor Multipath Propagation",
            authors="A. A. M. Saleh and R. A. Valenzuela",
            journal="IEEE Journal on Selected Areas in Communications",
            volume="5",
            number="2",
            pages="128-137",
            year=1987,
            url="https://ieeexplore.ieee.org/document/1146527",
        ),
    ),
    "ppp_stochastic_geometry": MechanismProvenance(
        id="ppp_stochastic_geometry",
        name="Homogeneous Poisson Point Process (PPP) Cellular Coverage",
        role_plain="Models random BS topologies via PPP density lambda and closed-form SINR outage benchmark.",
        citation=_article_cit(
            key="andrews2011tractable",
            title="A Tractable Approach to Coverage and Rate in Cellular Networks",
            authors="Jeffrey G. Andrews, Francois Baccelli, and Radha Krishna Ganti",
            journal="IEEE Transactions on Communications",
            volume="59",
            number="11",
            pages="3122-3134",
            year=2011,
            url="https://arxiv.org/abs/1009.0516",
        ),
    ),
    "tr38901_pathloss_los": MechanismProvenance(
        id="tr38901_pathloss_los",
        name="3GPP TR 38.901 Dual-Slope Breakpoint Pathloss & LoS Probability",
        role_plain="Applies exact Table 7.4.1-1 pathloss, shadow fading sigma_SF, and Table 7.4.2-1 LoS probability.",
        standard_id="gpp_tr38901",
        citation=STANDARDS_PROVENANCE["gpp_tr38901"].citation,
    ),
    "tr38901_o2i_penetration": MechanismProvenance(
        id="tr38901_o2i_penetration",
        name="3GPP TR 38.901 O2I Low-Loss / High-Loss Penetration",
        role_plain="Adds external glass/IRR-glass/concrete wall loss plus indoor depth attenuation 0.5*d_2D_in.",
        standard_id="gpp_tr38901",
        citation=STANDARDS_PROVENANCE["gpp_tr38901"].citation,
    ),
    "nr_ofdm_numerology": MechanismProvenance(
        id="nr_ofdm_numerology",
        name="3GPP TS 38.211 Scalable OFDM Numerology",
        role_plain="Maps mu=0..4 to subcarrier spacing 15*2^mu kHz and slot duration 1/2^mu ms.",
        standard_id="gpp_ts38211",
        citation=STANDARDS_PROVENANCE["gpp_ts38211"].citation,
    ),
    "al_hourani_a2g": MechanismProvenance(
        id="al_hourani_a2g",
        name="Al-Hourani Elevation-Angle Probabilistic A2G LoS",
        role_plain="Computes elevation-dependent UAV LoS probability P_LoS(theta) and excess NLoS attenuation.",
        citation=_article_cit(
            key="alhourani2014optimal",
            title="Optimal LAP Altitude for Maximum Coverage",
            authors="Akram Al-Hourani, Sithamparanathan Kandeepan, and Simon Lardner",
            journal="IEEE Wireless Communications Letters",
            volume="3",
            number="6",
            pages="569-572",
            year=2014,
            url="https://ieeexplore.ieee.org/document/6863654",
        ),
    ),
    "polyanskiy_fbl_dispersion": MechanismProvenance(
        id="polyanskiy_fbl_dispersion",
        name="Finite Blocklength (FBL) Channel Dispersion Capacity",
        role_plain="Penalizes Shannon rate with sqrt(V/n)*Q^{-1}(epsilon) for short URLLC packets.",
        citation=_article_cit(
            key="polyanskiy2010channel",
            title="Channel Coding Rate in the Finite Blocklength Regime",
            authors="Yury Polyanskiy, H. Vincent Poor, and Sergio Verdu",
            journal="IEEE Transactions on Information Theory",
            volume="56",
            number="5",
            pages="2307-2359",
            year=2010,
            url="https://ieeexplore.ieee.org/document/5452208",
        ),
    ),
    "wu_zhang_passive_ris": MechanismProvenance(
        id="wu_zhang_passive_ris",
        name="Passive RIS Two-Hop Cascade & Unit-Modulus Constraint",
        role_plain="Models direct + G*Phi*H_r cascade enforcing passive |phi_m|=1 energy conservation.",
        standard_id="etsi_gr_ris_003",
        citation=_article_cit(
            key="wu2019intelligent",
            title="Intelligent Reflecting Surface Enhanced Wireless Network via Joint Active and Passive Beamforming",
            authors="Qingqing Wu and Rui Zhang",
            journal="IEEE Transactions on Wireless Communications",
            volume="18",
            number="11",
            pages="5394-5409",
            year=2019,
            doi="10.1109/TWC.2019.2936025",
            url="https://doi.org/10.1109/TWC.2019.2936025",
        ),
    ),
    "star_ris_energy_split": MechanismProvenance(
        id="star_ris_energy_split",
        name="STAR-RIS Full-Space Transmission/Reflection Energy Split",
        role_plain="Splits incident power into 360-deg transmitted & reflected waves with |beta_T|^2+|beta_R|^2<=1.",
        citation=_article_cit(
            key="xu2021starriss",
            title="STAR-RISs: Simultaneous Transmitting and Reflecting Reconfigurable Intelligent Surfaces",
            authors="Jiaqi Xu, Yuanwei Liu, Xidong Mu, and Octavia A. Dobre",
            journal="IEEE Communications Letters",
            volume="25",
            number="9",
            pages="3134-3138",
            year=2021,
            url="https://arxiv.org/abs/2101.08705",
        ),
    ),
    "active_ris_noise": MechanismProvenance(
        id="active_ris_noise",
        name="Active RIS Amplification & Dynamic Thermal Noise Modeling",
        role_plain="Includes active reflection power budget and dynamic RIS amplifier noise G*Phi*n_v.",
        citation=_article_cit(
            key="zhang2023active",
            title="Active RIS vs. Passive RIS: Which Will Prevail in 6G?",
            authors="Zijian Zhang, Linglong Dai, Xibi Chen, Changhao Liu, Fan Yang, Robert Schober, and H. Vincent Poor",
            journal="IEEE Transactions on Communications",
            volume="71",
            number="3",
            pages="1707-1725",
            year=2023,
            doi="10.1109/TCOMM.2022.3231893",
            url="https://doi.org/10.1109/TCOMM.2022.3231893",
        ),
    ),
    "cui_dai_nearfield_spherical": MechanismProvenance(
        id="cui_dai_nearfield_spherical",
        name="Near-Field Spherical Wavefront & Rayleigh Distance Regime",
        role_plain="Switches from planar steering to exact distance-angle spherical steering when d < 2*D^2/lambda.",
        citation=_article_cit(
            key="cui2022channel",
            title="Channel Estimation for Extremely Large-Scale MIMO: Far-Field or Near-Field?",
            authors="Mingyao Cui and Linglong Dai",
            journal="IEEE Transactions on Communications",
            volume="70",
            number="4",
            pages="2663-2677",
            year=2022,
            doi="10.1109/TCOMM.2022.3146400",
            url="https://doi.org/10.1109/TCOMM.2022.3146400",
        ),
    ),
    "itur_p676_thz_absorption": MechanismProvenance(
        id="itur_p676_thz_absorption",
        name="ITU-R P.676 Atmospheric Gas & Water Vapor Absorption",
        role_plain="Applies frequency-selective O2 + H2O molecular absorption loss for mmWave/sub-THz links.",
        standard_id="itur_p676",
        citation=STANDARDS_PROVENANCE["itur_p676"].citation,
    ),
    "wong_fluid_antenna": MechanismProvenance(
        id="wong_fluid_antenna",
        name="Fluid Antenna Spatial Port Diversity (FAS)",
        role_plain="Models tightly correlated spatial ports within a compact aperture for fading diversity.",
        citation=_article_cit(
            key="wong2021fluid",
            title="Fluid Antenna Systems",
            authors="Kai-Kit Wong, Arman Shojaeifard, Kin-Fai Tong, and Yangyang Zhang",
            journal="IEEE Transactions on Wireless Communications",
            volume="20",
            number="3",
            pages="1950-1962",
            year=2021,
            url="https://ieeexplore.ieee.org/document/9264694",
        ),
    ),
    "ding_pinching_antenna": MechanismProvenance(
        id="ding_pinching_antenna",
        name="Pinching-Antenna Waveguide LoS Reconfiguration (PASS)",
        role_plain="Models leaky-waveguide dielectric pinch placement to establish short strong LoS paths.",
        citation=_article_cit(
            key="ding2025flexible",
            title="Flexible-Antenna Systems: A Pinching-Antenna Perspective",
            authors="Zhiguo Ding, Robert Schober, and H. Vincent Poor",
            journal="arXiv preprint arXiv:2412.02376",
            year=2025,
            url="https://arxiv.org/abs/2412.02376",
        ),
    ),
    "liu_isac_crb": MechanismProvenance(
        id="liu_isac_crb",
        name="ISAC Target Echo RCS & Cramér-Rao Bound (CRB)",
        role_plain="Jointly models target RCS reflection echoes and estimation CRB alongside communication rate.",
        standard_id="gpp_tr38901_isac",
        citation=_article_cit(
            key="liu2022integrated",
            title="Integrated Sensing and Communications: Toward Dual-Functional Wireless Networks for 6G and Beyond",
            authors="Fan Liu, Yuanhao Cui, Christos Masouros, Jie Xu, Tony Xiao Han, Yonina C. Eldar, and Stefano Buzzi",
            journal="IEEE Journal on Selected Areas in Communications",
            volume="40",
            number="6",
            pages="1728-1767",
            year=2022,
            doi="10.1109/JSAC.2022.3156632",
            url="https://doi.org/10.1109/JSAC.2022.3156632",
        ),
    ),
    "hadani_otfs_delay_doppler": MechanismProvenance(
        id="hadani_otfs_delay_doppler",
        name="Delay-Doppler OTFS Channel Representation",
        role_plain="Maps doubly-dispersive high-mobility channels into stable 2D delay-Doppler taps via SFFT/ISFFT.",
        citation=_inproceedings_cit(
            key="hadani2017orthogonal",
            title="Orthogonal Time Frequency Space Modulation",
            authors="Ronny Hadani, Shlomo Rakib, Michail Tsatsanis, Anton Monk, Andrea J. Goldsmith, A. F. Molisch, and Robert Calderbank",
            booktitle="2017 IEEE Wireless Communications and Networking Conference (WCNC)",
            year=2017,
            doi="10.1109/WCNC.2017.7925924",
            url="https://doi.org/10.1109/WCNC.2017.7925924",
        ),
    ),
    "xie_deepsc_semantic": MechanismProvenance(
        id="xie_deepsc_semantic",
        name="Differentiable End-to-End Semantic Communication Baseband",
        role_plain="Provides differentiable AWGN/fading physical layers for joint semantic-channel optimization.",
        citation=_article_cit(
            key="xie2021deep",
            title="Deep Learning Enabled Semantic Communication Systems",
            authors="Huiqiang Xie, Zhijin Qin, Geoffrey Ye Li, and Biing-Hwang Juang",
            journal="IEEE Transactions on Signal Processing",
            volume="69",
            pages="2663-2675",
            year=2021,
            url="https://arxiv.org/abs/2006.10685",
        ),
    ),
    "zhu_aircomp_feel": MechanismProvenance(
        id="zhu_aircomp_feel",
        name="Broadband Analog Over-the-Air Computation (AirComp)",
        role_plain="Simulates waveform superposition of concurrent transmitters for federated edge aggregation.",
        citation=_article_cit(
            key="zhu2020broadband",
            title="Broadband Analog Aggregation for Low-Latency Federated Edge Learning",
            authors="Guangxu Zhu, Yong Wang, and Kaibin Huang",
            journal="IEEE Transactions on Wireless Communications",
            volume="19",
            number="1",
            pages="491-506",
            year=2020,
            url="https://arxiv.org/abs/1812.11494",
        ),
    ),
    "ngo_cell_free_mimo": MechanismProvenance(
        id="ngo_cell_free_mimo",
        name="Distributed Cell-Free Massive MIMO Macro-Diversity",
        role_plain="Replaces cell boundaries with distributed cooperating APs serving users jointly.",
        citation=_article_cit(
            key="ngo2017cellfree",
            title="Cell-Free Massive MIMO Versus Small Cells",
            authors="Hien Quoc Ngo, Alexei Ashikhmin, Hong Yang, Erik G. Larsson, and Thomas L. Marzetta",
            journal="IEEE Transactions on Wireless Communications",
            volume="16",
            number="3",
            pages="1834-1850",
            year=2017,
            url="https://arxiv.org/abs/1602.08232",
        ),
    ),
    "wyner_wiretap_secrecy": MechanismProvenance(
        id="wyner_wiretap_secrecy",
        name="Physical-Layer Wiretap Secrecy Capacity",
        role_plain="Computes positive secrecy rate C_s = max(C_Bob - C_Eve, 0) against eavesdroppers.",
        citation=_article_cit(
            key="wyner1975wiretap",
            title="The Wire-Tap Channel",
            authors="Aaron D. Wyner",
            journal="The Bell System Technical Journal",
            volume="54",
            number="8",
            pages="1355-1387",
            year=1975,
            url="https://ieeexplore.ieee.org/document/6772207",
        ),
    ),
    "bash_covert_sqrt_law": MechanismProvenance(
        id="bash_covert_sqrt_law",
        name="Covert Communications KL-Divergence Square-Root Law",
        role_plain="Evaluates warden detection error probability / KL divergence over n channel uses.",
        citation=_article_cit(
            key="bash2013limits",
            title="Limits of Reliable Communication with Low Probability of Detection on AWGN Channels",
            authors="Boulat A. Bash, Dennis Goeckel, and Don Towsley",
            journal="IEEE Journal on Selected Areas in Communications",
            volume="31",
            number="9",
            pages="1921-1930",
            year=2013,
            url="https://arxiv.org/abs/1202.6423",
        ),
    ),
    "sadeghi_rf_adversarial": MechanismProvenance(
        id="sadeghi_rf_adversarial",
        name="Over-the-Air Baseband Adversarial & JSR Jamming Interface",
        role_plain="Exposes power-constrained JSR-dB perturbations aligned with wannado-adv.",
        citation=_article_cit(
            key="sadeghi2019adversarial",
            title="Adversarial Attacks on Deep-Learning Based Radio Signal Classification",
            authors="Meysam Sadeghi and Erik G. Larsson",
            journal="IEEE Wireless Communications Letters",
            volume="8",
            number="1",
            pages="213-216",
            year=2019,
            url="https://arxiv.org/abs/1809.00051",
        ),
    ),
    "shi_wmmse_duality": MechanismProvenance(
        id="shi_wmmse_duality",
        name="Iteratively Weighted MMSE (WMMSE) Interference Beamforming",
        role_plain="Canonical iterative baseline optimizing multi-user MIMO sum-rate under power constraints.",
        citation=_article_cit(
            key="shi2011iteratively",
            title="An Iteratively Weighted MMSE Approach to Distributed Sum-Utility Maximization for a MIMO Interfering Broadcast Channel",
            authors="Qingjiang Shi, Meisam Razaviyayn, Zhi-Quan Luo, and Chen He",
            journal="IEEE Transactions on Signal Processing",
            volume="59",
            number="9",
            pages="4331-4340",
            year=2011,
            url="https://ieeexplore.ieee.org/document/5756489",
        ),
    ),
    "jalali_proportional_fair": MechanismProvenance(
        id="jalali_proportional_fair",
        name="Proportional-Fair (PF) Throughput-Fairness Scheduling",
        role_plain="Classic baseline weighting instantaneous rate against exponentially averaged past throughput.",
        citation=_inproceedings_cit(
            key="jalali2000data",
            title="Data Throughput of CDMA-HDR a High Efficiency-High Data Rate Personal Communication Wireless System",
            authors="Ahmad Jalali, Roberto Padovani, and Rajesh Pankaj",
            booktitle="2000 IEEE 51st Vehicular Technology Conference Proceedings (VTC2000-Spring)",
            pages="1854-1858",
            year=2000,
            url="https://ieeexplore.ieee.org/document/851593",
        ),
    ),
    "zeng_ckm_prior": MechanismProvenance(
        id="zeng_ckm_prior",
        name="Environment-Aware Channel Knowledge Map (CKM)",
        role_plain="Links 3D coordinates directly to location-specific channel gain and multipath priors.",
        citation=_article_cit(
            key="zeng2021toward",
            title="Toward Environment-Aware 6G Communications via Channel Knowledge Map",
            authors="Yong Zeng and Xiaoli Xu",
            journal="IEEE Wireless Communications",
            volume="28",
            number="3",
            pages="84-91",
            year=2021,
            url="https://arxiv.org/abs/2101.07497",
        ),
    ),
    "fu_ckmdiff_diffusion": MechanismProvenance(
        id="fu_ckmdiff_diffusion",
        name="Conditional Diffusion Radio Map / CKM Inpainting",
        role_plain="Reconstructs full spatial channel maps from sparse noisy pilot observations.",
        citation=_article_cit(
            key="fu2025ckmdiff",
            title="CKMDiff: A Generative Diffusion Model for CKM Construction via Inverse Problems with Learned Priors",
            authors="Shen Fu, Yong Zeng, Zijian Wu, Di Wu, Shi Jin, Cheng-Xiang Wang, and Xiqi Gao",
            journal="arXiv preprint arXiv:2504.17323",
            year=2025,
            url="https://arxiv.org/abs/2504.17323",
        ),
    ),
    "zhao_nerf2_rf": MechanismProvenance(
        id="zhao_nerf2_rf",
        name="Neural Radio-Frequency Radiance Field (NeRF2) Spatial Synthesis",
        role_plain="Continuous 3D RF field interpolation capturing multipath amplitude & phase across space.",
        citation=_inproceedings_cit(
            key="zhao2023nerf2",
            title="NeRF2: Neural Radio-Frequency Radiance Fields",
            authors="Xiaopeng Zhao, Zhenlin An, Qingrui Pan, and Lei Yang",
            booktitle="Proceedings of the 29th Annual International Conference on Mobile Computing and Networking (ACM MobiCom '23)",
            pages="393-407",
            year=2023,
            url="https://dl.acm.org/doi/10.1145/3570361.3592527",
        ),
    ),
}


BLOCK_DEFAULT_MECHANISMS: dict[str, tuple[str, ...]] = {
    "awgn_link": ("shannon_awgn",),
    "rayleigh_clarke": ("shannon_awgn", "clarke_jakes_doppler"),
    "rician_los": ("shannon_awgn", "clarke_jakes_doppler"),
    "nakagami_m": ("shannon_awgn",),
    "saleh_valenzuela": ("saleh_valenzuela_cluster",),
    "ppp_stochastic_geo": ("ppp_stochastic_geometry", "shannon_awgn"),
    "tr38901_uma_umi_rma": ("tr38901_pathloss_los", "tr38901_o2i_penetration"),
    "tr38901_inf": ("tr38901_pathloss_los",),
    "tr38901_cdl_tdl": ("nr_ofdm_numerology", "clarke_jakes_doppler"),
    "ntn_leo_satellite": ("shannon_awgn",),
    "uav_air_to_ground": ("al_hourani_a2g",),
    "v2x_vehicular_sl": ("clarke_jakes_doppler",),
    "farfield_upa": ("saleh_valenzuela_cluster",),
    "nearfield_xlmimo_thz": ("cui_dai_nearfield_spherical", "itur_p676_thz_absorption"),
    "passive_ris_cascade": ("wu_zhang_passive_ris",),
    "star_ris": ("star_ris_energy_split",),
    "active_ris_flexible": ("active_ris_noise", "wong_fluid_antenna", "ding_pinching_antenna"),
    "traffic_ftp_models": (),
    "traffic_poisson_pareto_voip": (),
    "traffic_xr_immersive": (),
    "multicell_interference_mac": ("tr38901_pathloss_los", "jalali_proportional_fair"),
    "isac_jsac_channel": ("liu_isac_crb",),
    "wifi_sensing_80211bf": (),
    "otfs_delay_doppler": ("hadani_otfs_delay_doppler", "clarke_jakes_doppler"),
    "hrllc_fbl_urllc": ("polyanskiy_fbl_dispersion", "shannon_awgn"),
    "ambient_iot_backscatter": ("shannon_awgn",),
    "semantic_aircomp_channel": ("xie_deepsc_semantic", "zhu_aircomp_feel", "ngo_cell_free_mimo"),
    "pls_covert_antijamming": ("wyner_wiretap_secrecy", "bash_covert_sqrt_law", "sadeghi_rf_adversarial"),
    "water_filling": ("shannon_awgn",),
    "wmmse_precoder": ("shi_wmmse_duality",),
    "ls_lmmse_csi": ("shannon_awgn",),
    "proportional_fair": ("jalali_proportional_fair",),
    "ckm_radiomap_bridge": ("zeng_ckm_prior", "fu_ckmdiff_diffusion", "zhao_nerf2_rf"),
}


def get_standard_ref(standard_id: str) -> StandardRef:
    """Lookup an authoritative standard reference or raise KeyError (never fabricate)."""
    if standard_id not in STANDARDS_PROVENANCE:
        raise KeyError(f"Unknown standard_id '{standard_id}'. Available: {sorted(STANDARDS_PROVENANCE.keys())}")
    return STANDARDS_PROVENANCE[standard_id]


def resolve_block_provenance(
    block_id: str,
    *,
    standard_ids: tuple[str, ...] = (),
    explicit_mechanisms: list[str] | None = None,
) -> tuple[list[StandardRef], list[MechanismProvenance]]:
    """Resolve exact StandardRefs and MechanismProvenance cards active for a provisioned block."""
    stds: list[StandardRef] = []
    seen_stds: set[str] = set()
    for sid in standard_ids:
        if sid in STANDARDS_PROVENANCE and sid not in seen_stds:
            seen_stds.add(sid)
            stds.append(STANDARDS_PROVENANCE[sid])

    mechs: list[MechanismProvenance] = []
    seen_mechs: set[str] = set()
    for mid in list(BLOCK_DEFAULT_MECHANISMS.get(block_id, ())) + list(explicit_mechanisms or ()):
        if mid in MECHANISM_PROVENANCE and mid not in seen_mechs:
            seen_mechs.add(mid)
            m = MECHANISM_PROVENANCE[mid]
            mechs.append(m)
            if m.standard_id and m.standard_id in STANDARDS_PROVENANCE and m.standard_id not in seen_stds:
                seen_stds.add(m.standard_id)
                stds.append(STANDARDS_PROVENANCE[m.standard_id])
    return stds, mechs
