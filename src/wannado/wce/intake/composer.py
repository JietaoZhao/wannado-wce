"""4-Slot Compositional Wireless Environment Assembler & LiveProvenance Diff Engine (v1.1)."""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from wannado.wce.blocks import BLOCK_ID_TO_RUNNER
from wannado.wce.catalog.build_catalog import load_catalog
from wannado.wce.core.provenance import MECHANISM_PROVENANCE, STANDARDS_PROVENANCE
from wannado.wce.core.types import ParamProvenance
from wannado.wce.core.units import rayleigh_distance_m, wavelength_m
from wannado.wce.intake.archetypes import audit_wce_spec
from wannado.wce.intake.spec import WCESpec


@dataclass
class ComposedSlotDecision:
    """Selection record for one orthogonal layer of the 4-slot wireless pipeline."""

    slot_name: str
    blocks: list[str]
    rationale: str
    triggered_rules: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SpecDiffReport:
    """LiveProvenance Diff output produced when modifying WCESpec knobs dynamically."""

    before_physics: dict[str, float]
    after_physics: dict[str, float]
    changed_fields: dict[str, dict[str, Any]]
    boundary_alerts: list[str]
    auto_promoted_blocks: list[str]
    auto_added_baselines: list[str]
    auto_updated_standards: list[str]
    audit_ok: bool
    audit_warnings: list[str]
    updated_spec: WCESpec
    updated_table1_md: str
    updated_table1_tex: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "before_physics": self.before_physics,
            "after_physics": self.after_physics,
            "changed_fields": self.changed_fields,
            "boundary_alerts": self.boundary_alerts,
            "auto_promoted_blocks": self.auto_promoted_blocks,
            "auto_added_baselines": self.auto_added_baselines,
            "auto_updated_standards": self.auto_updated_standards,
            "audit_ok": self.audit_ok,
            "audit_warnings": self.audit_warnings,
            "updated_spec": self.updated_spec.to_dict(),
        }

    def format_summary(self) -> str:
        lines = ["[wannado-wce LiveProvenance Diff]"]
        for k, delta in self.changed_fields.items():
            lines.append(f"  * {k}: `{delta['before']}` -> `{delta['after']}`")
        if self.boundary_alerts:
            lines.append("  Boundary Alerts:")
            lines.extend(f"    ! {a}" for a in self.boundary_alerts)
        if self.auto_promoted_blocks:
            lines.append(f"  Auto-Promoted Blocks : {', '.join(self.auto_promoted_blocks)}")
        if self.auto_added_baselines:
            lines.append(f"  Auto-Added Baselines : {', '.join(self.auto_added_baselines)}")
        if self.auto_updated_standards:
            lines.append(f"  Synced Standards     : {', '.join(self.auto_updated_standards)}")
        lines.append(
            f"  Derived Physics Shift: Z_R={self.before_physics['rayleigh_distance_m']:.2f}m -> "
            f"{self.after_physics['rayleigh_distance_m']:.2f}m | "
            f"T_c={self.before_physics['coherence_time_ms']:.2f}ms -> {self.after_physics['coherence_time_ms']:.2f}ms"
        )
        return "\n".join(lines)


def _prov(param: str, val: Any, unit: str, plain: str, auth_id: str, section: str, rationale: str) -> ParamProvenance:
    if auth_id in STANDARDS_PROVENANCE:
        ckey = STANDARDS_PROVENANCE[auth_id].citation.key
    elif auth_id in MECHANISM_PROVENANCE:
        ckey = MECHANISM_PROVENANCE[auth_id].citation.key
    else:
        ckey = auth_id
    return ParamProvenance(
        parameter=param,
        value=val,
        unit=unit,
        plain_meaning=plain,
        authority_id=auth_id,
        authority_section=section,
        citation_key=ckey,
        rationale=rationale,
    )


def _extract_numeric_clues(query: str) -> dict[str, Any]:
    """Parse carrier frequency, bandwidth, antenna/RIS count, speed, and XR fps from NL query."""
    q = query.lower()
    extracted: dict[str, Any] = {}

    m_fc = re.search(r"(\d+(?:\.\d+)?)\s*(ghz|mhz|thz)", q)
    if m_fc:
        val, unit = float(m_fc.group(1)), m_fc.group(2)
        if unit == "ghz":
            extracted["carrier_fc_ghz"] = val
        elif unit == "thz":
            extracted["carrier_fc_ghz"] = val * 1000.0
        elif unit == "mhz" and ("carrier" in q or "频" in q or "band" in q or val >= 400.0):
            extracted["carrier_fc_ghz"] = val / 1000.0

    m_bw = re.search(r"(?:bw|bandwidth|带宽|频宽)[^0-9]{0,8}(\d+(?:\.\d+)?)\s*(mhz|ghz|khz)", q)
    if not m_bw:
        # Second frequency-unit token if carrier already matched
        tokens = list(re.finditer(r"(\d+(?:\.\d+)?)\s*(mhz|ghz)", q))
        if len(tokens) >= 2:
            val, unit = float(tokens[1].group(1)), tokens[1].group(2)
            extracted["bandwidth_hz"] = val * (1e9 if unit == "ghz" else 1e6)
    else:
        val, unit = float(m_bw.group(1)), m_bw.group(2)
        scale = {"khz": 1e3, "mhz": 1e6, "ghz": 1e9}[unit]
        extracted["bandwidth_hz"] = val * scale

    # Extract RIS/IRS/STAR-RIS element counts SEPARATELY from BS/UE antenna counts (no conflation or 4x inflation)
    m_ris = re.search(
        r"(\d+)\s*(?:-|\s+)?(?:(?:passive|active|star|reflecting)\s+)?(?:elements?|元|个|阵元)?\s*(?:-|\s+)?"
        r"(?:star[-_ ]?ris|active[-_ ]?ris|ris|irs|超表面|反射面)",
        q,
    ) or re.search(
        r"(?:star[-_ ]?ris|active[-_ ]?ris|ris|irs|超表面|反射面)(?:\s+to|\s+with|\s+of)?\s+(\d+)\s*(?:-|\s+)?(?:elements?|元|个|阵元)?",
        q,
    )
    if m_ris:
        extracted["num_ris_elements"] = max(4, min(4096, int(m_ris.group(1))))

    m_ant = re.search(
        r"(\d+)\s*(?:[x×]\s*(\d+)\s*)?(?:-|\s+)?(?:bs|tx|rx|base\s+station)?\s*(?:-|\s+)?"
        r"(?:antennas?|ant|tx|mimo|天线|xl-mimo|elaa)",
        q,
    )
    if m_ant:
        extracted["num_tx"] = max(1, min(2048, int(m_ant.group(1))))
        if m_ant.group(2):
            extracted["num_rx"] = max(1, min(256, int(m_ant.group(2))))

    m_bits = re.search(r"(\d+)\s*(?:-|\s+)?bits?(?:\s+(?:ris|irs|phase|quantiz))?", q)
    if m_bits and any(k in q for k in ("ris", "irs", "phase", "quantiz", "超表面")):
        extracted["ris_phase_bits"] = max(1, min(8, int(m_bits.group(1))))

    m_alt = re.search(r"(?:altitude|height|alt|高度|悬停)[^0-9]{0,12}(\d+(?:\.\d+)?)\s*m\b", q)
    if m_alt:
        extracted["uav_altitude_m"] = float(m_alt.group(1))

    m_el = re.search(r"(?:elevation|仰角)[^0-9]{0,12}(\d+(?:\.\d+)?)\s*(?:deg|°)?", q)
    if m_el:
        extracted["ntn_elevation_deg"] = float(m_el.group(1))

    m_orb = re.search(r"(\d+(?:\.\d+)?)\s*km\s*(?:leo|orbit|altitude|satellite)", q)
    if m_orb:
        extracted["ntn_orbit_altitude_km"] = float(m_orb.group(1))

    m_dist = re.search(r"(\d+(?:\.\d+)?)\s*m(?:eters?)?\s*(?:距离|distance|内|外|range)?", q)
    if m_dist and any(k in q for k in ("距离", "distance", "近场", "near-field", "nearfield", "米")):
        d_val = float(m_dist.group(1))
        if 0.5 <= d_val <= 2000.0:
            extracted["min_tx_rx_distance_m"] = d_val

    m_kmh = re.search(r"(\d+(?:\.\d+)?)\s*km/h", q)
    m_mps = re.search(r"(\d+(?:\.\d+)?)\s*m/s", q)
    if m_kmh:
        extracted["max_speed_mps"] = round(float(m_kmh.group(1)) / 3.6, 2)
    elif m_mps:
        extracted["max_speed_mps"] = float(m_mps.group(1))

    m_fps = re.search(r"(\d+)\s*fps", q)
    if m_fps:
        fps = int(m_fps.group(1))
        if fps >= 90:
            extracted["traffic_model"] = "xr_90fps"
        else:
            extracted["traffic_model"] = "xr_60fps"

    return extracted


def _infer_nr_numerology(fc_ghz: float, max_speed_mps: float, has_urllc: bool) -> tuple[int, float]:
    """Select 3GPP TS 38.211 numerology mu and slot duration ms matching band & Doppler."""
    if fc_ghz >= 70.0:
        mu = 4
    elif fc_ghz >= 24.0:
        mu = 3
    elif has_urllc or max_speed_mps >= 30.0:
        mu = 2
    else:
        mu = 1
    return mu, 1.0 / (2 ** mu)


def _merge_unique(items: list[str]) -> list[str]:
    out: list[str] = []
    for item in items:
        if item and item not in out:
            out.append(item)
    return out


def _estimate_array_aperture_m(num_tx: int, fc_ghz: float) -> float:
    """Compute physical array aperture D [m] at half-wavelength spacing (ULA/ELAA up to 256, planar above)."""
    lam = wavelength_m(fc_ghz * 1e9)
    if num_tx > 256:
        side = math.sqrt(num_tx) - 1.0
        return round(max(0.05, math.sqrt(2.0) * side * 0.5 * lam), 4)
    return round(max(0.05, max(1, num_tx - 1) * 0.5 * lam), 4)


def _sync_provenance_and_physics(spec: WCESpec, *, triggered_rules: list[str], auto_scale_aperture: bool = True) -> list[str]:
    """Ensure standard_ids, near-field/band/coherence checks, and Table I provenance stay synchronized."""
    alerts: list[str] = []
    cards_by_id = {c["id"]: c for c in load_catalog()}

    if auto_scale_aperture and spec.num_tx >= 32:
        spec.aperture_m = max(spec.aperture_m, _estimate_array_aperture_m(spec.num_tx, spec.carrier_fc_ghz))

    zr = rayleigh_distance_m(spec.aperture_m, spec.carrier_fc_ghz * 1e9)
    if spec.min_tx_rx_distance_m < zr and "nearfield_xlmimo_thz" not in spec.blocks_selected:
        spec.blocks_selected.append("nearfield_xlmimo_thz")
        alerts.append(
            f"Near-field transition triggered: min_tx_rx_distance_m={spec.min_tx_rx_distance_m:.1f} m < "
            f"Rayleigh distance Z_R={zr:.1f} m -> auto-added `nearfield_xlmimo_thz` (@cui2022channel)."
        )
        triggered_rules.append("auto_nearfield_rayleigh_upgrade")

    if spec.carrier_fc_ghz > 100.0 and "tr38901_uma_umi_rma" in spec.blocks_selected:
        alerts.append(
            f"Band boundary alert: fc={spec.carrier_fc_ghz:.1f} GHz exceeds 3GPP TR 38.901 [0.5, 100] GHz ceiling; "
            "switching primary high-band attenuation authority to ITU-R P.676 THz gas absorption (`nearfield_xlmimo_thz`)."
        )
        if "nearfield_xlmimo_thz" not in spec.blocks_selected:
            spec.blocks_selected.append("nearfield_xlmimo_thz")
        if "itur_p676" not in spec.standard_ids:
            spec.standard_ids.append("itur_p676")
        triggered_rules.append("band_gt_100ghz_itur_p676_boundary")

    phys = spec.derived_physics_summary()
    if phys["coherence_time_ms"] < spec.slot_duration_ms:
        target_mu = spec.numerology_mu
        while target_mu < 4 and (1.0 / (2 ** target_mu)) > phys["coherence_time_ms"]:
            target_mu += 1
        spec.numerology_mu = target_mu
        spec.slot_duration_ms = 1.0 / (2 ** target_mu)
        alerts.append(
            f"Coherence alert: channel coherence time T_c={phys['coherence_time_ms']:.3f} ms < slot; "
            f"raised NR numerology to mu={spec.numerology_mu} (slot={spec.slot_duration_ms} ms) & enabled `otfs_delay_doppler`."
        )
        if "otfs_delay_doppler" not in spec.blocks_selected:
            spec.blocks_selected.append("otfs_delay_doppler")
        triggered_rules.append("auto_coherence_numerology_otfs")

    if "ambient_iot_backscatter" in spec.blocks_selected and "traffic_xr_immersive" in spec.blocks_selected:
        alerts.append(
            "Feasibility advisory: Ambient IoT backscatter (uW power budget) conflicts with broadband >=15 Mbps XR streaming (`traffic_xr_immersive`); "
            "split IoT sensing/control from active broadband offloading."
        )

    # Sync standard_ids from all selected blocks + baselines
    stds = list(spec.standard_ids)
    for bid in spec.blocks_selected + spec.baseline_blocks:
        card = cards_by_id.get(bid)
        if card:
            stds.extend(card.get("standard_ids", []))
    spec.standard_ids = _merge_unique(stds)

    # Rebuild synchronized Table I parameter_provenance rows without duplication
    band_auth = "itur_p676" if spec.carrier_fc_ghz > 100.0 else "gpp_tr38901"
    band_sec = "Annex 1 (>100 GHz Sub-THz)" if spec.carrier_fc_ghz > 100.0 else "Table 7.4.1-1 (0.5-100 GHz)"
    band_why = (
        "Sub-THz molecular O2/H2O attenuation dominates free-space loss."
        if spec.carrier_fc_ghz > 100.0
        else "Validated 3GPP NR spectrum model across macro/micro/indoor links."
    )
    rows: list[ParamProvenance] = [
        _prov("carrier_fc_ghz", round(spec.carrier_fc_ghz, 3), "GHz", "Carrier frequency", band_auth, band_sec, band_why),
        _prov(
            "bandwidth_hz",
            round(spec.bandwidth_hz / 1e6, 2),
            "MHz",
            "Allocated RF system bandwidth",
            "gpp_ts38211",
            "Clause 4.2",
            "Governs thermal noise integration k_B*T*B*NF and range/symbol rate.",
        ),
        _prov(
            "numerology_mu",
            f"{spec.numerology_mu} (slot={spec.slot_duration_ms} ms)",
            "enum",
            "3GPP NR OFDM numerology & slot width",
            "gpp_ts38211",
            "Table 4.2-1",
            f"Matches fc={spec.carrier_fc_ghz:.1f} GHz & speed={spec.max_speed_mps:.1f} m/s (T_c={phys['coherence_time_ms']:.2f} ms).",
        ),
        _prov(
            "rayleigh_distance_m",
            round(zr, 2),
            "m",
            f"Rayleigh near-field boundary Z_R (d_min={spec.min_tx_rx_distance_m:.1f} m)",
            "cui_dai_nearfield_spherical" if spec.min_tx_rx_distance_m < zr else "gpp_tr38901",
            "IEEE TCOM 2022" if spec.min_tx_rx_distance_m < zr else "Clause 7.1",
            "Inside Z_R requires spherical beamfocusing; outside Z_R planar far-field holds.",
        ),
        _prov(
            "csi_assumption",
            spec.csi_assumption,
            "enum",
            "Channel State Information causality/noise",
            "gpp_tr38843" if spec.csi_assumption != "ideal_genie_baseline" else "shannon_awgn",
            "Clause 5 (Causal CSI)" if spec.csi_assumption != "ideal_genie_baseline" else "Oracle Upper Bound",
            "Prevents future-CSI leakage unless explicitly marked as an unachievable upper bound.",
        ),
        _prov(
            "traffic_model",
            spec.traffic_model,
            "enum",
            "Packet arrival & queue traffic pattern",
            "gpp_tr38838" if "xr" in spec.traffic_model else "gpp_tr36814",
            "Clause 4-6" if "xr" in spec.traffic_model else "Annex A.2.1.3",
            "Enforces non-full-buffer queues and deadline/delay-budget evaluation.",
        ),
    ]
    for bid in spec.blocks_selected:
        card = cards_by_id.get(bid)
        if not card:
            continue
        first_mech = (card.get("default_mechanisms") or [None])[0]
        first_std = (card.get("standard_ids") or [None])[0]
        auth = first_mech or first_std or "itur_m2160"
        rows.append(
            _prov(
                f"module[{bid}]",
                card["era"],
                "block",
                card["name"],
                auth,
                card["summary_plain"][:42] + ("..." if len(card["summary_plain"]) > 42 else ""),
                card["diff_from_parent"],
            )
        )
    if spec.num_ris_elements > 0 or any(b in spec.blocks_selected for b in ("passive_ris_cascade", "star_ris", "active_ris_flexible")):
        n_el_eff = spec.num_ris_elements if spec.num_ris_elements > 0 else 64
        rows.append(
            _prov(
                "num_ris_elements",
                n_el_eff,
                "elements",
                f"RIS meta-atoms M (phase_bits={spec.ris_phase_bits or 'continuous'})",
                "wu_zhang_passive_ris",
                "ETSI GR RIS 003 / IEEE TWC 2019",
                "Explicitly tracks metasurface element count M separately from BS antennas N_t without artificial inflation.",
            )
        )
    if "uav_air_to_ground" in spec.blocks_selected:
        rows.append(
            _prov(
                "uav_altitude_m",
                round(spec.uav_altitude_m, 1),
                "m",
                "UAV flight height above ground",
                "gpp_tr36777",
                "Table B-1",
                "Determines aerial LoS probability & shadow fading scaling up to 300 m.",
            )
        )
    if "ntn_leo_satellite" in spec.blocks_selected:
        rows.append(
            _prov(
                "ntn_elevation_deg",
                round(spec.ntn_elevation_deg, 1),
                "deg",
                f"Satellite elevation (orbit={spec.ntn_orbit_altitude_km:.0f} km)",
                "gpp_tr38811",
                "Table 6.6.1-1 / 6.6.2",
                "Sets orbital slant range, orbital Doppler, LoS probability, and band-specific clutter loss.",
            )
        )
    spec.parameter_provenance = rows
    return alerts


def compose_wce_spec(
    query: str,
    *,
    overrides: dict[str, Any] | None = None,
    problem_name: str | None = None,
) -> tuple[WCESpec, dict[str, Any]]:
    """Decompose any natural-language wireless research prompt across 4 orthogonal slots into a coupled WCESpec."""
    q = query.lower()
    nums = _extract_numeric_clues(query)
    if overrides:
        nums.update(overrides)

    triggered_rules: list[str] = []

    # --- Slot 1: Propagation & Topology ---
    prop_blocks: list[str] = []
    prop_reasons: list[str] = []
    if any(k in q for k in ("uav", "drone", "aerial", "drone-assisted", "uav_gis", "空地", "无人机")):
        prop_blocks.append("uav_air_to_ground")
        prop_reasons.append("TR 36.777 / Al-Hourani aerial altitude LoS geometry")
    if any(k in q for k in ("leo", "ntn", "satellite", "starlink", "oneweb", "kuiper", "卫星", "星地", "非地面")):
        prop_blocks.append("ntn_leo_satellite")
        prop_reasons.append("TR 38.811/38.821 LEO elevation slant channel")
    if any(k in q for k in ("v2x", "v2v", "v2i", "v2v_control", "platooning", "autonomous driving", "highway", "sidelink", "车联网", "车载")):
        prop_blocks.append("v2x_vehicular_sl")
        prop_reasons.append("TR 37.885 V2X urban/highway + NLOSv shadowing")
    if any(k in q for k in ("rain", "p.618", "p618", "rainfall", "weather")):
        triggered_rules.append("itu_p618_rain_attenuation")
    if any(k in q for k in ("lora", "lorawan", "lpwan", "sub-ghz")):
        triggered_rules.append("lpwan_subghz_lora")
    if any(k in q for k in ("factory", "industrial", "inf-", "工厂", "工业")):
        prop_blocks.append("tr38901_inf")
        prop_reasons.append("TR 38.901 Indoor Factory metallic clutter profile")
    if any(k in q for k in ("cdl", "tdl", "ofdm", "wideband", "宽带", "多径")):
        prop_blocks.append("tr38901_cdl_tdl")
        prop_reasons.append("TR 38.901 CDL/TDL wideband OFDM channel")
    if any(k in q for k in ("uma", "umi", "rma", "o2i", "urban", "cellular", "基站", "蜂窝", "城域")) or not prop_blocks:
        prop_blocks.append("tr38901_uma_umi_rma")
        prop_reasons.append("TR 38.901 UMa/UMi/RMa 3D spatial baseline")
    if any(k in q for k in ("ppp", "stochastic geometry", "随机几何")):
        prop_blocks.append("ppp_stochastic_geo")
    if any(k in q for k in ("rician", "莱斯")):
        prop_blocks.append("rician_los")
    if any(k in q for k in ("nakagami", "仲上")):
        prop_blocks.append("nakagami_m")
    prop_blocks = _merge_unique(prop_blocks)

    # --- Slot 2: Antennas, Near-Field & Metasurfaces ---
    ant_blocks: list[str] = []
    ant_reasons: list[str] = []
    if any(k in q for k in ("star-ris", "star_ris", "starris", "全空间超表面", "透反射")):
        ant_blocks.append("star_ris")
        ant_reasons.append("STAR-RIS energy-splitting reflection+transmission")
    if any(k in q for k in ("active ris", "active-ris", "有源超表面", "fluid antenna", "pinching", "流体天线", "夹断")) or re.search(
        r"\b(?:fas|pass)\b", q
    ):
        ant_blocks.append("active_ris_flexible")
        ant_reasons.append("Active RIS / FAS / PASS beyond-passive array")
    if (re.search(r"\b(?:ris|irs)\b", q) or any(k in q for k in ("reconfigurable", "超表面", "反射面"))) and not any(
        b in ant_blocks for b in ("star_ris", "active_ris_flexible")
    ):
        ant_blocks.append("passive_ris_cascade")
        ant_reasons.append("ETSI GR RIS 003 unit-modulus passive RIS cascade")
    if any(k in q for k in ("near-field", "nearfield", "xl-mimo", "elaa", "thz", "terahertz", "近场", "太赫兹")):
        ant_blocks.append("nearfield_xlmimo_thz")
        ant_reasons.append("Spherical wavefront XL-MIMO + ITU-R P.676 gas attenuation")
    if any(k in q for k in ("upa", "mimo", "beamform", "波束")) and "nearfield_xlmimo_thz" not in ant_blocks:
        ant_blocks.append("farfield_upa")
        ant_reasons.append("Far-field 2D UPA antenna manifold")
    ant_blocks = _merge_unique(ant_blocks)

    # --- Slot 3: IMT-2030 6G Features & Traffic / MAC ---
    feat_blocks: list[str] = []
    feat_reasons: list[str] = []
    if any(k in q for k in ("isac", "jsac", "radar", "rcs", "crb", "sensing", "通感")):
        feat_blocks.append("isac_jsac_channel")
        feat_reasons.append("3GPP Rel-19 ISAC monostatic/bistatic echo + CRB")
    if any(k in q for k in ("wifi", "wi-fi", "802.11bf", "802.11bn")):
        feat_blocks.append("wifi_sensing_80211bf")
    if any(k in q for k in ("otfs", "delay-doppler", "high speed", "高速", "高铁")):
        feat_blocks.append("otfs_delay_doppler")
        feat_reasons.append("2D Delay-Doppler OTFS modulation channel")
    if any(k in q for k in ("urllc", "hrllc", "fbl", "finite blocklength", "short packet", "v2v_control", "uav_gis", "autonomous driving", "短包", "超低时延", "高可靠")):
        feat_blocks.append("hrllc_fbl_urllc")
        feat_reasons.append("Polyanskiy FBL dispersion short-packet bound")
    if any(k in q for k in ("ambient", "backscatter", "a-iot", "lora", "lorawan", "lpwan", "反向散射", "零功耗")):
        feat_blocks.append("ambient_iot_backscatter")
    if any(k in q for k in ("semantic", "deepsc", "aircomp", "over-the-air", "federated", "federated_edge", "语义", "空中计算", "联邦学习")):
        feat_blocks.append("semantic_aircomp_channel")
        feat_reasons.append("Differentiable DeepSC / AirComp / Federated Edge AI channel")
    if any(k in q for k in ("jam", "jsr", "secrecy", "wiretap", "covert", "pls", "抗干扰", "窃听", "隐蔽", "物理层安全")):
        feat_blocks.append("pls_covert_antijamming")
        feat_reasons.append("Wyner secrecy / Bash covert KL / JSR-dB jamming")

    if re.search(r"\b(?:xr|vr|ar)\b", q) or any(k in q for k in ("fps", "immersive", "沉浸式", "38.838", "pdb")):
        feat_blocks.append("traffic_xr_immersive")
        feat_reasons.append("3GPP TR 38.838 bursty XR traffic & PDB evaluation")
    if any(k in q for k in ("multi-cell", "multicell", "schedul", "queue", "mac", "drl", "federated", "federated_edge", "资源分配", "多小区", "队列")):
        feat_blocks.append("multicell_interference_mac")
        feat_reasons.append("Wrap-around multi-cell interference + queue Gym loop")
    if any(k in q for k in ("ftp", "bursty", "文件到达")):
        feat_blocks.append("traffic_ftp_models")
    if any(k in q for k in ("pareto", "voip", "poisson", "泊松", "帕累托")):
        feat_blocks.append("traffic_poisson_pareto_voip")
    feat_blocks = _merge_unique(feat_blocks)

    # --- Physical Defaults + Extracted Numeric Overrides ---
    if "carrier_fc_ghz" in nums:
        fc_ghz = float(nums["carrier_fc_ghz"])
    elif "lpwan_subghz_lora" in triggered_rules:
        fc_ghz = 0.915
    elif "nearfield_xlmimo_thz" in ant_blocks:
        fc_ghz = 100.0
    elif "isac_jsac_channel" in feat_blocks or any(k in q for k in ("mmwave", "fr2", "毫米波")):
        fc_ghz = 28.0
    elif "v2x_vehicular_sl" in prop_blocks:
        fc_ghz = 5.9
    elif "ntn_leo_satellite" in prop_blocks or "ambient_iot_backscatter" in feat_blocks:
        fc_ghz = 2.0
    else:
        fc_ghz = 3.5

    if "bandwidth_hz" in nums:
        bw_hz = float(nums["bandwidth_hz"])
    elif "lpwan_subghz_lora" in triggered_rules:
        bw_hz = 125e3
    elif fc_ghz >= 70.0:
        bw_hz = 1e9
    elif fc_ghz >= 24.0:
        bw_hz = 100e6
    else:
        bw_hz = 20e6

    max_speed = float(nums.get("max_speed_mps", 60.0 if "otfs_delay_doppler" in feat_blocks else (18.0 if "uav_air_to_ground" in prop_blocks else 10.0)))
    mu, slot_ms = _infer_nr_numerology(fc_ghz, max_speed, "hrllc_fbl_urllc" in feat_blocks)

    num_tx = int(nums.get("num_tx", 128 if "nearfield_xlmimo_thz" in ant_blocks else (32 if fc_ghz >= 24.0 else 8)))
    min_dist = float(nums.get("min_tx_rx_distance_m", 12.0 if "nearfield_xlmimo_thz" in ant_blocks else 25.0))
    aperture_m = float(nums["aperture_m"]) if "aperture_m" in nums else _estimate_array_aperture_m(num_tx, fc_ghz)

    # --- Slot 4: CSI Causality & Mandatory Analytical Baselines ---
    if any(k in q for k in ("delayed csi", "outdated csi", "反馈延迟", "过期")):
        csi_assumption = "delayed_feedback"
    elif any(k in q for k in ("ideal csi", "genie csi", "oracle", "理想csi")):
        csi_assumption = "ideal_genie_baseline"
    else:
        csi_assumption = "realistic_estimated"

    baselines: list[str] = ["awgn_link"]
    if csi_assumption != "ideal_genie_baseline" or any(k in q for k in ("lmmse", "csi", "pilot", "信道估计", "导频")):
        baselines.append("ls_lmmse_csi")
    if any(b in (*ant_blocks, *feat_blocks) for b in ("passive_ris_cascade", "star_ris", "active_ris_flexible", "nearfield_xlmimo_thz", "farfield_upa", "isac_jsac_channel")):
        baselines.append("wmmse_precoder")
    else:
        baselines.append("water_filling")
    if any(b in feat_blocks for b in ("multicell_interference_mac", "traffic_xr_immersive", "traffic_ftp_models", "traffic_poisson_pareto_voip")):
        baselines.append("proportional_fair")
    if any(k in q for k in ("ckm", "radio map", "radiomap", "知识图谱")):
        baselines.append("ckm_radiomap_bridge")
    baselines = _merge_unique(baselines)

    # IMT-2030 scenario mapping
    if "isac_jsac_channel" in feat_blocks or "wifi_sensing_80211bf" in feat_blocks:
        imt_scen = "ISAC"
    elif "hrllc_fbl_urllc" in feat_blocks or "v2x_vehicular_sl" in prop_blocks or "tr38901_inf" in prop_blocks:
        imt_scen = "HRLLC"
    elif "ntn_leo_satellite" in prop_blocks or "uav_air_to_ground" in prop_blocks:
        imt_scen = "UC"
    elif "ambient_iot_backscatter" in feat_blocks:
        imt_scen = "MC"
    elif "semantic_aircomp_channel" in feat_blocks or "ckm_radiomap_bridge" in baselines:
        imt_scen = "AIAC"
    else:
        imt_scen = "IC"

    # Traffic model enum
    if "traffic_model" in nums:
        traffic_model = str(nums["traffic_model"])
    elif "traffic_xr_immersive" in feat_blocks:
        traffic_model = "xr_60fps"
    elif "traffic_poisson_pareto_voip" in feat_blocks:
        traffic_model = "pareto"
    else:
        traffic_model = "ftp_model_3"

    # External ecosystem intents (including Chinese natural-language physical scene hints)
    eco = ["wannado_wce_builtin"]
    if any(k in q for k in ("sionna rt", "ray tracing", "ray-tracing", "射线追踪", "真实建筑", "建筑遮挡", "3d建筑", "街区", "城市三维", "site-specific")):
        eco.extend(["sionna_rt", "sionna", "deepmimo"])
    elif "sionna" in q:
        eco.append("sionna")
    if "deepmimo" in q:
        eco.append("deepmimo")
    if "quadriga" in q:
        eco.append("quadriga_lib")
    if any(k in q for k in ("ns-3", "ns3", "5g-lena", "ns3-gym")):
        eco.append("ns3_5glena_ns3gym")
    if any(k in q for k in ("oai", "openairinterface")):
        eco.append("openairinterface")
    if "srsran" in q:
        eco.append("srsran_project")
    if "torchsig" in q:
        eco.append("torchsig")
    if "hermespy" in q:
        eco.append("hermespy")

    export_targets = ["pytorch_complex", "markdown_report", "latex_table"]
    if "multicell_interference_mac" in feat_blocks or any(k in q for k in ("drl", "ppo", "sac", "reinforcement", "强化学习")):
        export_targets.extend(["gym_env", "wannado_drl"])
    if "pls_covert_antijamming" in feat_blocks or any(k in q for k in ("adversarial", "jam", "鲁棒", "攻击")):
        export_targets.append("wannado_adv")

    all_selected = _merge_unique(prop_blocks + ant_blocks + feat_blocks)
    # Keep focused: validate all exist in registry
    all_selected = [b for b in all_selected if b in BLOCK_ID_TO_RUNNER]

    has_ris = any(b in all_selected for b in ("passive_ris_cascade", "star_ris", "active_ris_flexible"))
    num_ris = int(nums["num_ris_elements"]) if "num_ris_elements" in nums else (64 if has_ris else 0)
    ris_bits = int(nums.get("ris_phase_bits", 0))
    uav_alt = float(nums.get("uav_altitude_m", 100.0))
    ntn_alt = float(nums.get("ntn_orbit_altitude_km", 600.0))
    ntn_el = float(nums.get("ntn_elevation_deg", 30.0))

    topo_label = " + ".join(b for b in prop_blocks[:2])
    default_tx_dbm = (
        68.0 if "ntn_leo_satellite" in prop_blocks else (35.0 if fc_ghz < 10.0 else 30.0)
    )
    spec = WCESpec(
        problem_name=problem_name or f"Composed_{imt_scen}_Wireless_Env",
        archetype="composed_custom",
        imt2030_scenario=imt_scen,
        goal_plain=query.strip() or "4-slot Compositional Standards-Anchored Wireless Environment.",
        carrier_fc_ghz=fc_ghz,
        bandwidth_hz=bw_hz,
        numerology_mu=mu,
        tx_power_dbm=float(nums.get("tx_power_dbm", default_tx_dbm)),
        noise_figure_db=float(nums.get("noise_figure_db", 7.0)),
        num_tx=num_tx,
        num_rx=int(nums.get("num_rx", 2)),
        num_ris_elements=num_ris,
        ris_phase_bits=ris_bits,
        aperture_m=aperture_m,
        min_tx_rx_distance_m=min_dist,
        max_speed_mps=max_speed,
        slot_duration_ms=slot_ms,
        uav_altitude_m=uav_alt,
        ntn_orbit_altitude_km=ntn_alt,
        ntn_elevation_deg=ntn_el,
        csi_assumption=csi_assumption,
        traffic_model=traffic_model,
        topology_type=topo_label,
        blocks_selected=all_selected,
        baseline_blocks=baselines,
        ecosystem_tools=_merge_unique(eco),
        export_targets=_merge_unique(export_targets),
        notes="Assembled dynamically by wannado-wce v1.1 4-Slot Compositional Assembler.",
    )
    alerts = _sync_provenance_and_physics(spec, triggered_rules=triggered_rules)

    slots = [
        ComposedSlotDecision("1_propagation_topology", prop_blocks, "; ".join(prop_reasons)),
        ComposedSlotDecision("2_antenna_metasurface", ant_blocks or ["farfield_upa"], "; ".join(ant_reasons) or "Standard planar far-field baseline"),
        ComposedSlotDecision("3_domain_6g_and_traffic", feat_blocks, "; ".join(feat_reasons) or "Default link-level transport"),
        ComposedSlotDecision("4_causality_and_baselines", baselines, f"csi={csi_assumption} + mandatory analytic benchmarks"),
    ]
    plan = {
        "mode": "compositional_4_slot_v1.1",
        "slots": [s.to_dict() for s in slots],
        "pipeline_order": list(spec.blocks_selected),
        "baseline_blocks": list(spec.baseline_blocks),
        "extracted_numeric_overrides": nums,
        "triggered_rules": triggered_rules,
        "boundary_alerts": alerts,
    }
    return spec, plan


def parse_param_updates_from_text(text: str) -> dict[str, Any]:
    """Parse natural-language parameter edits (e.g. 'Switch carrier frequency to 140 GHz, upgrade RIS to 256 elements')."""
    updates = _extract_numeric_clues(text)
    q = text.lower()
    if any(k in q for k in ("ideal csi", "genie", "理想csi")):
        updates["csi_assumption"] = "ideal_genie_baseline"
    elif any(k in q for k in ("delayed csi", "outdated", "延迟csi")):
        updates["csi_assumption"] = "delayed_feedback"
    elif any(k in q for k in ("realistic csi", "lmmse", "真实csi")):
        updates["csi_assumption"] = "realistic_estimated"
    return updates


def update_wce_spec_with_diff(
    spec: WCESpec,
    updates: dict[str, Any] | str,
) -> SpecDiffReport:
    """Apply parameter changes to a WCESpec, auto-trigger boundary alerts/promotions, and refresh Table I."""
    patch = parse_param_updates_from_text(updates) if isinstance(updates, str) else dict(updates)
    before = WCESpec.from_dict(spec.to_dict())
    after = WCESpec.from_dict(spec.to_dict())
    before_phys = before.derived_physics_summary()
    before_blocks = set(before.blocks_selected)
    before_baselines = set(before.baseline_blocks)
    before_stds = set(before.standard_ids)

    changed: dict[str, dict[str, Any]] = {}
    valid_fields = {
        "carrier_fc_ghz",
        "bandwidth_hz",
        "numerology_mu",
        "tx_power_dbm",
        "noise_figure_db",
        "num_tx",
        "num_rx",
        "num_ris_elements",
        "ris_phase_bits",
        "aperture_m",
        "min_tx_rx_distance_m",
        "max_speed_mps",
        "slot_duration_ms",
        "uav_altitude_m",
        "ntn_orbit_altitude_km",
        "ntn_elevation_deg",
        "csi_assumption",
        "traffic_model",
        "topology_type",
    }
    for k, v in patch.items():
        if k in valid_fields:
            old_val = getattr(after, k)
            setattr(after, k, v)
            if k == "numerology_mu":
                after.slot_duration_ms = 1.0 / (2 ** int(v))
                changed["slot_duration_ms"] = {"before": before.slot_duration_ms, "after": after.slot_duration_ms}
            if old_val != v:
                changed[k] = {"before": old_val, "after": v}

    # Recompute physical aperture when num_tx changes unless explicit aperture_m supplied
    if "num_tx" in patch and "aperture_m" not in patch:
        after.aperture_m = _estimate_array_aperture_m(after.num_tx, after.carrier_fc_ghz)

    # Ensure LMMSE baseline is present if realistic CSI or high-rate change requested
    if after.csi_assumption == "realistic_estimated" and "ls_lmmse_csi" not in after.baseline_blocks:
        after.baseline_blocks.append("ls_lmmse_csi")

    triggered_rules: list[str] = []
    alerts = _sync_provenance_and_physics(after, triggered_rules=triggered_rules, auto_scale_aperture=("aperture_m" not in patch))
    if "carrier_fc_ghz" in changed and re.search(r"\b\d+(?:\.\d+)?\s*ghz\b", after.problem_name, flags=re.IGNORECASE):
        after.problem_name = re.sub(
            r"\b\d+(?:\.\d+)?\s*ghz\b",
            f"{after.carrier_fc_ghz:g} GHz",
            after.problem_name,
            flags=re.IGNORECASE,
        )
    if after.problem_name == f"Composed_{before.imt2030_scenario}_Wireless_Env" and after.imt2030_scenario != before.imt2030_scenario:
        after.problem_name = f"Composed_{after.imt2030_scenario}_Wireless_Env"
    for sync_k in ("aperture_m", "numerology_mu", "slot_duration_ms", "problem_name"):
        if getattr(after, sync_k) != getattr(before, sync_k):
            changed[sync_k] = {"before": getattr(before, sync_k), "after": getattr(after, sync_k)}

    aud = audit_wce_spec(after)
    return SpecDiffReport(
        before_physics=before_phys,
        after_physics=after.derived_physics_summary(),
        changed_fields=changed,
        boundary_alerts=alerts,
        auto_promoted_blocks=[b for b in after.blocks_selected if b not in before_blocks],
        auto_added_baselines=[b for b in after.baseline_blocks if b not in before_baselines],
        auto_updated_standards=[s for s in after.standard_ids if s not in before_stds],
        audit_ok=aud.ok,
        audit_warnings=list(aud.warnings),
        updated_spec=after,
        updated_table1_md=after.format_system_parameter_table_md(),
        updated_table1_tex=after.format_system_parameter_table_tex(),
    )
