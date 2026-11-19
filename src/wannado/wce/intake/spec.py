"""Formal Stage 0 Wireless Environment Contract Schema (WCESpec) with Per-Parameter Provenance."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from wannado.wce.core.types import ParamProvenance
from wannado.wce.core.units import (
    coherence_time_sec,
    doppler_max_hz,
    rayleigh_distance_m,
    thermal_noise_power_dbm,
)


@dataclass
class WCESpec:
    """Complete Stage 0 wireless communication environment specification with authoritative provenance."""

    problem_name: str
    archetype: str
    imt2030_scenario: str
    goal_plain: str
    carrier_fc_ghz: float = 3.5
    bandwidth_hz: float = 20e6
    numerology_mu: int = 1
    tx_power_dbm: float = 43.0
    noise_figure_db: float = 7.0
    num_tx: int = 8
    num_rx: int = 2
    num_ris_elements: int = 0
    ris_phase_bits: int = 0
    aperture_m: float = 0.30
    min_tx_rx_distance_m: float = 25.0
    max_speed_mps: float = 10.0
    slot_duration_ms: float = 0.5
    uav_altitude_m: float = 100.0
    ntn_orbit_altitude_km: float = 600.0
    ntn_elevation_deg: float = 30.0
    csi_assumption: str = "realistic_estimated"  # realistic_estimated | delayed_feedback | ideal_genie_baseline
    traffic_model: str = "ftp_model_3"
    topology_type: str = "UMa"
    blocks_selected: list[str] = field(default_factory=lambda: ["tr38901_uma_umi_rma", "tr38901_cdl_tdl"])
    baseline_blocks: list[str] = field(default_factory=lambda: ["awgn_link", "wmmse_precoder"])
    ecosystem_tools: list[str] = field(default_factory=lambda: ["wannado_wce_builtin"])
    export_targets: list[str] = field(default_factory=lambda: ["pytorch_complex", "markdown_report", "latex_table"])
    parameter_provenance: list[ParamProvenance] = field(default_factory=list)
    standard_ids: list[str] = field(default_factory=lambda: ["itur_m2160", "gpp_tr38901", "gpp_ts38211"])
    notes: str = ""

    def derived_physics_summary(self) -> dict[str, float]:
        """Compute key physical boundary invariants directly from spec settings."""
        fc_hz = self.carrier_fc_ghz * 1e9
        fd = doppler_max_hz(self.max_speed_mps, fc_hz)
        tc_ms = coherence_time_sec(fd) * 1e3 if fd > 0 else float("inf")
        zr = rayleigh_distance_m(self.aperture_m, fc_hz)
        noise_dbm = thermal_noise_power_dbm(self.bandwidth_hz, noise_figure_db=self.noise_figure_db)
        return {
            "noise_floor_dbm": noise_dbm,
            "max_doppler_hz": fd,
            "coherence_time_ms": tc_ms,
            "rayleigh_distance_m": zr,
            "is_nearfield_regime": float(self.min_tx_rx_distance_m < zr),
        }

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["derived_physics"] = self.derived_physics_summary()
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True)

    def save(self, path: str | Path) -> Path:
        out = Path(path).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.suffix.lower() in (".yaml", ".yml"):
            out.write_text(self.to_yaml(), encoding="utf-8")
        else:
            out.write_text(self.to_json() + "\n", encoding="utf-8")
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WCESpec:
        prov_raw = data.get("parameter_provenance", [])
        prov = [p if isinstance(p, ParamProvenance) else ParamProvenance(**p) for p in prov_raw]
        return cls(
            problem_name=data["problem_name"],
            archetype=data.get("archetype", "custom"),
            imt2030_scenario=data.get("imt2030_scenario", "IC"),
            goal_plain=data.get("goal_plain", ""),
            carrier_fc_ghz=float(data.get("carrier_fc_ghz", 3.5)),
            bandwidth_hz=float(data.get("bandwidth_hz", 20e6)),
            numerology_mu=int(data.get("numerology_mu", 1)),
            tx_power_dbm=float(data.get("tx_power_dbm", 43.0)),
            noise_figure_db=float(data.get("noise_figure_db", 7.0)),
            num_tx=int(data.get("num_tx", 8)),
            num_rx=int(data.get("num_rx", 2)),
            num_ris_elements=int(data.get("num_ris_elements", 0)),
            ris_phase_bits=int(data.get("ris_phase_bits", 0)),
            aperture_m=float(data.get("aperture_m", 0.30)),
            min_tx_rx_distance_m=float(data.get("min_tx_rx_distance_m", 25.0)),
            max_speed_mps=float(data.get("max_speed_mps", 10.0)),
            slot_duration_ms=float(data.get("slot_duration_ms", 0.5)),
            uav_altitude_m=float(data.get("uav_altitude_m", 100.0)),
            ntn_orbit_altitude_km=float(data.get("ntn_orbit_altitude_km", 600.0)),
            ntn_elevation_deg=float(data.get("ntn_elevation_deg", 30.0)),
            csi_assumption=str(data.get("csi_assumption", "realistic_estimated")),
            traffic_model=str(data.get("traffic_model", "ftp_model_3")),
            topology_type=str(data.get("topology_type", "UMa")),
            blocks_selected=list(data.get("blocks_selected", [])),
            baseline_blocks=list(data.get("baseline_blocks", [])),
            ecosystem_tools=list(data.get("ecosystem_tools", ["wannado_wce_builtin"])),
            export_targets=list(data.get("export_targets", ["pytorch_complex", "markdown_report", "latex_table"])),
            parameter_provenance=prov,
            standard_ids=list(data.get("standard_ids", [])),
            notes=str(data.get("notes", "")),
        )

    @classmethod
    def load(cls, path: str | Path) -> WCESpec:
        p = Path(path)
        raw = p.read_text(encoding="utf-8")
        if p.suffix.lower() in (".yaml", ".yml"):
            return cls.from_dict(yaml.safe_load(raw))
        return cls.from_dict(json.loads(raw))

    def format_system_parameter_table_md(self) -> str:
        """Render the complete System Simulation Parameters Table (Table I) with authoritative provenance in Markdown."""
        lines = [
            "| Parameter | Value | Unit | Plain Meaning | Authority & Clause | Citation Key | Rationale |",
            "| :--- | :---: | :---: | :--- | :--- | :--- | :--- |",
        ]
        for p in self.parameter_provenance:
            lines.append(
                f"| `{p.parameter}` | `{p.value}` | {p.unit} | {p.plain_meaning} | "
                f"`{p.authority_id}` ({p.authority_section}) | `@{p.citation_key}` | {p.rationale} |"
            )
        return "\n".join(lines)

    def format_system_parameter_table_tex(self) -> str:
        """Render publication-ready LaTeX booktabs Table I: System Simulation Parameters with per-row \\cite{...}."""
        def _esc(val: Any) -> str:
            return (
                str(val)
                .replace("&", "\\&")
                .replace("%", "\\%")
                .replace("$", "\\$")
                .replace("#", "\\#")
                .replace("_", "\\_")
                .replace("^", "\\^{}")
                .replace("~", "\\~{}")
            )

        lines = [
            "% Requires \\usepackage{booktabs}",
            "\\begin{table}[t]",
            "  \\centering",
            f"  \\caption{{Standards-Anchored System Simulation Parameters for {_esc(self.problem_name)} (IMT-2030 {_esc(self.imt2030_scenario)}).}}",
            "  \\label{tab:wce_system_parameters}",
            "  \\begin{tabular}{llll}",
            "    \\toprule",
            "    Parameter & Value & Authority / Clause & Reference \\\\",
            "    \\midrule",
        ]
        for p in self.parameter_provenance:
            val_s = str(p.value).strip()
            unit_s = str(p.unit).strip() if p.unit is not None else ""
            if not unit_s or unit_s in ("enum", "dimensionless", "block") or unit_s.lower() in val_s.lower():
                val_unit = _esc(val_s)
            else:
                val_unit = f"{_esc(val_s)} {_esc(unit_s)}"
            auth = f"{_esc(p.authority_id)} ({_esc(p.authority_section)})"
            cit = f"\\cite{{{p.citation_key}}}" if p.citation_key else "--"
            lines.append(f"    {_esc(p.plain_meaning)} & {val_unit} & {auth} & {cit} \\\\")
        lines.extend(["    \\bottomrule", "  \\end{tabular}", "\\end{table}", ""])
        return "\n".join(lines)
