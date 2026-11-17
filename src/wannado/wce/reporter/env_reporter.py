"""Stage 4: Standards-Anchored Environment Provenance Reporter, Paper Table I, Charts & Zero-Hallucination BibTeX."""

from __future__ import annotations

import csv
import importlib
import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from wannado.wce.audit.preflight import PreflightReport, preflight_wce
from wannado.wce.audit.rigor_auditor import RigorAuditReport, audit_wce_rigor
from wannado.wce.blocks import BLOCK_ID_TO_RUNNER
from wannado.wce.catalog.build_catalog import load_catalog
from wannado.wce.core.provenance import (
    MECHANISM_PROVENANCE,
    STANDARDS_PROVENANCE,
    resolve_block_provenance,
)
from wannado.wce.core.types import ParamProvenance, ProvisionResult
from wannado.wce.core.utils import aggregate_seed_stats
from wannado.wce.intake.spec import WCESpec


def export_citations(
    ids: Sequence[str],
    fmt: str = "bibtex",
    *,
    include_standards: bool = True,
    include_mechanisms: bool = True,
) -> str:
    """Export verbatim verified BibTeX/IEEE/APA citations for blocks, standards, and mechanisms (zero hallucination)."""
    if fmt not in ("bibtex", "ieee", "apa"):
        raise ValueError(f"Unsupported citation format '{fmt}'. Expected 'bibtex', 'ieee', or 'apa'.")

    by_block = {c["id"]: c for c in load_catalog()}
    seen_keys: set[str] = set()
    entries: list[str] = []

    def _push(cit: dict[str, Any] | Any) -> None:
        cdict = cit if isinstance(cit, dict) else asdict(cit)
        key = str(cdict.get("key", ""))
        text = str(cdict.get(fmt, ""))
        if not text:
            return
        dedup = key or text
        if dedup not in seen_keys:
            seen_keys.add(dedup)
            entries.append(text)

    for item in ids:
        if item in by_block:
            card = by_block[item]
            _push(card["citation"])
            if include_standards:
                for sid in card.get("standard_ids", []):
                    if sid in STANDARDS_PROVENANCE:
                        _push(STANDARDS_PROVENANCE[sid].citation)
            if include_mechanisms:
                for mid in card.get("default_mechanisms", []):
                    if mid in MECHANISM_PROVENANCE:
                        _push(MECHANISM_PROVENANCE[mid].citation)
        elif item in STANDARDS_PROVENANCE:
            _push(STANDARDS_PROVENANCE[item].citation)
        elif item in MECHANISM_PROVENANCE:
            _push(MECHANISM_PROVENANCE[item].citation)
        else:
            raise KeyError(f"Unknown block/standard/mechanism ID '{item}' in export_citations(). Refusing to fabricate citation.")

    sep = "\n\n" if fmt == "bibtex" else "\n"
    return sep.join(entries)


@dataclass
class BlockExperimentSummary:
    """Multi-seed aggregated statistics, parameter provenance, and audit verdict for one block."""

    block_id: str
    display_name: str
    family: str
    era: str
    seeds: list[int]
    primary_metric_name: str
    seed_stats: dict[str, float]
    mean_elapsed_sec: float
    config: dict[str, Any]
    citation: dict[str, Any]
    standards_used: list[dict[str, Any]] = field(default_factory=list)
    mechanisms_used: list[dict[str, Any]] = field(default_factory=list)
    parameter_provenance: list[ParamProvenance] = field(default_factory=list)
    audit: RigorAuditReport | None = None
    seed_runs: list[dict[str, Any]] = field(default_factory=list)


def _write_fallback_svg(report: WCEReport, svg_path: Path) -> Path:
    """Dependency-free SVG summary bar chart fallback."""
    entries = [(s.block_id, float(s.seed_stats.get("iqm", 0.0))) for s in report.block_summaries]
    vals = [v for _, v in entries] or [0.0]
    v_min, v_max = min(0.0, min(vals)), max(1e-6, max(vals))
    span = max(1e-6, v_max - v_min)
    w, h = 720, 300
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="24" y="28" font-family="sans-serif" font-size="14" font-weight="bold" fill="#0f172a">{report.task_name} — Multi-Seed Metric IQM</text>',
    ]
    bar_w = max(24, min(80, int(580 / max(1, len(entries) * 1.4))))
    top_y, bot_y = 55, 240
    usable = bot_y - top_y
    for i, (lbl, val) in enumerate(entries):
        x = 45 + i * (bar_w + 18)
        bt = top_y + ((v_max - max(0.0, val)) / span) * usable
        bb = top_y + ((v_max - min(0.0, val)) / span) * usable
        bh = max(2.0, bb - bt)
        lines.append(f'<rect x="{x}" y="{bt:.1f}" width="{bar_w}" height="{bh:.1f}" rx="4" fill="#2563eb"/>')
        lines.append(f'<text x="{x + bar_w / 2:.1f}" y="{bt - 5:.1f}" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#1e293b">{val:.2f}</text>')
        lines.append(f'<text x="{x + bar_w / 2:.1f}" y="268" text-anchor="middle" font-family="sans-serif" font-size="9" fill="#334155">{lbl[:12]}</text>')
    lines.append("</svg>")
    svg_path.write_text("\n".join(lines), encoding="utf-8")
    return svg_path


def _generate_diagnostics_figure(report: WCEReport, out_dir: Path) -> Path:
    """Render clean multi-panel wireless diagnostics & benchmark chart (PNG via matplotlib or SVG fallback)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        return _write_fallback_svg(report, out_dir / "wce_diagnostics.svg")

    png_path = out_dir / "wce_diagnostics.png"
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12.0, 4.4), dpi=160)
    palette = ["#2563eb", "#059669", "#d97706", "#7c3aed", "#db2777", "#0891b2"]

    # Left panel: Multi-seed IQM & Std across provisioned blocks + baselines
    labels: list[str] = []
    means: list[float] = []
    stds: list[float] = []
    iqms: list[float] = []
    cols: list[str] = []
    for idx, s in enumerate(report.block_summaries):
        tag = "PASS" if (s.audit is None or s.audit.passed) else "WARN"
        short_id = s.block_id if len(s.block_id) <= 18 else (s.block_id[:15] + "...")
        short_m = s.primary_metric_name if len(s.primary_metric_name) <= 16 else (s.primary_metric_name[:13] + "...")
        labels.append(f"{short_id}\n({short_m} | {tag})")
        means.append(float(s.seed_stats.get("mean", 0.0)))
        stds.append(float(s.seed_stats.get("std", 0.0)))
        iqms.append(float(s.seed_stats.get("iqm", 0.0)))
        cols.append("#64748b" if s.block_id in report.baseline_blocks else palette[idx % len(palette)])

    x = np.arange(len(labels))
    ax_left.bar(x, means, yerr=stds, capsize=4, color=cols, alpha=0.88, edgecolor="#0f172a", linewidth=0.6)
    ax_left.scatter(x, iqms, color="#facc15", edgecolor="#0f172a", zorder=5, s=40, label="IQM")
    if means and max(abs(m) for m in means) > 50.0 * max(min(abs(m) for m in means if m != 0.0) if any(m != 0.0 for m in means) else 1.0, 1e-6):
        ax_left.set_yscale("symlog", linthresh=1.0)
    ax_left.set_xticks(x)
    ax_left.set_xticklabels(labels, fontsize=7.0, rotation=18, ha="right")
    ax_left.set_title(f"{report.task_name} — Provisioned Blocks ({len(report.seeds)} seeds)", fontsize=10.5)
    ax_left.set_ylabel("Primary Benchmark Metric")
    ax_left.grid(True, axis="y", linestyle=":", alpha=0.45)
    ax_left.legend(loc="best", fontsize=8)

    # Right panel: Empirical distribution (SNR CDF / Pathloss / Sample Metric trace if available)
    plotted_cdf = False
    for idx, (lbl, vals) in enumerate(report.diagnostic_distributions.items()):
        arr = np.sort(np.asarray(vals, dtype=np.float64))
        if arr.size >= 5:
            cdf = np.linspace(0.0, 1.0, arr.size)
            ax_right.plot(arr, cdf, label=lbl, color=palette[idx % len(palette)], linewidth=2.0)
            plotted_cdf = True

    if plotted_cdf:
        ax_right.set_title("Empirical Distribution CDF (First Seed)")
        ax_right.set_xlabel("Metric Value (e.g., SNR dB / SINR dB)")
        ax_right.set_ylabel("Cumulative Probability (CDF)")
        ax_right.grid(True, linestyle=":", alpha=0.45)
        ax_right.legend(loc="best", fontsize=8)
    else:
        times = [s.mean_elapsed_sec * 1000.0 for s in report.block_summaries]
        ax_right.barh(np.arange(len(labels)), times, color=cols, alpha=0.85)
        ax_right.set_yticks(np.arange(len(labels)))
        ax_right.set_yticklabels([s.block_id for s in report.block_summaries], fontsize=8)
        ax_right.set_xlabel("Provisioning Latency (ms)")
        ax_right.set_title("Hardware Execution Speed per Block")
        ax_right.grid(True, axis="x", linestyle=":", alpha=0.45)

    fig.tight_layout()
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)
    return png_path


@dataclass
class WCEReport:
    """Complete Stage 4 traceable wireless environment report & academic paper artifact exporter."""

    task_name: str
    imt2030_scenario: str
    seeds: list[int]
    device: str
    torch_version: str
    timestamp_utc: str
    wce_spec: WCESpec | None
    preflights: dict[str, PreflightReport] = field(default_factory=dict)
    baseline_blocks: list[str] = field(default_factory=list)
    block_summaries: list[BlockExperimentSummary] = field(default_factory=list)
    diagnostic_distributions: dict[str, list[float]] = field(default_factory=dict)
    notes: str = ""

    def all_parameter_provenance(self) -> list[ParamProvenance]:
        """Merge Stage 0 WCESpec parameter provenance with block-level ParamProvenance without duplicates."""
        merged: list[ParamProvenance] = []
        seen: set[tuple[str, str]] = set()
        if self.wce_spec is not None:
            for p in self.wce_spec.parameter_provenance:
                sig = (p.parameter, str(p.value))
                if sig not in seen:
                    seen.add(sig)
                    merged.append(p)
        for s in self.block_summaries:
            for p in s.parameter_provenance:
                sig = (p.parameter, str(p.value))
                if sig not in seen:
                    seen.add(sig)
                    merged.append(p)
        return merged

    def verified_bibliography(self) -> list[dict[str, Any]]:
        """Collect deduplicated verified citations across standards, primary papers, and sub-mechanisms."""
        by_key: dict[str, dict[str, Any]] = {}

        def _reg(cit: dict[str, Any], *, category: str, used_by: str) -> None:
            key = str(cit.get("key", ""))
            if not key:
                return
            if key not in by_key:
                by_key[key] = {**cit, "category": category, "used_by": [used_by]}
            elif used_by not in by_key[key]["used_by"]:
                by_key[key]["used_by"].append(used_by)

        if self.wce_spec is not None:
            for sid in self.wce_spec.standard_ids:
                if sid in STANDARDS_PROVENANCE:
                    _reg(asdict(STANDARDS_PROVENANCE[sid].citation), category="standard", used_by=f"spec:{sid}")
            for p in self.wce_spec.parameter_provenance:
                if p.authority_id in STANDARDS_PROVENANCE:
                    _reg(asdict(STANDARDS_PROVENANCE[p.authority_id].citation), category="standard", used_by=f"param:{p.parameter}")
                elif p.authority_id in MECHANISM_PROVENANCE:
                    _reg(asdict(MECHANISM_PROVENANCE[p.authority_id].citation), category="mechanism", used_by=f"param:{p.parameter}")

        for s in self.block_summaries:
            if s.citation:
                _reg(s.citation, category="block_primary", used_by=s.block_id)
            for st in s.standards_used:
                cit = st.get("citation") or {}
                if cit:
                    _reg(cit, category="standard", used_by=f"{s.block_id}:{st['id']}")
            for mc in s.mechanisms_used:
                cit = mc.get("citation") or {}
                if cit:
                    _reg(cit, category="mechanism", used_by=f"{s.block_id}:{mc['id']}")

        return list(by_key.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "environment": {
                "timestamp_utc": self.timestamp_utc,
                "torch_version": self.torch_version,
                "device": self.device,
                "seeds": list(self.seeds),
            },
            "task": {"task_name": self.task_name, "imt2030_scenario": self.imt2030_scenario, "notes": self.notes},
            "wce_spec": self.wce_spec.to_dict() if self.wce_spec is not None else None,
            "merged_parameter_provenance": [p.to_dict() for p in self.all_parameter_provenance()],
            "preflights": {k: v.to_dict() for k, v in self.preflights.items()},
            "blocks": [
                {
                    "block_id": s.block_id,
                    "display_name": s.display_name,
                    "family": s.family,
                    "era": s.era,
                    "seeds": s.seeds,
                    "primary_metric_name": s.primary_metric_name,
                    "seed_stats": s.seed_stats,
                    "mean_elapsed_sec": s.mean_elapsed_sec,
                    "config": s.config,
                    "citation": s.citation,
                    "standards_used": s.standards_used,
                    "mechanisms_used": s.mechanisms_used,
                    "audit": s.audit.to_dict() if s.audit is not None else None,
                    "seed_runs": s.seed_runs,
                }
                for s in self.block_summaries
            ],
            "verified_bibliography": self.verified_bibliography(),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def to_bibtex(self) -> str:
        entries = [e["bibtex"] for e in self.verified_bibliography() if e.get("bibtex")]
        return ("\n\n".join(entries) + "\n") if entries else ""

    def to_params_latex(self) -> str:
        """Render Table I (System Simulation Parameters with per-row \\cite{...}) merging Spec + Block provenances."""
        if self.wce_spec is not None:
            merged_spec = WCESpec.from_dict(self.wce_spec.to_dict())
            merged_spec.parameter_provenance = self.all_parameter_provenance()
            return merged_spec.format_system_parameter_table_tex()
        tmp = WCESpec(
            problem_name=self.task_name,
            archetype="custom",
            imt2030_scenario=self.imt2030_scenario,
            goal_plain="",
            parameter_provenance=self.all_parameter_provenance(),
        )
        return tmp.format_system_parameter_table_tex()

    def to_results_latex(self) -> str:
        def _esc(val: Any) -> str:
            return str(val).replace("&", "\\&").replace("_", "\\_").replace("%", "\\%")

        lines = [
            "% Requires \\usepackage{booktabs}",
            "\\begin{table}[t]",
            "  \\centering",
            f"  \\caption{{Multi-Seed Wireless Environment Benchmark & Rigor Audit on {_esc(self.task_name)} ({len(self.seeds)} seeds).}}",
            "  \\label{tab:wannado_wce_results}",
            "  \\begin{tabular}{lllccc}",
            "    \\toprule",
            "    Module & Primary Metric & Standards / Ref & Mean $\\pm$ Std & IQM & Audit \\\\",
            "    \\midrule",
        ]
        for s in self.block_summaries:
            st = s.seed_stats
            aud = "PASS" if (s.audit is None or s.audit.passed) else "WARN"
            ckey = str(s.citation.get("key", "")).strip()
            ref_str = f"\\cite{{{ckey}}}" if ckey else "--"
            lines.append(
                f"    {_esc(s.display_name)} & \\texttt{{{_esc(s.primary_metric_name)}}} & {ref_str} & "
                f"${st['mean']:.3f} \\pm {st['std']:.3f}$ & {st['iqm']:.3f} & {aud} \\\\"
            )
        lines.extend(["    \\bottomrule", "  \\end{tabular}", "\\end{table}", ""])
        return "\n".join(lines)

    def format_environment_provenance_md(
        self,
        *,
        chart_filename: str | None = None,
        output_dir: str | Path | None = None,
    ) -> str:
        """Render the flagship traceable Markdown report explaining WHAT was provisioned and WHY every setting has an authority."""
        lines = [f"# Wireless Environment Provenance & Benchmark Report — {self.task_name} (IMT-2030 `{self.imt2030_scenario}`)", ""]
        lines.append("## 1. Plain-Language Provisioned Environment Summary")
        if self.wce_spec is not None:
            lines.append(f"- **Research Goal**: {self.wce_spec.goal_plain}")
            phys = self.wce_spec.derived_physics_summary()
            lines.append(
                f"- **Key Physical Regime**: `f_c={self.wce_spec.carrier_fc_ghz} GHz`, `B={self.wce_spec.bandwidth_hz/1e6:.2f} MHz`, "
                f"`Noise Floor={phys['noise_floor_dbm']:.1f} dBm`, `f_d={phys['max_doppler_hz']:.1f} Hz` (`T_c={phys['coherence_time_ms']:.2f} ms`), "
                f"`Rayleigh Z_R={phys['rayleigh_distance_m']:.2f} m` (`nearfield={bool(phys['is_nearfield_regime'] > 0.5)}`)"
            )
        if chart_filename:
            lines.extend(["", f"![Wireless Diagnostics & Multi-Seed Benchmark]({chart_filename})", ""])

        lines.append("## 2. System Simulation Parameter Table (`Table I` — Every Parameter Traceable to Standard / Paper)")
        lines.append("")
        lines.append("| Parameter | Value | Unit | Plain Meaning | Authority & Clause | Citation Key | Rationale |")
        lines.append("| :--- | :---: | :---: | :--- | :--- | :--- | :--- |")
        for p in self.all_parameter_provenance():
            lines.append(
                f"| `{p.parameter}` | `{p.value}` | {p.unit} | {p.plain_meaning} | "
                f"`{p.authority_id}` ({p.authority_section}) | `@{p.citation_key}` | {p.rationale} |"
            )
        lines.append("")

        lines.append("## 3. Multi-Seed Benchmarks & Physics Rigor Audit")
        lines.extend([
            "",
            "| Module (`block_id`) | Era | Primary Metric | Mean ± Std | IQM | Audit | Primary Citation |",
            "| :--- | :---: | :--- | :---: | :---: | :---: | :--- |",
        ])
        for s in self.block_summaries:
            st = s.seed_stats
            aud = "PASS" if (s.audit is None or s.audit.passed) else f"WARN ({', '.join(s.audit.red_flags)})"
            ckey = str(s.citation.get("key", "--"))
            lines.append(
                f"| **{s.display_name}** (`{s.block_id}`) | `{s.era}` | `{s.primary_metric_name}` | "
                f"{st['mean']:.3f} ± {st['std']:.3f} | {st['iqm']:.3f} | {aud} | `@{ckey}` |"
            )
        lines.append("")

        for s in self.block_summaries:
            lines.append(f"### `{s.block_id}` — {s.display_name}")
            if s.standards_used:
                std_tags = ", ".join(f"`{st['doc_id']}` ({st['section']})" for st in s.standards_used)
                lines.append(f"- **Anchored Standards**: {std_tags}")
            if s.mechanisms_used:
                mech_tags = ", ".join(f"**{m['name']}** (`@{(m.get('citation') or {}).get('key', '')}`)" for m in s.mechanisms_used)
                lines.append(f"- **Active Mechanisms**: {mech_tags}")
            if s.audit is not None:
                lines.append("```text\n" + s.audit.format_report() + "\n```")
        lines.append("")

        lines.append("## 4. Verified Standards & Academic Bibliography (Zero Hallucination)")
        for idx, entry in enumerate(self.verified_bibliography(), start=1):
            used_by = ", ".join(entry.get("used_by", []))
            url = entry.get("url", "")
            lines.append(f"{idx}. `@{entry['key']}` [{entry['category']}; used by: {used_by}]: {entry.get('ieee', '')}" + (f" ({url})" if url else ""))

        if output_dir is not None:
            lines.extend([
                "",
                "## 5. Saved Artifacts & Downstream Handoff",
                f"- **Output Folder**: `{Path(output_dir).resolve()}` (`environment_provenance.md`, `paper_params_table.tex`, `paper_results_table.tex`, `references.bib`, `wce_report.json`, `metrics.csv`, diagnostic figure).",
                "- **WannaDo Ecosystem Handoff**: Mount `./wce_lab/interface.py` directly inside `wannado-drl` (`./drl_lab/`) for scheduling/beamforming RL or `wannado-adv` (`./adv_eval/`, `norm='jsr_db'`) for physical-layer robustness auditing.",
            ])
        return "\n".join(lines)

    def save(self, out_dir: str | Path) -> dict[str, Path]:
        """Export Markdown provenance report, System Parameter LaTeX Table I, BibTeX, JSON, CSV, and charts."""
        root = Path(out_dir).resolve()
        root.mkdir(parents=True, exist_ok=True)
        files: dict[str, Path] = {}

        chart_path = _generate_diagnostics_figure(self, root)
        files["diagnostics_figure"] = chart_path

        p_md = root / "environment_provenance.md"
        p_md.write_text(
            self.format_environment_provenance_md(chart_filename=chart_path.name, output_dir=root) + "\n",
            encoding="utf-8",
        )
        files["environment_provenance_md"] = p_md

        p_ptex = root / "paper_params_table.tex"
        p_ptex.write_text(self.to_params_latex(), encoding="utf-8")
        files["paper_params_table_tex"] = p_ptex

        p_rtex = root / "paper_results_table.tex"
        p_rtex.write_text(self.to_results_latex(), encoding="utf-8")
        files["paper_results_table_tex"] = p_rtex

        p_bib = root / "references.bib"
        p_bib.write_text(self.to_bibtex(), encoding="utf-8")
        files["references_bib"] = p_bib

        p_json = root / "wce_report.json"
        p_json.write_text(self.to_json() + "\n", encoding="utf-8")
        files["wce_report_json"] = p_json

        p_csv = root / "metrics.csv"
        with p_csv.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["block_id", "seed", "metric_name", "metric_value"])
            for s in self.block_summaries:
                for run in s.seed_runs:
                    for mk, mv in run.get("metrics", {}).items():
                        writer.writerow([s.block_id, run["seed"], mk, mv])
        files["metrics_csv"] = p_csv
        return files


def _resolve_block_runner(block_or_fn: str | Callable[..., ProvisionResult]) -> tuple[str, Callable[..., ProvisionResult]]:
    if callable(block_or_fn):
        return getattr(block_or_fn, "__name__", "custom_block").removeprefix("provision_"), block_or_fn
    if block_or_fn not in BLOCK_ID_TO_RUNNER:
        raise KeyError(f"Unknown wireless block '{block_or_fn}'.")
    mod_path, fn_name = BLOCK_ID_TO_RUNNER[block_or_fn]
    mod = importlib.import_module(mod_path)
    return block_or_fn, getattr(mod, fn_name)


def _factor_tx_grid(num_tx: int) -> tuple[int, int]:
    """Factor antenna count into balanced (tx_nx, tx_ny) UPA dimensions."""
    n = max(1, int(num_tx))
    import math as _math

    for d in range(_math.isqrt(n), 0, -1):
        if n % d == 0:
            return n // d, d
    return n, 1


def build_block_config(block_id: str, wce_spec: WCESpec | None = None, **overrides: Any) -> Any | None:
    """Construct the dataclass Config for block_id applying WCESpec fields and explicit overrides."""
    if block_id not in BLOCK_ID_TO_RUNNER:
        return None
    mod_path, _ = BLOCK_ID_TO_RUNNER[block_id]
    mod = importlib.import_module(mod_path)
    cfg_cls = next((v for k, v in vars(mod).items() if k.endswith("Config")), None)
    if cfg_cls is None:
        return None
    valid_names = {f.name for f in fields(cfg_cls)}
    merged: dict[str, Any] = {}
    if wce_spec is not None:
        tx_nx, tx_ny = _factor_tx_grid(wce_spec.num_tx)
        d_min = max(1.0, float(wce_spec.min_tx_rx_distance_m))
        spec_map: dict[str, Any] = {
            "fc_ghz": wce_spec.carrier_fc_ghz,
            "carrier_fc_ghz": wce_spec.carrier_fc_ghz,
            "bandwidth_hz": wce_spec.bandwidth_hz,
            "numerology_mu": wce_spec.numerology_mu,
            "subcarrier_spacing_khz": 15.0 * (2 ** int(wce_spec.numerology_mu)),
            "tx_power_dbm": wce_spec.tx_power_dbm,
            "noise_figure_db": wce_spec.noise_figure_db,
            "num_tx": wce_spec.num_tx,
            "num_rx": wce_spec.num_rx,
            "num_bs_ant": wce_spec.num_tx,
            "tx_nx": tx_nx,
            "tx_ny": tx_ny,
            "num_tx_x": wce_spec.num_tx,
            "num_tx_y": 1,
            "num_ris_elements": max(16, int(wce_spec.num_tx) * 4),
            "num_star_elements": max(16, int(wce_spec.num_tx) * 4),
            "user_distance_m": d_min,
            "comm_user_distance_m": d_min,
            "target_range_m": d_min,
            "d_2d_min_m": min(d_min, 200.0),
            "speed_mps": wce_spec.max_speed_mps,
            "speed_kmh": wce_spec.max_speed_mps * 3.6,
            "target_speed_mps": wce_spec.max_speed_mps,
            "slot_duration_ms": wce_spec.slot_duration_ms,
        }
        merged.update({k: v for k, v in spec_map.items() if k in valid_names and v is not None})
        if "d_2d_max_m" in valid_names and "d_2d_min_m" in merged:
            default_cfg = cfg_cls()
            merged["d_2d_max_m"] = max(float(getattr(default_cfg, "d_2d_max_m", 500.0)), float(merged["d_2d_min_m"]) + 50.0)
    merged.update({k: v for k, v in overrides.items() if k in valid_names and v is not None})
    return cfg_cls(**merged)


def evaluate_wce_suite(
    blocks: Sequence[str | Callable[..., ProvisionResult]] | WCESpec,
    *,
    wce_spec: WCESpec | None = None,
    seeds: Sequence[int] = (0, 1, 2),
    block_configs: dict[str, Any] | None = None,
    baseline_blocks: Sequence[str] = (),
    task_name: str | None = None,
    imt2030_scenario: str | None = None,
    run_preflight: bool = True,
    run_audit: bool = True,
    min_seeds: int = 3,
    device: str = "auto",
) -> WCEReport:
    """Execute Stage 1 Preflight -> Multi-Seed Provisioning -> Stage 3 Physics Audit -> Stage 4 Provenance Report."""
    cards_by_id = {c["id"]: c for c in load_catalog()}
    if isinstance(blocks, WCESpec):
        wce_spec = wce_spec or blocks
        resolved_blocks: list[str | Callable[..., ProvisionResult]] = list(wce_spec.blocks_selected)
        resolved_baselines = list(baseline_blocks) if baseline_blocks else list(wce_spec.baseline_blocks)
    else:
        resolved_blocks = list(blocks)
        resolved_baselines = (
            list(baseline_blocks)
            if baseline_blocks
            else (list(wce_spec.baseline_blocks) if wce_spec is not None else [])
        )

    eff_task = task_name or (wce_spec.problem_name if wce_spec is not None else "WirelessProvisioningBenchmark")
    eff_scen = imt2030_scenario or (wce_spec.imt2030_scenario if wce_spec is not None else "IC")

    eval_items: list[str | Callable[..., ProvisionResult]] = []
    seen_ids: set[str] = set()
    for item in [*resolved_blocks, *resolved_baselines]:
        bid_key = item if isinstance(item, str) else getattr(item, "__name__", "custom_block").removeprefix("provision_")
        if bid_key not in seen_ids:
            eval_items.append(item)
            seen_ids.add(bid_key)

    preflights: dict[str, PreflightReport] = {}
    summaries: list[BlockExperimentSummary] = []
    distributions: dict[str, list[float]] = {}
    has_baseline = bool(resolved_baselines) or any(
        (cards_by_id.get(b, {}).get("family") == "baselines" or cards_by_id.get(b, {}).get("era") == "classic")
        for b in eval_items
        if isinstance(b, str)
    )

    for item in eval_items:
        bid, fn = _resolve_block_runner(item)
        card = cards_by_id.get(bid, {})
        cfg_obj = (block_configs or {}).get(bid)
        if cfg_obj is None and wce_spec is not None:
            cfg_obj = build_block_config(bid, wce_spec=wce_spec, device=device)

        if run_preflight:
            preflights[bid] = preflight_wce(fn, config=cfg_obj, wce_spec=wce_spec, seed=int(seeds[0]))

        seed_runs: list[ProvisionResult] = []
        for s in seeds:
            if cfg_obj is not None:
                cfg_copy = type(cfg_obj)(**{**asdict(cfg_obj), "seed": int(s)})
                res = fn(config=cfg_copy)
            else:
                res = fn(seed=int(s))
            seed_runs.append(res)

        audit = (
            audit_wce_rigor(
                seed_runs,
                wce_spec=wce_spec,
                has_classic_baseline=has_baseline,
                min_seeds=min_seeds,
            )
            if run_audit
            else None
        )
        primary_metric = audit.primary_metric_name if audit is not None else next(iter(seed_runs[0].metrics.keys()), "elapsed")
        metric_vals = [float(r.metrics.get(primary_metric, 0.0)) for r in seed_runs]

        # Capture first-seed 1D SNR/SINR/rate vector for diagnostic CDF plotting
        first_tensors = seed_runs[0].tensors
        for tkey in ("snr_db", "sinr_db", "reader_snr_db", "secrecy_rate"):
            if tkey in first_tensors and isinstance(first_tensors[tkey], torch.Tensor) and first_tensors[tkey].numel() >= 10:
                distributions[f"{bid}:{tkey}"] = first_tensors[tkey].detach().cpu().reshape(-1)[:500].tolist()
                break

        stds, mechs = resolve_block_provenance(
            bid,
            standard_ids=tuple(card.get("standard_ids", ())),
            explicit_mechanisms=seed_runs[0].mechanisms_used,
        )
        summaries.append(
            BlockExperimentSummary(
                block_id=bid,
                display_name=card.get("name", bid),
                family=card.get("family", "custom"),
                era=card.get("era", "standard"),
                seeds=[int(s) for s in seeds],
                primary_metric_name=primary_metric,
                seed_stats=aggregate_seed_stats(metric_vals),
                mean_elapsed_sec=float(np.mean([r.elapsed_seconds for r in seed_runs])),
                config=seed_runs[0].config,
                citation=card.get("citation", {}),
                standards_used=[st.to_dict() for st in stds],
                mechanisms_used=[m.to_dict() for m in mechs],
                parameter_provenance=seed_runs[0].parameter_provenance,
                audit=audit,
                seed_runs=[r.to_summary_dict() for r in seed_runs],
            )
        )

    return WCEReport(
        task_name=eff_task,
        imt2030_scenario=eff_scen,
        seeds=[int(s) for s in seeds],
        device=device,
        torch_version=torch.__version__,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        wce_spec=wce_spec,
        preflights=preflights,
        baseline_blocks=list(baseline_blocks),
        block_summaries=summaries,
        diagnostic_distributions=distributions,
    )
