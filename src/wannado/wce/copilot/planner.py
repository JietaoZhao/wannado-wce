"""Co-Pilot Planner: Arsenal showcase, 2-step goal dialog, zero-jargon explanations, runtime ETA & debrief."""

from __future__ import annotations

import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from wannado.wce.catalog.build_catalog import get_taxonomy_tree, load_catalog, query_catalog
from wannado.wce.copilot.project_inspector import ProjectScan
from wannado.wce.core.provenance import STANDARDS_PROVENANCE
from wannado.wce.core.utils import complex_normal, resolve_device
from wannado.wce.intake.archetypes import build_wce_archetype
from wannado.wce.intake.composer import compose_wce_spec
from wannado.wce.intake.jit_planner import build_jit_plan
from wannado.wce.intake.spec import WCESpec
from wannado.wce.standards.ecosystem import ECOSYSTEM_REGISTRY
from wannado.wce.standards.itu_imt2030 import match_imt2030_scenario
from wannado.wce.standards.jit_recipes import JIT_RECIPES

_FAMILY_LABELS: dict[str, str] = {
    "channel_classic": "Classic Statistical & Stochastic Geometry (经典衰落与随机几何)",
    "channel_3gpp": "3GPP NR / NTN / UAV / V2X Standardized Channels (3GPP 权威标准信道)",
    "arrays_and_ris": "Arrays, Near-Field XL-MIMO, RIS & Flexible Antennas (阵列/近场/超表面)",
    "traffic_and_mac": "3GPP Traffic Generators & Multi-Cell MAC Queues (业务流量与多小区调度)",
    "domain_6g": "IMT-2030 / 6G Frontier (ISAC / OTFS / HRLLC / A-IoT / AIAC / Security)",
    "baselines": "Analytic Baselines & CKM/RadioMap Reconstruction (解析基线与环境知识图谱)",
}


def _get_package_version() -> str:
    pkg = sys.modules.get("wannado.wce")
    return str(getattr(pkg, "__version__", "1.0.0"))


@dataclass
class RuntimeEstimate:
    """Hardware-calibrated multi-seed provisioning & audit runtime forecast."""

    device: str
    micro_bench_ms: float
    num_blocks: int
    num_seeds: int
    per_block_seconds: dict[str, float]
    total_seconds: float
    advisory: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def format_summary(self) -> str:
        lines = [
            f"[wannado-wce Runtime Forecast] device={self.device}, blocks={self.num_blocks}, seeds={self.num_seeds}"
        ]
        for bid, sec in self.per_block_seconds.items():
            lines.append(f"  - {bid:<26s}: ~{sec:.2f}s")
        lines.append(f"  Total Estimated Time: ~{self.total_seconds:.2f}s")
        if self.advisory:
            lines.append(f"  Advisory: {self.advisory}")
        return "\n".join(lines)


def get_sota_lineage(*, family: str | None = None) -> list[dict[str, Any]]:
    """Trace classic-to-frontier lineage across all 33 wireless modules."""
    cards = query_catalog(family=family)
    by_id = {c["id"]: c for c in load_catalog()}
    entries: list[dict[str, Any]] = []
    for c in cards:
        chain = [c["id"]]
        cur = c
        seen = {c["id"]}
        while cur.get("parent_id") and cur["parent_id"] not in seen:
            pid = cur["parent_id"]
            seen.add(pid)
            chain.append(pid)
            cur = by_id.get(pid, {})
            if not cur:
                break
        chain.reverse()
        entries.append(
            {
                "id": c["id"],
                "name": c["name"],
                "family": c["family"],
                "era": c["era"],
                "tier": c["tier"],
                "year": c["citation"]["year"],
                "evolution_path": " -> ".join(chain),
                "diff_from_parent": c["diff_from_parent"],
                "standard_ids": c["standard_ids"],
            }
        )
    entries.sort(key=lambda x: (x["family"], x["year"], x["id"]))
    return entries


def build_arsenal_showcase() -> dict[str, Any]:
    """Build versioned wannado-wce arsenal overview (Blocks + Standards + JIT Ecosystem)."""
    ver = _get_package_version()
    tree = get_taxonomy_tree()
    cards = load_catalog()
    era_breakdown: dict[str, list[str]] = {"classic": [], "standard": [], "frontier": []}
    families: list[dict[str, Any]] = []

    for fam, fam_cards in tree.items():
        items = []
        for c in fam_cards:
            era_breakdown.setdefault(c["era"], []).append(c["id"])
            items.append(
                {
                    "id": c["id"],
                    "name": c["name"],
                    "era": c["era"],
                    "year": c["citation"]["year"],
                    "standard_ids": c["standard_ids"],
                    "summary_plain": c["summary_plain"],
                }
            )
        families.append({"family": fam, "label": _FAMILY_LABELS.get(fam, fam), "blocks": items})

    return {
        "version": ver,
        "headline": (
            f"As of wannado-wce v{ver}, {len(cards)} executable single-file physics/traffic/baseline modules across "
            f"{len(tree)} families, {len(STANDARDS_PROVENANCE)} ITU/3GPP/ETSI/IEEE standards, and "
            f"{len(ECOSYSTEM_REGISTRY)} JIT open-source wireless tools are cataloged."
        ),
        "total_blocks": len(cards),
        "total_families": len(tree),
        "total_standards": len(STANDARDS_PROVENANCE),
        "total_ecosystem_tools": len(ECOSYSTEM_REGISTRY),
        "era_breakdown": era_breakdown,
        "families": families,
    }


def format_arsenal_overview() -> str:
    """Format a Markdown table summarizing the entire wannado-wce wireless arsenal."""
    ars = build_arsenal_showcase()
    lines = [
        f"## wannado-wce v{ars['version']} Wireless Environment Arsenal "
        f"({ars['total_blocks']} Blocks / {ars['total_standards']} Standards / {ars['total_ecosystem_tools']} JIT Tools)",
        "",
        f"- **Classic Foundations ({len(ars['era_breakdown']['classic'])})**: `{'`, `'.join(ars['era_breakdown']['classic'])}`",
        f"- **Authoritative Standards ({len(ars['era_breakdown']['standard'])})**: `{'`, `'.join(ars['era_breakdown']['standard'])}`",
        f"- **2022–2026 Frontier 6G ({len(ars['era_breakdown']['frontier'])})**: `{'`, `'.join(ars['era_breakdown']['frontier'])}`",
        "",
        "| Family | Executable Modules (Classic -> Standard -> Frontier) | Highlights |",
        "| :--- | :--- | :--- |",
    ]
    for fam in ars["families"]:
        tags = ", ".join(f"`{b['id']}` ({b['year']})" for b in fam["blocks"])
        last = fam["blocks"][-1]
        lines.append(f"| {fam['label']} | {tags} | Latest: `{last['id']}` — {last['summary_plain']} |")
    return "\n".join(lines)


def build_goal_confirmation_dialog(project_scan: ProjectScan | None = None) -> dict[str, Any]:
    """Build mandatory 2-step interactive research goal alignment dialog."""
    conf = project_scan.understanding_confidence if project_scan is not None else "low"
    summary = (
        project_scan.project_summary_guess
        if project_scan is not None and project_scan.project_summary_guess
        else "No local project directory was scanned yet; let's align on your wireless research requirements."
    )
    inferred = (
        project_scan.inferred_wce_goal
        if project_scan is not None and project_scan.inferred_wce_goal
        else "Provision a standards-anchored 3GPP/ITU wireless communication environment."
    )
    return {
        "must_explain_understanding_before_asking": True,
        "ai_understanding_summary": summary,
        "understanding_confidence": conf,
        "evidence_clues": list(project_scan.evidence_clues) if project_scan is not None else [],
        "dialog_1_confirm_goal": {
            "question": "Based on reading your project, is this the wireless communication environment you want to provision?",
            "options": [
                {
                    "id": "confirm_inferred_goal",
                    "label": (
                        f"Yes — provision: {inferred} (Recommended)"
                        if conf != "low"
                        else f"Rough framing fits — explore: {inferred}"
                    ),
                    "is_recommended": conf != "low",
                },
                {
                    "id": "custom_one_sentence_goal",
                    "label": "No / Custom — let me describe in one sentence what wireless scenario & problem I am studying",
                    "is_recommended": conf == "low",
                },
            ],
        },
        "dialog_2_one_sentence_prompt": {
            "trigger": "Ask when Option 2 is chosen or understanding_confidence == 'low'.",
            "prompt": (
                "In one sentence, what wireless communication research are you doing (e.g., frequency band, scenario such as "
                "ISAC / RIS / LEO NTN / UAV / XL-MIMO / Multi-Cell DRL, and whether you need packet-level TCP/O-RAN or tensor channels)?"
            ),
        },
    }


def _knob(
    *,
    parameter: str,
    terminology: str,
    plain_analogy: str,
    equivalent_to: str,
    if_higher_vs_lower: str,
    options: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "parameter": parameter,
        "terminology": terminology,
        "plain_analogy": plain_analogy,
        "equivalent_to": equivalent_to,
        "if_higher_vs_lower": if_higher_vs_lower,
        "options": options,
    }


def explain_parameters(
    *,
    wce_spec: WCESpec | None = None,
    output_dir: str = "./wce_lab/outputs",
) -> dict[str, Any]:
    """Return zero-jargon everyday analogies, equivalents, trade-offs, and storage options for wireless knobs."""
    spec = wce_spec or build_wce_archetype("imt2030_immersive_xr")
    phys = spec.derived_physics_summary()

    return {
        "carrier_and_bandwidth": _knob(
            parameter="carrier_fc_ghz & bandwidth_hz",
            terminology="Carrier Frequency (载频 f_c) & Bandwidth (带宽 B)",
            plain_analogy=(
                "Think of Carrier Frequency as which radio highway you drive on (3.5 GHz penetrates walls well; 28/100 GHz is a wide empty "
                "expressway blocked easily by walls/rain), and Bandwidth as the total lane width (wider B carries more data AND collects more thermal noise k_B*T*B)."
            ),
            equivalent_to=f"At B={spec.bandwidth_hz/1e6:.1f} MHz & NF={spec.noise_figure_db:.1f} dB, thermal noise floor is {phys['noise_floor_dbm']:.1f} dBm.",
            if_higher_vs_lower="Higher f_c gives huge bandwidth and tight Rayleigh focus (Z_R = 2D^2/lambda) but suffers harsh pathloss and fast Doppler.",
            options=[
                {"fc_ghz": 3.5, "bw_hz": 20e6, "label": "5G/6G Mid-band workhorse (3.5 GHz, 20 MHz) — balanced urban coverage & standard benchmarks [Recommended]"},
                {"fc_ghz": 28.0, "bw_hz": 100e6, "label": "FR2 mmWave high-capacity (28 GHz, 100 MHz) — wide sensing/ISAC & beam training"},
                {"fc_ghz": 100.0, "bw_hz": 1e9, "label": "Sub-THz 6G frontier (100 GHz, 1 GHz) — activates ITU-R P.676 gas loss & near-field spherical waves"},
            ],
        ),
        "numerology_and_mobility": _knob(
            parameter="numerology_mu & max_speed_mps",
            terminology="NR Numerology (子载波间隔 mu) & Doppler Coherence (多普勒与相干时间)",
            plain_analogy=(
                "Raising numerology mu slices each millisecond into 2^mu shorter slots with wider subcarriers—like making camera shutter speeds faster so "
                "fast-moving cars/satellites don't blur across frames."
            ),
            equivalent_to=f"Current f_d={phys['max_doppler_hz']:.1f} Hz -> channel stays roughly stable for T_c ≈ {phys['coherence_time_ms']:.2f} ms.",
            if_higher_vs_lower="Too small SCS under high speed creates inter-carrier interference (ICI); larger mu shortens slots but raises cyclic-prefix overhead.",
            options=[
                {"mu": 1, "speed_mps": 10.0, "label": "Urban pedestrian/driving (mu=1 -> 30 kHz SCS, 0.5 ms slot, 36 km/h) [Recommended]"},
                {"mu": 2, "speed_mps": 35.0, "label": "Fast highway V2X / URLLC (mu=2 -> 60 kHz SCS, 0.25 ms slot, 126 km/h)"},
                {"mu": 3, "speed_mps": 140.0, "label": "mmWave / ultra-high mobility (mu=3 -> 120 kHz SCS, 0.125 ms slot; pair OTFS for >500 km/h)"},
            ],
        ),
        "csi_assumption": _knob(
            parameter="csi_assumption",
            terminology="CSI Knowledge Assumption (信道状态信息假设: 理想 vs 真实导频估计 vs 延迟反馈)",
            plain_analogy=(
                "Driving with a clean future GPS map (Ideal Genie CSI) vs reading slightly noisy live dashboard radar (Realistic Estimated CSI). "
                "Reviewers reject papers that accidentally feed future/noiseless channels to causal algorithms!"
            ),
            equivalent_to="Determines whether downstream precoders/schedulers observe exact H_t or noisy/delayed H_hat(t).",
            if_higher_vs_lower="Genie CSI reveals theoretical upper bounds only; Realistic Estimated CSI (`ls_lmmse_csi`) prevents future leakage.",
            options=[
                {"value": "realistic_estimated", "label": "Realistic noisy pilot CSI (LS/LMMSE error + causal delay) — publication-rigorous default [Recommended]"},
                {"value": "delayed_feedback", "label": "Outdated TDD/NTN feedback CSI — stresses robustness to channel aging"},
                {"value": "ideal_genie_baseline", "label": "Ideal noiseless CSI — compute ONLY as an upper-bound reference alongside realistic CSI"},
            ],
        ),
        "traffic_model": _knob(
            parameter="traffic_model",
            terminology="Traffic Arrival Model (业务流量模型: 满队列陷阱 vs 3GPP FTP / Pareto / XR)",
            plain_analogy=(
                "Full-buffer assumes every user always has an infinite backlog (unrealistic!); 3GPP FTP/Poisson/Pareto/XR models real packets arriving in bursts "
                "with actual queue wait times and deadlines."
            ),
            equivalent_to="Follows 3GPP TR 36.814 (FTP 1/2/3, VoIP) and TR 38.838 (XR periodic frames with 10-20 ms PDB).",
            if_higher_vs_lower="Full-buffer hides delay starvation; bursty Poisson/Pareto/XR reveals real scheduler tail-latency behavior.",
            options=[
                {"value": "ftp_model_3", "label": "3GPP FTP Model 3 (Poisson burst arrivals, 0.5 MB files) — standard bursty data [Recommended]"},
                {"value": "xr_60fps", "label": "3GPP TR 38.838 XR 60 fps (10 ms PDB) — strict immersive delay-budget test"},
                {"value": "pareto", "label": "Heavy-tailed Pareto bursts — stresses scheduler queue tail stability"},
            ],
        ),
        "seeds_and_storage": _knob(
            parameter="num_seeds & output_dir",
            terminology="Multi-Seed Rigor & Traceable Output Directory (多种子统计与环境溯源保存路径)",
            plain_analogy=(
                "Repeating wireless Monte Carlo realizations across >=3 independent random seeds avoids single-seed luck, while keeping everything in "
                f"`{output_dir}` leaves your own project files 100% untouched."
            ),
            equivalent_to=f"Saves `environment_provenance.md`, `paper_params_table.tex`, `references.bib`, `wce_diagnostics.png`, and `wce_report.json` in `{output_dir}`.",
            if_higher_vs_lower="1 seed is only for wiring smoke tests; >=3 seeds computes trustworthy IQM & catches physical traps.",
            options=[
                {"seeds": [0, 1, 2], "path": output_dir, "label": f"3 random seeds + full provenance report saved to `{output_dir}` [Recommended]"},
                {"seeds": [0, 1, 2, 3, 4], "path": output_dir, "label": f"5 random seeds (publication-grade IQM) saved to `{output_dir}`"},
            ],
        ),
    }


def recommend_wce_blueprint(
    query_or_archetype: str,
    *,
    problem_name: str | None = None,
    compositional: bool | None = None,
) -> dict[str, Any]:
    """Map user's natural-language research goal or archetype ID into a complete WCESpec + 4-Slot Plan + JIT Plan."""
    q = query_or_archetype.lower()
    known_archetypes = (
        "imt2030_immersive_xr",
        "imt2030_hrllc_factory",
        "imt2030_massive_ambient_iot",
        "imt2030_ubiquitous_leo_ntn",
        "imt2030_aiac_semantic_aircomp",
        "imt2030_isac_monostatic",
        "uav_ris_relay",
        "xlmimo_nearfield_thz",
        "v2x_highway_urllc",
        "multicell_ofdma_mac",
        "jamming_contested_rf",
    )
    composed_spec, comp_plan = compose_wce_spec(query_or_archetype, problem_name=problem_name)

    distinct_signals = sum([
        any(k in q for k in ("isac", "jsac", "sensing", "radar", "通感")),
        any(k in q for k in ("near-field", "nearfield", "xl-mimo", "thz", "近场", "太赫兹")),
        any(k in q for k in ("uav", "drone", "无人机")),
        any(k in q for k in ("ris", "irs", "star-ris", "star_ris", "超表面")),
        any(k in q for k in ("leo", "ntn", "satellite", "卫星")),
        any(k in q for k in ("v2x", "v2v", "highway", "车联网")),
        any(k in q for k in ("urllc", "hrllc", "fbl", "短包", "工厂")),
        any(k in q for k in ("xr", "fps", "38.838", "沉浸式")),
        any(k in q for k in ("semantic", "aircomp", "ckm", "语义", "空中计算")),
        any(k in q for k in ("jam", "covert", "wiretap", "secrecy", "抗干扰", "隐蔽")),
    ])
    use_composed = (q not in known_archetypes) and (
        compositional is True
        or (compositional is None and (distinct_signals >= 2 or bool(comp_plan["extracted_numeric_overrides"])))
    )

    if q in known_archetypes:
        arch = q
    elif use_composed:
        arch = "composed_custom"
    elif any(k in q for k in ("isac", "jsac", "sensing", "radar", "通感")):
        arch = "imt2030_isac_monostatic"
    elif any(k in q for k in ("near-field", "nearfield", "xl-mimo", "thz", "近场", "太赫兹")):
        arch = "xlmimo_nearfield_thz"
    elif any(k in q for k in ("uav", "drone", "ris", "irs", "star-ris", "无人机", "超表面")):
        arch = "uav_ris_relay"
    elif any(k in q for k in ("leo", "ntn", "satellite", "卫星", "星地")):
        arch = "imt2030_ubiquitous_leo_ntn"
    elif any(k in q for k in ("v2x", "v2v", "highway", "车联网")):
        arch = "v2x_highway_urllc"
    elif any(k in q for k in ("urllc", "hrllc", "fbl", "factory", "短包", "工厂")):
        arch = "imt2030_hrllc_factory"
    elif any(k in q for k in ("ambient", "backscatter", "iot", "物联网", "反向散射")):
        arch = "imt2030_massive_ambient_iot"
    elif any(k in q for k in ("semantic", "aircomp", "ckm", "federated", "语义", "联邦")):
        arch = "imt2030_aiac_semantic_aircomp"
    elif any(k in q for k in ("schedul", "mac", "drl", "multi-cell", "调度", "资源分配", "强化学习")):
        arch = "multicell_ofdma_mac"
    elif any(k in q for k in ("jam", "covert", "wiretap", "secrecy", "抗干扰", "隐蔽", "安全")):
        arch = "jamming_contested_rf"
    else:
        matched = match_imt2030_scenario(q)[0]
        arch_map = {
            "IC": "imt2030_immersive_xr",
            "HRLLC": "imt2030_hrllc_factory",
            "MC": "imt2030_massive_ambient_iot",
            "UC": "imt2030_ubiquitous_leo_ntn",
            "AIAC": "imt2030_aiac_semantic_aircomp",
            "ISAC": "imt2030_isac_monostatic",
        }
        arch = arch_map.get(matched.id, "imt2030_immersive_xr")

    spec = composed_spec if use_composed else build_wce_archetype(arch, problem_name=problem_name)
    jit = build_jit_plan(
        spec,
        needs_ray_tracing=any(k in q for k in ("ray tracing", "ray-tracing", "deepmimo", "sionna rt", "射线追踪")),
        needs_full_protocol_stack=any(k in q for k in ("ns-3", "ns3", "tcp", "rlc", "pdcp", "5g-lena", "simu5g", "协议栈")),
        needs_real_sdr_oran=any(k in q for k in ("oai", "openairinterface", "srsran", "usrp", "oran", "o-ran", "ric")),
        needs_rf_iq_ml=any(k in q for k in ("torchsig", "modulation", "amc", "调制识别")),
        needs_isac_waveform="isac" in q or "isac" in arch,
    )
    return {
        "archetype": arch,
        "composition_plan": comp_plan,
        "wce_spec": spec.to_dict(),
        "jit_plan": jit.to_dict(),
        "jit_recipes_available": sorted(JIT_RECIPES.keys()),
        "spec_obj": spec,
        "jit_obj": jit,
    }


def estimate_runtime(
    blocks: Sequence[str] | WCESpec,
    *,
    num_seeds: int = 3,
    device: str = "auto",
) -> RuntimeEstimate:
    """Forecast multi-seed provisioning & audit ETA via a tiny complex64 micro-benchmark."""
    block_ids = list(blocks.blocks_selected) if isinstance(blocks, WCESpec) else list(blocks)
    dev = resolve_device(device)
    t0 = time.perf_counter()
    for _ in range(5):
        z = complex_normal((128, 16), device=dev)
        _ = torch.fft.fft(z, dim=-1).abs().sum()
    micro_ms = ((time.perf_counter() - t0) * 1000.0) / 5.0

    per_block: dict[str, float] = {}
    for bid in block_ids:
        # WMMSE / CKM / CDL-TDL take slightly longer than simple link formulas
        base_ms = 25.0 if bid in ("wmmse_precoder", "ckm_radiomap_bridge", "tr38901_cdl_tdl", "ppp_stochastic_geo") else 8.0
        sec = max(0.05, ((base_ms + micro_ms) * num_seeds) / 1000.0)
        per_block[bid] = round(sec, 2)
    total = round(sum(per_block.values()), 2)
    advisory = (
        "Instant built-in Tier 0 provisioning (< 5s); ready to run immediately."
        if total < 5.0
        else f"Moderate multi-seed sweep (~{total:.1f}s); confirm external JIT downloads separately if selected."
    )
    return RuntimeEstimate(
        device=str(dev),
        micro_bench_ms=round(micro_ms, 3),
        num_blocks=len(block_ids),
        num_seeds=num_seeds,
        per_block_seconds=per_block,
        total_seconds=total,
        advisory=advisory,
    )


def build_interactive_brief(
    *,
    project_scan: ProjectScan | None = None,
    query: str | None = None,
    num_seeds: int = 3,
    output_dir: str = "./wce_lab/outputs",
) -> dict[str, Any]:
    """Assemble full Stage 0-2 AI Co-Pilot brief (clues, arsenal, 2-step dialog, JIT plan, zero-jargon knobs, ETA)."""
    ars = build_arsenal_showcase()
    dlg = build_goal_confirmation_dialog(project_scan)
    target_hint = query or (project_scan.likely_archetype if project_scan is not None else "imt2030_immersive_xr")
    bp = recommend_wce_blueprint(target_hint)
    spec_obj: WCESpec = bp["spec_obj"]
    jit_obj = bp["jit_obj"]
    param_guide = explain_parameters(wce_spec=spec_obj, output_dir=output_dir)
    eta = estimate_runtime(spec_obj, num_seeds=num_seeds)

    return {
        "interaction_guidelines": {
            "language_rule": "Reply naturally in the user's conversation language (e.g., natural Chinese when chatting in Chinese).",
            "traverse_first_rule": "Share observed project clues FIRST before asking; never cold-ask without summarizing what was scanned.",
            "arsenal_showcase_rule": f"Showcase wannado-wce v{ars['version']} ({ars['total_blocks']} blocks, {ars['total_standards']} standards, {ars['total_ecosystem_tools']} JIT tools).",
            "goal_dialog_rule": "Pop `goal_confirmation_dialog` BEFORE finalizing ./wce_lab/wce_spec.json or pulling external packages.",
            "jit_philosophy_rule": "Never auto-install large external simulators without confirming disk/OS/GPU requirements in `jit_plan`.",
            "zero_jargon_rule": "Explain every RF parameter with `plain_analogy`, `equivalent_to`, and `if_higher_vs_lower` before running.",
        },
        "project_scan": project_scan.to_dict() if project_scan is not None else None,
        "arsenal_showcase": ars,
        "goal_confirmation_dialog": dlg,
        "recommended_archetype": bp["archetype"],
        "composition_plan": bp.get("composition_plan"),
        "wce_spec": bp["wce_spec"],
        "jit_plan": bp["jit_plan"],
        "jit_recipes_available": bp.get("jit_recipes_available", []),
        "parameter_explanations": param_guide,
        "questions_to_ask_user": list(param_guide.values()),
        "runtime_estimate": eta.to_dict(),
    }


def build_post_run_debrief(report: Any, *, output_dir: str | Path | None = None) -> dict[str, Any]:
    """Build plain-language Stage 4 verdict, standard/paper provenance list, and next-step actions."""
    summaries = list(getattr(report, "block_summaries", []))
    bib = report.verified_bibliography() if hasattr(report, "verified_bibliography") else []
    verdicts: list[str] = []
    all_flags: list[str] = []
    for s in summaries:
        flags = list(s.audit.red_flags) if s.audit is not None else []
        all_flags.extend(flags)
        status = "passed all physical conservation & multi-seed rigor checks" if not flags else f"flagged `{', '.join(flags)}`"
        verdicts.append(
            f"{s.display_name} (`{s.block_id}`): {s.primary_metric_name} IQM = {s.seed_stats.get('iqm', 0.0):.3f} "
            f"(mean {s.seed_stats.get('mean', 0.0):.3f} ± {s.seed_stats.get('std', 0.0):.3f}) — {status}."
        )
    out_str = str(Path(output_dir).resolve()) if output_dir is not None else "./wce_lab/outputs"
    return {
        "plain_verdict": " ".join(verdicts) if verdicts else "No blocks provisioned yet.",
        "red_flags": list(dict.fromkeys(all_flags)),
        "verified_bibliography_count": len(bib),
        "storage_directory": out_str,
        "handoff_hints": [
            "Exported Gym / PyTorch interfaces inside `./wce_lab/interface.py` are ready for `wannado-drl` (`./drl_lab/`) or `wannado-adv` (`./adv_eval/`).",
        ],
    }
