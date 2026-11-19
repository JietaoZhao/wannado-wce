"""Command-line interface for the wannado-wce standards-anchored wireless environment provisioning platform."""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections.abc import Sequence
from pathlib import Path

from wannado.wce.agent_kit import install_agent_kit
from wannado.wce.audit import preflight_wce, verify_wce_target
from wannado.wce.catalog.build_catalog import build_registry_json, query_catalog
from wannado.wce.copilot import (
    build_arsenal_showcase,
    build_interactive_brief,
    format_arsenal_overview,
    get_sota_lineage,
    recommend_wce_blueprint,
    scan_project,
)
from wannado.wce.core.provenance import STANDARDS_PROVENANCE
from wannado.wce.intake import (
    WCESpec,
    audit_wce_spec,
    build_wce_archetype,
    compose_wce_spec,
    update_wce_spec_with_diff,
)
from wannado.wce.reporter import export_citations
from wannado.wce.standards.ecosystem import ECOSYSTEM_REGISTRY, format_ecosystem_table, probe_local_ecosystem
from wannado.wce.standards.jit_recipes import (
    JIT_RECIPES,
    check_tool_platform_readiness,
    verify_jit_adapter_file,
    write_jit_adapter_bundle,
)
from wannado.wce.workspace import eject_block, scaffold_wce_lab


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wannado-wce",
        description="Standards-anchored, Just-in-Time wireless communication environment quick provisioning.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_inspect = sub.add_parser("inspect", help="Scan project, surface wireless clues, arsenal, goal dialog, and JIT plan.")
    p_inspect.add_argument("project_dir", nargs="?", default=".")
    p_inspect.add_argument("--query", default=None, help="Optional one-sentence research topic override.")
    p_inspect.add_argument("--json", action="store_true", dest="as_json")

    p_ars = sub.add_parser("arsenal", help="Display versioned wannado-wce 33-block + 21-standard + 18-tool arsenal.")
    p_ars.add_argument("--json", action="store_true", dest="as_json")

    p_std = sub.add_parser("standards", help="List built-in ITU-R, 3GPP, ETSI, and IEEE authoritative standards.")
    p_std.add_argument("--org", default=None, help="Filter by organization (ITU-R, 3GPP, ETSI, IEEE).")
    p_std.add_argument("--json", action="store_true", dest="as_json")

    p_eco = sub.add_parser("ecosystem", help="Inspect open-source wireless tools & probe local OS/GPU/Python availability.")
    p_eco.add_argument("--probe", action="store_true", help="Include local installed tool probe.")
    p_eco.add_argument("--json", action="store_true", dest="as_json")

    p_intake = sub.add_parser("intake", help="Generate & audit a Stage 0 WCESpec (wce_spec.json) from an archetype.")
    p_intake.add_argument("--archetype", default="imt2030_immersive_xr")
    p_intake.add_argument("--name", default=None)
    p_intake.add_argument("--out", default="./wce_lab/wce_spec.json")
    p_intake.add_argument("--json", action="store_true", dest="as_json")

    p_cat = sub.add_parser("catalog", help="Search the 33 executable single-file wireless blocks.")
    p_cat.add_argument("--family", default=None)
    p_cat.add_argument("--era", default=None)
    p_cat.add_argument("--tier", default=None)
    p_cat.add_argument("--standard", default=None)
    p_cat.add_argument("--rebuild", action="store_true")
    p_cat.add_argument("--json", action="store_true", dest="as_json")

    p_lin = sub.add_parser("lineage", help="Inspect evolution chains from classic fading to 6G frontier modules.")
    p_lin.add_argument("--family", default=None)
    p_lin.add_argument("--json", action="store_true", dest="as_json")

    p_rec = sub.add_parser("recommend", help="Map a natural-language wireless goal to a WCESpec + JIT plan.")
    p_rec.add_argument("query", nargs="+")
    p_rec.add_argument("--json", action="store_true", dest="as_json")

    p_comp = sub.add_parser("compose", help="4-Slot Compositional Assembler parsing free-form multi-tech goals.")
    p_comp.add_argument("query", nargs="+")
    p_comp.add_argument("--name", default=None)
    p_comp.add_argument("--out", default=None, help="Optional path to save composed wce_spec.json.")
    p_comp.add_argument("--json", action="store_true", dest="as_json")

    p_diff = sub.add_parser("diff-spec", help="Apply live parameter edits with LiveProvenance Diff & boundary alerts.")
    p_diff.add_argument("--spec", default="./wce_lab/wce_spec.json")
    p_diff.add_argument("--request", default=None, help="Natural-language update e.g. 'Switch frequency to 140 GHz'.")
    p_diff.add_argument("--set", nargs="*", default=[], dest="kv_sets", help="key=value overrides.")
    p_diff.add_argument("--save", action="store_true", help="Write updated WCESpec back to --spec path.")
    p_diff.add_argument("--json", action="store_true", dest="as_json")

    p_jit = sub.add_parser("jit-recipe", help="Check platform/disk readiness, generate adapter glue, and run --self-test.")
    p_jit.add_argument("tool_id", choices=sorted(JIT_RECIPES.keys()))
    p_jit.add_argument("--write-dir", default=None, dest="write_dir", help="Directory to write verified adapter script.")
    p_jit.add_argument("--scenario", default="asu_campus_3p5", help="DeepMIMO scenario disk check target.")
    p_jit.add_argument("--verify", action="store_true", help="Execute adapter --self-test after writing.")
    p_jit.add_argument("--require-live", action="store_true", dest="require_live", help="Fail verification unless the real external library is installed and passes live import.")
    p_jit.add_argument("--json", action="store_true", dest="as_json")

    p_cite = sub.add_parser("cite", help="Export verbatim verified BibTeX/IEEE/APA citations.")
    p_cite.add_argument("ids", nargs="+", help="Block, standard, or mechanism IDs.")
    p_cite.add_argument("--format", default="bibtex", choices=["bibtex", "ieee", "apa"])
    p_cite.add_argument("--no-standards", action="store_true")
    p_cite.add_argument("--no-mechanisms", action="store_true")

    p_eject = sub.add_parser("eject", help="Copy ONE standalone single-file block into ./wce_lab/blocks/ for custom edits.")
    p_eject.add_argument("block_id")
    p_eject.add_argument("--dest", default="./wce_lab/blocks")
    p_eject.add_argument("--name", default=None)
    p_eject.add_argument("--overwrite", action="store_true")

    p_scaf = sub.add_parser("scaffold", help="Generate non-intrusive external ./wce_lab/ harness.")
    p_scaf.add_argument("--dir", default="./wce_lab", dest="target_dir")
    p_scaf.add_argument("--archetype", default="imt2030_immersive_xr")
    p_scaf.add_argument("--query", default=None, help="Optional free-form goal using 4-slot Compositional Assembler.")
    p_scaf.add_argument("--name", default=None)
    p_scaf.add_argument("--overwrite", action="store_true")

    p_pf = sub.add_parser("preflight", help="Run Stage 1 physics sanity & speed preflight on a block or wce_lab/interface.py.")
    p_pf.add_argument("--block", default=None, help="Block ID to preflight directly.")
    p_pf.add_argument("--interface", default="./wce_lab/interface.py", dest="interface_path")
    p_pf.add_argument("--json", action="store_true", dest="as_json")

    p_ver = sub.add_parser(
        "verify",
        help="One-command machine gate: Stage 0 spec + Preflight + Rigor 2.0 + Link-Level PHY validation.",
    )
    p_ver.add_argument("--dir", default=None, dest="lab_dir", help="Path to ./wce_lab directory (auto-resolves wce_spec.json or interface.py).")
    p_ver.add_argument("--spec", default=None, dest="spec_path", help="Path to saved WCESpec (.json/.yaml).")
    p_ver.add_argument("--interface", default=None, dest="interface_path", help="Path to ./wce_lab/interface.py.")
    p_ver.add_argument("--query", default=None, help="Natural-language wireless prompt to compose & verify.")
    p_ver.add_argument("--blocks", nargs="*", default=None, help="Direct wireless block IDs to verify.")
    p_ver.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2], help="Seeds for multi-seed rigor verification.")
    p_ver.add_argument("--min-seeds", type=int, default=3, dest="min_seeds", help="Minimum seeds required by Stage 3 Rigor Auditor.")
    p_ver.add_argument("--phy-snr-db", type=float, default=12.0, help="SNR (dB) for Link-Level OFDM-MIMO PHY smoke check.")
    p_ver.add_argument("--no-phy-check", action="store_true", help="Skip Link-Level OFDM-MIMO PHY smoke test.")
    p_ver.add_argument("--strict-warnings", action="store_true", help="Fail gate on any Stage 0/1 advisory warning.")
    p_ver.add_argument("--json", action="store_true", dest="as_json")

    p_init = sub.add_parser("init-agent", help="Install universal AGENTS.md / Agent Skills rules + IDE adapters.")
    p_init.add_argument("--dir", default=".", dest="project_dir")
    p_init.add_argument("--ide", default="all", choices=["all", "universal", "cursor", "antigravity", "claude", "codex"])

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "inspect":
        scan = scan_project(args.project_dir)
        brief = build_interactive_brief(project_scan=scan, query=args.query)
        if args.as_json:
            print(json.dumps(brief, indent=2, ensure_ascii=False))
        else:
            print(scan.format_summary())
            ars = brief["arsenal_showcase"]
            print(f"\n[wannado-wce v{ars['version']} Arsenal]\n  {ars['headline']}")
            dlg = brief["goal_confirmation_dialog"]
            print("\n[Goal Confirmation Dialog]")
            print(f"  AI Understanding: {dlg['ai_understanding_summary']}")
            for idx, opt in enumerate(dlg["dialog_1_confirm_goal"]["options"], start=1):
                print(f"    [{idx}] {opt['label']}")
            print(f"\n[Recommended Blueprint: archetype='{brief['recommended_archetype']}']")
            print(f"  Blocks: {', '.join(brief['wce_spec']['blocks_selected'])}")
            print(f"  Baselines: {', '.join(brief['wce_spec']['baseline_blocks'])}")
            if brief.get("clarifying_forks"):
                print("\n[Minimal Clarifying Forks (1-2 targeted questions)]:")
                for idx, fk in enumerate(brief["clarifying_forks"], start=1):
                    print(f"  Q{idx}. {fk['question']}")
                    for opt in fk["options"]:
                        print(f"      - {opt}")
            rd = (brief.get("jit_plan") or {}).get("readiness_reports") or []
            if rd:
                print("\n[External JIT Readiness & Disk Check]:")
                for r in rd:
                    print(
                        f"  * {r['tool_id']}: ready={r['ready_to_pull']}, need~{r['estimated_required_gb']}GB, "
                        f"free={r['free_disk_gb']}GB, OS={r['os']}, CUDA={r['cuda_available']} | cmd: {'; '.join(r['install_commands'])}"
                    )
                    for w in r.get("warnings", []):
                        print(f"      ! Advisory: {w}")
                    for b in r.get("blockers", []):
                        print(f"      X Blocker : {b}")
            dm_cat = (brief.get("jit_plan") or {}).get("deepmimo_catalog") or {}
            if dm_cat:
                print("  * DeepMIMO Scenario Sizes: " + ", ".join(f"{k}(~{v['est_disk_gb']}GB)" for k, v in dm_cat.items()))
            print(f"\n  ETA: ~{brief['runtime_estimate']['total_seconds']:.2f}s ({brief['runtime_estimate']['advisory']})")
        return 0

    if args.command == "arsenal":
        print(json.dumps(build_arsenal_showcase(), indent=2, ensure_ascii=False) if args.as_json else format_arsenal_overview())
        return 0

    if args.command == "standards":
        items = [s.to_dict() for s in STANDARDS_PROVENANCE.values() if args.org is None or s.org.lower() == args.org.lower()]
        if args.as_json:
            print(json.dumps(items, indent=2, ensure_ascii=False))
        else:
            for s in items:
                print(f"  [{s['org']}] {s['id']:<18s} | {s['doc_id']} ({s['year']}) — {s['summary_plain']}")
        return 0

    if args.command == "ecosystem":
        payload: dict[str, object] = {"tools": [t.to_dict() for t in ECOSYSTEM_REGISTRY.values()]}
        if args.probe:
            payload["local_probe"] = probe_local_ecosystem()
        if args.as_json:
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        else:
            print(format_ecosystem_table())
            if args.probe:
                print("\n[Local Probe]:", json.dumps(payload["local_probe"], indent=2))
        return 0

    if args.command == "intake":
        spec = build_wce_archetype(args.archetype, problem_name=args.name)
        aud = audit_wce_spec(spec)
        out_path = spec.save(args.out)
        if args.as_json:
            print(json.dumps({"saved_to": str(out_path), "wce_spec": spec.to_dict(), "audit": aud.to_dict()}, indent=2, ensure_ascii=False))
        else:
            print(spec.format_system_parameter_table_md())
            print(f"\nSaved WCESpec to: {out_path} (audit ok={aud.ok})")
        return 0

    if args.command == "catalog":
        if args.rebuild:
            build_registry_json()
        cards = query_catalog(family=args.family, era=args.era, tier=args.tier, standard_id=args.standard)
        if args.as_json:
            print(json.dumps(cards, indent=2, ensure_ascii=False))
        else:
            print(f"Matched {len(cards)} blocks:")
            for c in cards:
                print(f"  [{c['era']}][{c['family']}] {c['id']:<28s} — {c['summary_plain']}")
        return 0

    if args.command == "lineage":
        entries = get_sota_lineage(family=args.family)
        if args.as_json:
            print(json.dumps(entries, indent=2, ensure_ascii=False))
        else:
            for e in entries:
                print(f"  [{e['year']}][{e['era']}][{e['family']}] {e['id']:<26s} | {e['evolution_path']}")
        return 0

    if args.command == "recommend":
        bp = recommend_wce_blueprint(" ".join(args.query))
        out = {
            "archetype": bp["archetype"],
            "composition_plan": bp.get("composition_plan"),
            "wce_spec": bp["wce_spec"],
            "jit_plan": bp["jit_plan"],
        }
        if args.as_json:
            print(json.dumps(out, indent=2, ensure_ascii=False))
        else:
            print(f"Matched Archetype: {bp['archetype']}")
            print(bp["jit_obj"].format_summary())
        return 0

    if args.command == "compose":
        spec, plan = compose_wce_spec(" ".join(args.query), problem_name=args.name)
        aud = audit_wce_spec(spec)
        saved_to = str(spec.save(args.out)) if args.out else None
        payload = {"saved_to": saved_to, "composition_plan": plan, "wce_spec": spec.to_dict(), "audit": aud.to_dict()}
        if args.as_json:
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        else:
            print("[4-Slot Compositional Wireless Plan]")
            for sl in plan["slots"]:
                print(f"  - {sl['slot_name']}: {', '.join(sl['blocks'])}  ({sl['rationale']})")
            if plan["boundary_alerts"]:
                print("\n[Coupled Physics & Standard Alerts]:")
                for a in plan["boundary_alerts"]:
                    print(f"  ! {a}")
            print("\n" + spec.format_system_parameter_table_md())
        return 0

    if args.command == "diff-spec":
        spec = WCESpec.load(args.spec)
        updates: dict[str, object] = {}
        if args.request:
            from wannado.wce.intake.composer import parse_param_updates_from_text

            updates.update(parse_param_updates_from_text(args.request))
        for item in args.kv_sets:
            if "=" in item:
                k, raw_v = item.split("=", 1)
                k = k.strip()
                try:
                    updates[k] = float(raw_v) if ("." in raw_v or "e" in raw_v.lower()) else int(raw_v)
                except ValueError:
                    updates[k] = raw_v
        diff_rep = update_wce_spec_with_diff(spec, updates)
        if args.save:
            diff_rep.updated_spec.save(args.spec)
        if args.as_json:
            print(json.dumps(diff_rep.to_dict(), indent=2, ensure_ascii=False))
        else:
            print(diff_rep.format_summary())
            print("\n" + diff_rep.updated_table1_md)
        return 0

    if args.command == "jit-recipe":
        readiness = check_tool_platform_readiness(args.tool_id, deepmimo_scenario=args.scenario)
        written_files: dict[str, str] = {}
        verify_res: dict[str, object] | None = None
        if args.write_dir or args.verify:
            target = Path(args.write_dir) if args.write_dir else Path("./wce_lab/adapters")
            bundle = write_jit_adapter_bundle(args.tool_id, dest_dir=target, overwrite=True)
            written_files = {k: str(v) for k, v in bundle.items()}
            if args.verify and "adapter_py" in bundle:
                verify_res = verify_jit_adapter_file(bundle["adapter_py"], require_live=args.require_live)
        payload = {"readiness": readiness, "written_files": written_files, "verification": verify_res}
        if args.as_json:
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        else:
            print(f"[wannado-wce JIT Recipe: {args.tool_id}] ready_to_pull={readiness['ready_to_pull']}, free_disk={readiness['free_disk_gb']} GB")
            for w in readiness["warnings"]:
                print(f"  Warning: {w}")
            for b in readiness["blockers"]:
                print(f"  Blocker: {b}")
            for k, p in written_files.items():
                print(f"  Wrote {k}: {p}")
            if verify_res is not None:
                print(f"  Adapter --self-test ok={verify_res.get('ok')}")
        if verify_res is not None and not verify_res.get("ok"):
            return 1
        return 0

    if args.command == "cite":
        print(export_citations(args.ids, fmt=args.format, include_standards=not args.no_standards, include_mechanisms=not args.no_mechanisms))
        return 0

    if args.command == "eject":
        dest = eject_block(args.block_id, dest_dir=args.dest, custom_name=args.name, overwrite=args.overwrite)
        print(f"Ejected single-file wireless block to: {dest}")
        return 0

    if args.command == "scaffold":
        written = scaffold_wce_lab(target_dir=args.target_dir, archetype=args.archetype, query=args.query, problem_name=args.name, overwrite=args.overwrite)
        print(f"Scaffolded isolated wce_lab workspace under {Path(args.target_dir).resolve()}:")
        for role, path in written.items():
            print(f"  - {role}: {path}")
        return 0

    if args.command == "preflight":
        if args.block:
            pf = preflight_wce(args.block)
            print(json.dumps(pf.to_dict(), indent=2, ensure_ascii=False) if args.as_json else pf.format_report())
            return 0 if pf.ok else 1
        iface_path = Path(args.interface_path).resolve()
        mod_spec = importlib.util.spec_from_file_location("_wannado_wce_user_iface", iface_path)
        if mod_spec is None or mod_spec.loader is None:
            raise FileNotFoundError(f"Cannot load interface module at {iface_path}")
        mod = importlib.util.module_from_spec(mod_spec)
        mod_spec.loader.exec_module(mod)
        iface = mod.load_wireless_interface()
        reports = [preflight_wce(bid, wce_spec=iface.wce_spec) for bid in iface.blocks]
        if args.as_json:
            print(json.dumps([r.to_dict() for r in reports], indent=2, ensure_ascii=False))
        else:
            for r in reports:
                print(r.format_report())
        return 0 if all(r.ok for r in reports) else 1

    if args.command == "verify":
        spec_path = args.spec_path
        interface_path = args.interface_path
        if args.lab_dir and spec_path is None and interface_path is None:
            lab_root = Path(args.lab_dir).resolve()
            if (lab_root / "interface.py").is_file():
                interface_path = str(lab_root / "interface.py")
            elif (lab_root / "wce_spec.json").is_file():
                spec_path = str(lab_root / "wce_spec.json")
            elif (lab_root / "wce_spec.yaml").is_file():
                spec_path = str(lab_root / "wce_spec.yaml")
            else:
                raise FileNotFoundError(f"No interface.py or wce_spec.json/.yaml found under {lab_root}")
        rep = verify_wce_target(
            spec_path=spec_path,
            interface_path=interface_path,
            query=args.query,
            blocks=args.blocks,
            seeds=args.seeds,
            min_seeds=args.min_seeds,
            with_phy_check=not args.no_phy_check,
            phy_snr_db=args.phy_snr_db,
            strict_warnings=args.strict_warnings,
        )
        print(json.dumps(rep.to_dict(), indent=2, ensure_ascii=False) if args.as_json else rep.format_report())
        return rep.exit_code

    if args.command == "init-agent":
        written = install_agent_kit(project_dir=args.project_dir, ide=args.ide)
        print(f"Installed wannado-wce AI Co-Pilot skill/rules for ide='{args.ide}':")
        for role, path in written.items():
            print(f"  - {role}: {path}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
