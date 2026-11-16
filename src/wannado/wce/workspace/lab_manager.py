"""Non-intrusive external plug-in workspace (`./wce_lab/`) and single-file CleanRL block ejection."""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from wannado.wce.blocks import BLOCK_ID_TO_RUNNER
from wannado.wce.catalog.build_catalog import WCE_ROOT, load_catalog
from wannado.wce.intake.archetypes import build_wce_archetype
from wannado.wce.intake.composer import compose_wce_spec
from wannado.wce.intake.jit_planner import build_jit_plan
from wannado.wce.intake.spec import WCESpec
from wannado.wce.standards.jit_recipes import JIT_RECIPES, write_jit_adapter_bundle
from wannado.wce.workspace.composed_env import ComposedWirelessEnv


@dataclass
class WirelessEnvInterface:
    """Single clean boundary connecting provisioned wireless environments to user scripts / DRL / Adv."""

    wce_spec: WCESpec
    blocks: list[str]
    baseline_blocks: list[str] = field(default_factory=list)
    task_name: str = "MyWirelessEnv"
    extra_metadata: dict[str, Any] = field(default_factory=dict)

    def build_composed_env(self, *, batch_size: int = 32, seed: int = 42) -> ComposedWirelessEnv:
        """Instantiate forward-coupled ComposedWirelessEnv chaining all selected pipeline blocks."""
        return ComposedWirelessEnv(self.wce_spec, batch_size=batch_size, seed=seed)


def _resolve_block_source_path(block_id: str) -> tuple[dict[str, Any], Path]:
    cards = {c["id"]: c for c in load_catalog()}
    card = cards.get(block_id)
    mod_path = card["module_path"] if card is not None else None
    if mod_path is None and block_id in BLOCK_ID_TO_RUNNER:
        mod_path, _ = BLOCK_ID_TO_RUNNER[block_id]
        mod = importlib.import_module(mod_path)
        meta = getattr(mod, "META", None)
        if meta is not None:
            card = meta.to_dict()
    if card is None or mod_path is None:
        sample = ", ".join(sorted(cards.keys())[:10])
        raise KeyError(f"Unknown block '{block_id}'. Sample valid IDs: {sample} ...")

    if mod_path.startswith("wannado.wce."):
        rel_parts = mod_path.removeprefix("wannado.wce.").split(".")
        src_path = (WCE_ROOT / Path(*rel_parts)).with_suffix(".py")
        if src_path.exists():
            return card, src_path
    mod = importlib.import_module(mod_path)
    if mod.__file__ is None:
        raise FileNotFoundError(f"Cannot locate source file for module '{mod_path}'.")
    return card, Path(mod.__file__).resolve()


def eject_block(
    block_id: str,
    dest_dir: str | Path = "./wce_lab/blocks",
    *,
    custom_name: str | None = None,
    overwrite: bool = False,
) -> Path:
    """Copy ONE standalone single-file physics/traffic/6G block into ./wce_lab/blocks/ for custom paper edits."""
    _, src_path = _resolve_block_source_path(block_id)
    target_dir = Path(dest_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    filename = custom_name if custom_name is not None else src_path.name
    if not filename.endswith(".py"):
        filename = f"{filename}.py"
    dest_path = target_dir / filename

    if dest_path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing customized block at {dest_path}. Pass overwrite=True to replace."
        )

    banner = (
        f"# [wannado-wce Ejected Single-File Wireless Block: id='{block_id}']\n"
        f"# Edit ONLY this standalone file to prototype custom channel/physics/traffic models\n"
        f"# from new papers without polluting your project. Update `META` / `ParamProvenance` as needed.\n\n"
    )
    dest_path.write_text(banner + src_path.read_text(encoding="utf-8"), encoding="utf-8")
    return dest_path


def _interface_template(spec: WCESpec) -> str:
    return f'''"""Isolated wireless environment provisioning boundary (`wce_lab/interface.py`).

Loads Stage 0 `wce_spec.json` and exposes `load_wireless_interface()` plus forward-coupled
`load_composed_env()` without modifying your downstream scripts, DRL trainers, or Adv testers.
"""

from __future__ import annotations

import sys
from pathlib import Path

from wannado.wce.intake import WCESpec
from wannado.wce.workspace import ComposedWirelessEnv, WirelessEnvInterface

_LAB_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _LAB_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def load_wireless_interface() -> WirelessEnvInterface:
    """Load the standards-anchored wireless environment specification and block lineup."""
    spec = WCESpec.load(_LAB_DIR / "wce_spec.json")
    return WirelessEnvInterface(
        wce_spec=spec,
        blocks=list(spec.blocks_selected),
        baseline_blocks=list(spec.baseline_blocks),
        task_name=spec.problem_name,
    )


def load_composed_env(*, batch_size: int = 32, seed: int = 42) -> ComposedWirelessEnv:
    """Build forward-coupled ComposedWirelessEnv ready for ``for slot in range(T): env.step()``."""
    return load_wireless_interface().build_composed_env(batch_size=batch_size, seed=seed)
'''


def _runner_template() -> str:
    return '''"""Standalone non-intrusive wireless environment provisioning & reporting runner (`wce_lab/run_provision.py`).

Run directly from your project root:
    python wce_lab/run_provision.py --seeds 0 1 2
"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .interface import load_composed_env, load_wireless_interface
except ImportError:
    from interface import load_composed_env, load_wireless_interface

from wannado.wce.audit import audit_wce_rigor, preflight_wce
from wannado.wce.copilot import build_post_run_debrief
from wannado.wce.reporter import evaluate_wce_suite


def main() -> None:
    parser = argparse.ArgumentParser(description="Run isolated wannado-wce Stage 1-4 provisioning & provenance export.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--slots", type=int, default=20)
    parser.add_argument("--no-audit", action="store_true")
    args = parser.parse_args()

    iface = load_wireless_interface()
    all_blocks = list(dict.fromkeys(iface.blocks + iface.baseline_blocks))
    for bid in all_blocks:
        pf = preflight_wce(bid, wce_spec=iface.wce_spec, seed=args.seeds[0])
        print(pf.format_report())
        if not pf.ok:
            raise SystemExit(1)

    # Execute forward-coupled ComposedWirelessEnv multi-seed verification
    traj_runs = [load_composed_env(seed=s).to_rigor_result(num_slots=args.slots, seed=s) for s in args.seeds]
    traj_audit = audit_wce_rigor(traj_runs, wce_spec=iface.wce_spec, has_classic_baseline=bool(iface.baseline_blocks), min_seeds=len(args.seeds))
    print("\\n[ComposedWirelessEnv Forward Pipeline Check]:")
    print(traj_audit.format_report())

    report = evaluate_wce_suite(
        all_blocks,
        wce_spec=iface.wce_spec,
        seeds=args.seeds,
        baseline_blocks=iface.baseline_blocks,
        run_audit=not args.no_audit,
    )
    out_dir = Path(args.output_dir) if args.output_dir else (Path(__file__).resolve().parent / "outputs")
    saved = report.save(out_dir)
    debrief = build_post_run_debrief(report, output_dir=out_dir)
    print("\\n[Plain Verdict]:", debrief["plain_verdict"])
    print("[Saved Artifacts]:")
    for k, v in saved.items():
        print(f"  - {k}: {v}")


if __name__ == "__main__":
    main()
'''


def scaffold_wce_lab(
    target_dir: str | Path = "./wce_lab",
    *,
    archetype: str = "imt2030_immersive_xr",
    query: str | None = None,
    problem_name: str | None = None,
    wce_spec: WCESpec | None = None,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Create non-intrusive ./wce_lab/ harness (wce_spec.json, interface.py, run_provision.py, jit_setup.sh, adapters/)."""
    lab_dir = Path(target_dir).resolve()
    lab_dir.mkdir(parents=True, exist_ok=True)
    if wce_spec is not None:
        spec = wce_spec
    elif query:
        spec, _ = compose_wce_spec(query, problem_name=problem_name)
    else:
        spec = build_wce_archetype(archetype, problem_name=problem_name)

    written: dict[str, Path] = {}
    spec_path = lab_dir / "wce_spec.json"
    if not spec_path.exists() or overwrite:
        spec.save(spec_path)
    written["wce_spec_json"] = spec_path

    iface_path = lab_dir / "interface.py"
    if not iface_path.exists() or overwrite:
        iface_path.write_text(_interface_template(spec), encoding="utf-8")
    written["interface_py"] = iface_path

    runner_path = lab_dir / "run_provision.py"
    if not runner_path.exists() or overwrite:
        runner_path.write_text(_runner_template(), encoding="utf-8")
    written["run_provision_py"] = runner_path

    jit = build_jit_plan(spec)
    jit_script = lab_dir / "jit_setup.sh"
    if not jit_script.exists() or overwrite:
        lines = [
            "#!/usr/bin/env bash",
            "# [wannado-wce Just-in-Time External Tool Pull & Adapter Self-Test Script]",
            "# Run ONLY the lines explicitly approved by the user for heavy/external simulators.",
            "set -euo pipefail",
            "",
        ]
        for cmd in jit.missing_pip_packages:
            lines.append(f"# {cmd}")
        for cmd in jit.missing_heavy_recipes:
            lines.append(f"# {cmd}")
        if not jit.missing_pip_packages and not jit.missing_heavy_recipes:
            lines.append("echo 'All selected Tier 0 built-in modules require zero external downloads.'")
        jit_script.write_text("\n".join(lines) + "\n", encoding="utf-8")
    written["jit_setup_sh"] = jit_script

    # Materialize verified external adapters whenever non-builtin ecosystem tools are configured
    adapters_dir = lab_dir / "adapters"
    for tid in spec.ecosystem_tools:
        if tid in JIT_RECIPES:
            bundle = write_jit_adapter_bundle(tid, dest_dir=adapters_dir, overwrite=overwrite)
            for key, path in bundle.items():
                written[f"adapter_{tid}_{key}"] = path

    return written
