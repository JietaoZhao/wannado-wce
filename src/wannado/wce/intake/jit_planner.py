"""Just-in-Time (JIT) Provisioning Planner: Decides Tier 0 Built-in vs Tier 1-3 External Pull Recipes."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from wannado.wce.intake.spec import WCESpec
from wannado.wce.standards.ecosystem import (
    ECOSYSTEM_REGISTRY,
    EcosystemTool,
    probe_local_ecosystem,
    recommend_ecosystem_tools,
)
from wannado.wce.standards.jit_recipes import (
    DEEPMIMO_SCENARIO_CATALOG,
    JIT_RECIPES,
    check_tool_platform_readiness,
)


@dataclass
class JITProvisioningPlan:
    """Actionable Just-in-Time environment assembly & optional package pull manifest."""

    builtin_blocks: list[str]
    baseline_blocks: list[str]
    external_tools: list[dict[str, Any]]
    already_installed: list[str]
    missing_pip_packages: list[str]
    missing_heavy_recipes: list[str]
    warnings: list[str] = field(default_factory=list)
    local_probe: dict[str, Any] = field(default_factory=dict)
    readiness_reports: list[dict[str, Any]] = field(default_factory=list)
    deepmimo_catalog: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def format_summary(self) -> str:
        lines = [
            "[wannado-wce Just-in-Time (JIT) Parts Manifest]",
            f"  Tier 0 Built-in Blocks : {', '.join(self.builtin_blocks)}",
            f"  Analytic Baselines     : {', '.join(self.baseline_blocks) or 'none'}",
            f"  Already Installed      : {', '.join(self.already_installed) or 'wannado_wce_builtin'}",
        ]
        if self.readiness_reports:
            lines.append("  OS / GPU / Disk Pre-Check:")
            for r in self.readiness_reports:
                lines.append(
                    f"    * {r['tool_id']}: ready={r['ready_to_pull']}, need~{r['estimated_required_gb']}GB / free={r['free_disk_gb']}GB "
                    f"(OS={r['os']}, CUDA={r['cuda_available']}, Docker={r['docker_available']})"
                )
        if self.missing_pip_packages:
            lines.append("  JIT PyPI Pull Commands (ask user before running):")
            lines.extend(f"    $ {cmd}" for cmd in self.missing_pip_packages)
        if self.missing_heavy_recipes:
            lines.append("  JIT Heavy/C++/Docker Blueprints (only if user explicitly needs full-stack/hardware):")
            lines.extend(f"    * {cmd}" for cmd in self.missing_heavy_recipes)
        if self.warnings:
            lines.append("  Platform Advisories:")
            lines.extend(f"    - {w}" for w in self.warnings)
        return "\n".join(lines)


def build_jit_plan(
    spec: WCESpec,
    *,
    needs_ray_tracing: bool = False,
    needs_full_protocol_stack: bool = False,
    needs_real_sdr_oran: bool = False,
    needs_rf_iq_ml: bool = False,
    needs_fec_coding: bool = False,
    needs_isac_waveform: bool = False,
    needs_spatial_gscm_traj: bool = False,
) -> JITProvisioningPlan:
    """Inspect local OS/GPU/packages and construct a minimal JIT procurement plan matching WCESpec."""
    probe = probe_local_ecosystem()
    tools: list[EcosystemTool] = recommend_ecosystem_tools(
        needs_ray_tracing=needs_ray_tracing,
        needs_full_protocol_stack=needs_full_protocol_stack,
        needs_real_sdr_oran=needs_real_sdr_oran,
        needs_rf_iq_ml=needs_rf_iq_ml,
        needs_fec_coding=needs_fec_coding,
        needs_isac_waveform=needs_isac_waveform,
        needs_spatial_gscm_traj=needs_spatial_gscm_traj,
    )
    for tid in spec.ecosystem_tools:
        if tid in ECOSYSTEM_REGISTRY and all(t.id != tid for t in tools):
            tools.append(ECOSYSTEM_REGISTRY[tid])

    installed: list[str] = []
    missing_pip: list[str] = []
    missing_heavy: list[str] = []
    warnings: list[str] = []

    py_inst = probe.get("installed_python_tools", {})
    os_name = probe.get("os", "")

    readiness_reports: list[dict[str, Any]] = []
    has_deepmimo = False

    for t in tools:
        if t.id == "wannado_wce_builtin":
            installed.append(t.id)
            continue
        if t.id == "deepmimo":
            has_deepmimo = True
        if t.id in JIT_RECIPES:
            dm_scen = "O1_28" if spec.carrier_fc_ghz >= 20.0 else "asu_campus_3p5"
            rd = check_tool_platform_readiness(t.id, deepmimo_scenario=dm_scen)
            readiness_reports.append(rd)
            for w in rd["warnings"]:
                if w not in warnings:
                    warnings.append(w)
            for b in rd["blockers"]:
                if b not in warnings:
                    warnings.append(f"BLOCKER: {b}")

        if py_inst.get(t.id):
            installed.append(t.id)
            continue
        if "Tier 1" in t.tier and t.install_command.startswith("pip install"):
            missing_pip.append(f"{t.install_command}  # {t.name} ({t.disk_footprint_hint})")
        else:
            missing_heavy.append(f"[{t.id}] {t.install_command} | Constraints: {t.os_gpu_constraints}")

        if t.id in ("openairinterface", "pyaerial") and os_name != "Linux":
            msg = f"{t.name} requires Linux (cannot run natively on {os_name}); use Docker or Tier 0/1 alternatives."
            if msg not in warnings:
                warnings.append(msg)

    return JITProvisioningPlan(
        builtin_blocks=list(spec.blocks_selected),
        baseline_blocks=list(spec.baseline_blocks),
        external_tools=[t.to_dict() for t in tools],
        already_installed=installed,
        missing_pip_packages=missing_pip,
        missing_heavy_recipes=missing_heavy,
        warnings=warnings,
        local_probe=probe,
        readiness_reports=readiness_reports,
        deepmimo_catalog=dict(DEEPMIMO_SCENARIO_CATALOG) if has_deepmimo else {},
    )
