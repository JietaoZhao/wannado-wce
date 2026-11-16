"""wannado.wce — Standards-anchored, Just-in-Time wireless communication environment quick provisioning.

Uses PEP 562 lazy loading so importing a single physics/traffic/6G block (e.g.
``from wannado.wce import provision_tr38901_uma_umi_rma``) does not eagerly load the entire catalog.
"""

from __future__ import annotations

from typing import Any

from wannado.wce.blocks import BLOCK_EXPORTS, lazy_resolve

__version__ = "1.1.0"

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    **BLOCK_EXPORTS,
    # Stage 0 standards-anchored requirement intake, 4-slot composition, live diff & JIT planning
    "ComposedSlotDecision": ("wannado.wce.intake", "ComposedSlotDecision"),
    "JITProvisioningPlan": ("wannado.wce.intake", "JITProvisioningPlan"),
    "SpecAuditReport": ("wannado.wce.intake", "SpecAuditReport"),
    "SpecDiffReport": ("wannado.wce.intake", "SpecDiffReport"),
    "WCESpec": ("wannado.wce.intake", "WCESpec"),
    "audit_wce_spec": ("wannado.wce.intake", "audit_wce_spec"),
    "build_jit_plan": ("wannado.wce.intake", "build_jit_plan"),
    "build_wce_archetype": ("wannado.wce.intake", "build_wce_archetype"),
    "compose_wce_spec": ("wannado.wce.intake", "compose_wce_spec"),
    "parse_param_updates_from_text": ("wannado.wce.intake", "parse_param_updates_from_text"),
    "update_wce_spec_with_diff": ("wannado.wce.intake", "update_wce_spec_with_diff"),
    # Stage 1 preflight & Stage 3 wireless physics rigor auditor
    "PreflightReport": ("wannado.wce.audit", "PreflightReport"),
    "RigorAuditReport": ("wannado.wce.audit", "RigorAuditReport"),
    "audit_wce_rigor": ("wannado.wce.audit", "audit_wce_rigor"),
    "preflight_wce": ("wannado.wce.audit", "preflight_wce"),
    # Stage 2 AI Co-Pilot inspection, arsenal, dialogs, zero-jargon guides & ETA
    "ProjectScan": ("wannado.wce.copilot", "ProjectScan"),
    "RuntimeEstimate": ("wannado.wce.copilot", "RuntimeEstimate"),
    "build_arsenal_showcase": ("wannado.wce.copilot", "build_arsenal_showcase"),
    "build_goal_confirmation_dialog": ("wannado.wce.copilot", "build_goal_confirmation_dialog"),
    "build_interactive_brief": ("wannado.wce.copilot", "build_interactive_brief"),
    "build_post_run_debrief": ("wannado.wce.copilot", "build_post_run_debrief"),
    "estimate_runtime": ("wannado.wce.copilot", "estimate_runtime"),
    "explain_parameters": ("wannado.wce.copilot", "explain_parameters"),
    "format_arsenal_overview": ("wannado.wce.copilot", "format_arsenal_overview"),
    "get_sota_lineage": ("wannado.wce.copilot", "get_sota_lineage"),
    "recommend_wce_blueprint": ("wannado.wce.copilot", "recommend_wce_blueprint"),
    "scan_project": ("wannado.wce.copilot", "scan_project"),
    # Non-intrusive ./wce_lab/ boundary, forward-coupled ComposedWirelessEnv & single-file ejection
    "ComposedSlotOutput": ("wannado.wce.workspace", "ComposedSlotOutput"),
    "ComposedWirelessEnv": ("wannado.wce.workspace", "ComposedWirelessEnv"),
    "WirelessEnvInterface": ("wannado.wce.workspace", "WirelessEnvInterface"),
    "eject_block": ("wannado.wce.workspace", "eject_block"),
    "scaffold_wce_lab": ("wannado.wce.workspace", "scaffold_wce_lab"),
    # Stage 4 traceable provenance reporter & zero-hallucination citations
    "BlockExperimentSummary": ("wannado.wce.reporter", "BlockExperimentSummary"),
    "WCEReport": ("wannado.wce.reporter", "WCEReport"),
    "build_block_config": ("wannado.wce.reporter", "build_block_config"),
    "evaluate_wce_suite": ("wannado.wce.reporter", "evaluate_wce_suite"),
    "export_citations": ("wannado.wce.reporter", "export_citations"),
    # Multi-IDE agent skill & instruction installer
    "install_agent_kit": ("wannado.wce.agent_kit", "install_agent_kit"),
    # Knowledge graph catalog helpers
    "build_registry_json": ("wannado.wce.catalog", "build_registry_json"),
    "get_taxonomy_tree": ("wannado.wce.catalog", "get_taxonomy_tree"),
    "load_catalog": ("wannado.wce.catalog", "load_catalog"),
    "query_catalog": ("wannado.wce.catalog", "query_catalog"),
    # Authoritative standards & open-source ecosystem + JIT glue helpers
    "DEEPMIMO_SCENARIO_CATALOG": ("wannado.wce.standards", "DEEPMIMO_SCENARIO_CATALOG"),
    "ECOSYSTEM_REGISTRY": ("wannado.wce.standards", "ECOSYSTEM_REGISTRY"),
    "IMT2030_USAGE_SCENARIOS": ("wannado.wce.standards", "IMT2030_USAGE_SCENARIOS"),
    "JITAdapterRecipe": ("wannado.wce.standards", "JITAdapterRecipe"),
    "JIT_RECIPES": ("wannado.wce.standards", "JIT_RECIPES"),
    "check_tool_platform_readiness": ("wannado.wce.standards", "check_tool_platform_readiness"),
    "format_ecosystem_table": ("wannado.wce.standards", "format_ecosystem_table"),
    "get_jit_recipe": ("wannado.wce.standards", "get_jit_recipe"),
    "match_imt2030_scenario": ("wannado.wce.standards", "match_imt2030_scenario"),
    "probe_local_ecosystem": ("wannado.wce.standards", "probe_local_ecosystem"),
    "recommend_ecosystem_tools": ("wannado.wce.standards", "recommend_ecosystem_tools"),
    "render_jit_adapter_code": ("wannado.wce.standards", "render_jit_adapter_code"),
    "verify_jit_adapter_file": ("wannado.wce.standards", "verify_jit_adapter_file"),
    "write_jit_adapter_bundle": ("wannado.wce.standards", "write_jit_adapter_bundle"),
    # Core primitives, adapters & provenance registries
    "ChannelModuleAdapter": ("wannado.wce.core", "ChannelModuleAdapter"),
    "LitCitation": ("wannado.wce.core", "LitCitation"),
    "MECHANISM_PROVENANCE": ("wannado.wce.core", "MECHANISM_PROVENANCE"),
    "ParamProvenance": ("wannado.wce.core", "ParamProvenance"),
    "ProvisionResult": ("wannado.wce.core", "ProvisionResult"),
    "STANDARDS_PROVENANCE": ("wannado.wce.core", "STANDARDS_PROVENANCE"),
    "StandardRef": ("wannado.wce.core", "StandardRef"),
    "WCEMeta": ("wannado.wce.core", "WCEMeta"),
    "WirelessGymEnvAdapter": ("wannado.wce.core", "WirelessGymEnvAdapter"),
    "adapt_external_channel_tensor": ("wannado.wce.core", "adapt_external_channel_tensor"),
    "get_standard_ref": ("wannado.wce.core", "get_standard_ref"),
    "resolve_block_provenance": ("wannado.wce.core", "resolve_block_provenance"),
    "verify_external_channel_adapter": ("wannado.wce.core", "verify_external_channel_adapter"),
}

__all__ = ["__version__", *_LAZY_EXPORTS.keys()]


def __getattr__(name: str) -> Any:
    return lazy_resolve(name, _LAZY_EXPORTS, globals(), __name__)


def __dir__() -> list[str]:
    return sorted(__all__)
