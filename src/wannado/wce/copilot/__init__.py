"""AI Co-Pilot inspection, arsenal showcase, zero-jargon explanations, JIT planning, and debrief."""

from wannado.wce.copilot.planner import (
    RuntimeEstimate,
    build_arsenal_showcase,
    build_goal_confirmation_dialog,
    build_interactive_brief,
    build_post_run_debrief,
    estimate_runtime,
    explain_parameters,
    format_arsenal_overview,
    get_sota_lineage,
    recommend_wce_blueprint,
)
from wannado.wce.copilot.project_inspector import ProjectScan, scan_project

__all__ = [
    "ProjectScan",
    "RuntimeEstimate",
    "build_arsenal_showcase",
    "build_goal_confirmation_dialog",
    "build_interactive_brief",
    "build_post_run_debrief",
    "estimate_runtime",
    "explain_parameters",
    "format_arsenal_overview",
    "get_sota_lineage",
    "recommend_wce_blueprint",
    "scan_project",
]
