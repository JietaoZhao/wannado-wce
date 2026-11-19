<!-- wannado-wce:start -->
# wannado-wce Claude Code Collaboration Rules

Read `.claude/skills/wannado-wce/SKILL.md` (and `AGENTS.md`) before provisioning any wireless communication environment.

- **Scope & Handoff**: Use `wannado-wce` (`./wce_lab/`) only for wireless physical/MAC environment provisioning, standards provenance, JIT tool selection, and physical rigor checks. Hand off cleanly to `./drl_lab/` (`wannado-drl`) for RL training or `./adv_eval/` (`wannado-adv`) for JSR/adversarial testing without modifying `./wce_lab/`.

1. Run `wannado-wce inspect .` if entering an existing codebase; if underspecified, ask 1–2 targeted questions, otherwise compose directly (`wannado-wce compose "<goal>"`).
2. Isolate in `./wce_lab/` (`wannado-wce scaffold --query "..."`), handle edits with `wannado-wce diff-spec`, and verify JIT external glue (`wannado-wce jit-recipe <tool_id> --verify`, or `--require-live` when live simulator deps are present).
3. Validate physics, link-level coded PHY (`evaluate_link_level_phy()` when needed), and all 10 `RigorAuditReport` checks via `wannado-wce verify --dir ./wce_lab --seeds 0 1 2` before heavy training.
4. Run `evaluate_wce_suite(..., run_audit=True)` + `report.save("./wce_lab/outputs")` and highlight `environment_provenance.md` (Table I + plots), `paper_params_table.tex`, and verbatim `references.bib` (`wannado-wce cite`).
<!-- wannado-wce:end -->
