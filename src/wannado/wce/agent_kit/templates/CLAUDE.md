# wannado-wce Claude Code Collaboration Rules

Read `.claude/skills/wannado-wce/SKILL.md` (and `AGENTS.md`) before provisioning any wireless communication environment.

- **Scope & Handoff**: Use `wannado-wce` (`./wce_lab/`) only for wireless physical/MAC environment provisioning, standards provenance, JIT tool selection, and physical rigor checks. Hand off cleanly to `./drl_lab/` (`wannado-drl`) for RL training or `./adv_eval/` (`wannado-adv`) for JSR/adversarial testing without modifying `./wce_lab/`.

1. Run `wannado-wce inspect .` first: explain observed clues + `wannado-wce arsenal`, then pop the 2-step **Goal Confirmation Dialog** before Stage 0 `wce_spec.json` and JIT plan alignment.
2. Anchor every physical knob to 3GPP/ITU/ETSI/IEEE standards (`ParamProvenance`), isolate wiring inside `./wce_lab/` (`wannado-wce scaffold`), eject single files via `wannado-wce eject <block_id>`, and always confirm OS/GPU/disk impact before JIT installing heavy external simulators.
3. Run `preflight_wce()`, explain RF terms via `explain_parameters()` in plain analogies in the user's language, and confirm `./wce_lab/outputs/` + `estimate_runtime()` ETA before multi-seed sweeps.
4. Run `evaluate_wce_suite(..., run_audit=True)` + `report.save()`, catch any `RigorAuditReport` traps, and highlight `environment_provenance.md` (with System Parameter Table I & charts), `paper_params_table.tex`, and verbatim `references.bib` (`wannado-wce cite`).
