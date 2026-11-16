<!-- wannado-wce:start -->
# wannado-wce Claude Code Collaboration Rules

Read `.claude/skills/wannado-wce/SKILL.md` (and `AGENTS.md`) before provisioning any wireless communication environment.

- **Scope & Handoff**: Use `wannado-wce` (`./wce_lab/`) only for wireless physical/MAC environment provisioning, standards provenance, JIT tool selection, and physical rigor checks. Hand off cleanly to `./drl_lab/` (`wannado-drl`) for RL training or `./adv_eval/` (`wannado-adv`) for JSR/adversarial testing without modifying `./wce_lab/`.

1. Run `wannado-wce inspect .` first: explain observed clues + `wannado-wce arsenal`, then pop the 2-step **Goal Confirmation Dialog** before Stage 0 `wce_spec.json` and JIT plan alignment.
2. Use 4-slot composition (`wannado-wce compose` / `wannado-wce scaffold --query "..."`) + `wannado-wce diff-spec` for live parameter changes & boundary alerts (`Z_R`, `>100 GHz`, `T_c`). For external tools (`sionna_rt`, `deepmimo`, `ns3_5glena_ns3gym`, `torchsig`), run `wannado-wce jit-recipe <tool_id> --verify` first to check OS/Docker/disk limits and validate normalized complex64 glue adapters.
3. Run `preflight_wce()` + `ComposedWirelessEnv`, explain RF terms via `explain_parameters()` in plain analogies in the user's language, and confirm `./wce_lab/outputs/` + `estimate_runtime()` ETA before multi-seed sweeps.
4. Run `evaluate_wce_suite(..., run_audit=True)` + `report.save()`, catch any `RigorAuditReport` traps, and highlight `environment_provenance.md` (with System Parameter Table I & charts), `paper_params_table.tex`, and verbatim `references.bib` (`wannado-wce cite`).
<!-- wannado-wce:end -->
