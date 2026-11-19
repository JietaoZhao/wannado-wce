<!-- wannado-wce:start -->
# wannado-wce Universal AI Collaboration Rules

This repository uses `wannado-wce` under the open `AGENTS.md` + `.agents/skills/wannado-wce/SKILL.md` standard (compatible across Cursor, Antigravity, Codex, Claude Code, Windsurf, and other AI coding agents).

- **Scope & Sibling Coexistence (`wannado-drl` & `wannado-adv`)**: Activate `wannado-wce` ONLY for wireless communication environment provisioning, standards/channel/traffic configuration (`./wce_lab/`), Just-in-Time (JIT) open-source wireless library assembly, and physical consistency auditing. If `wannado-drl` (`./drl_lab/`) is used next for scheduling/beamforming RL or `wannado-adv` (`./adv_eval/`) is used next for RF/adversarial robustness testing, leave `./wce_lab/` untouched and bridge cleanly via exported Gym or `torch.complex64` interfaces (never leak genie/future CSI into causal RL states).

Follow this 5-stage Just-in-Time provisioning workflow directly (ask 1–2 clarifying questions only when underspecified):

1. **Inspect & Compose (`inspect` -> `compose`)**: Run `wannado-wce inspect .` if entering an existing repo, then map the goal with `wannado-wce compose "<goal>"` (4 slots: Propagation / Antenna·RIS / 6G·Traffic / Causality·Baselines) and handle parameter edits via `wannado-wce diff-spec` so Rayleigh `Z_R = 2D^2/lambda` or `TR 38.901 >100 GHz` bounds refresh `Table I` live.
2. **Verify JIT External Adapters (`jit-recipe --verify`)**: Default to built-in Tier 0 PyTorch/NumPy blocks (`wannado-wce arsenal`). When `sionna_rt`, `deepmimo`, `ns3_5glena_ns3gym`, `quadriga_lib`, `torchsig`, or `hermespy` is requested, run `wannado-wce jit-recipe <tool_id> --verify` (distinguishes `synthetic_contract_smoke` vs `--require-live` runtime probe).
3. **Isolate in `./wce_lab/` (`scaffold` / `eject` / `link_level_phy`)**: Keep `./wce_lab/` separate from application code via `wannado-wce scaffold --query "..."`, exposing `ComposedWirelessEnv` (`load_composed_env()`, `.as_gym_env()`, `.as_channel_module()`), optional `evaluate_link_level_phy()` (LDPC + OFDM-MIMO BLER/EVM), and single-file customization via `wannado-wce eject <block_id> --dest ./wce_lab/blocks`.
4. **Enforce Physical Rigor Gate (`wannado-wce verify`)**: Run `wannado-wce verify --dir ./wce_lab --seeds 0 1 2` and `evaluate_wce_suite(..., run_audit=True)` with at least one classic analytical baseline to verify all 10 `RigorAuditReport` physics, causality, quantization, and provenance checks.
5. **Deliver Provenance & Zero-Hallucination BibTeX**: Call `report.save("./wce_lab/outputs")` and surface `environment_provenance.md` (deduplicated System Simulation Parameter Table I + charts), `paper_params_table.tex`, `paper_results_table.tex`, and verbatim `references.bib` (`wannado-wce cite` only).
<!-- wannado-wce:end -->
