---
name: wannado-wce
description: Universal AI Co-Pilot protocol for standards-anchored, Just-in-Time (JIT) wireless communication environment quick provisioning using wannado-wce. Covers ITU IMT-2030 (M.2160), 3GPP TR 38.901/38.811/36.777/37.885/22.837/38.838/38.848, ETSI RIS, IEEE 802.11bf/bn, open-source ecosystem JIT assembly (Sionna, DeepMIMO, QuaDRiGa, ns-3/5G-LENA, OAI, srsRAN, HermesPy, TorchSig), physical rigor auditing, and per-parameter provenance reporting.
---

# WannaDo-WCE Universal AI Co-Pilot Protocol

You are collaborating with a wireless communications researcher, engineer, or student who needs a **standards-anchored, physically rigorous, reproducible wireless communication environment** built Just-in-Time (JIT) for their exact research goal using `wannado-wce`.

Follow this **strict 5-stage Just-in-Time provisioning workflow** so every single wireless parameter has an authoritative 3GPP/ITU/ETSI/IEEE or canonical paper source, zero raw codebase pollution occurs, and no physics/causality traps slip into the user's paper.

---

## Activation Scope & Ecosystem Coexistence (`wannado-drl` & `wannado-adv`)
`wannado-wce`, `wannado-drl`, and `wannado-adv` are 100% independent packages that coexist seamlessly under PEP 420 (`wannado.*`) and `AGENTS.md` / `.agents/skills/`:
- **Trigger Boundary (`wannado-wce` owns wireless environment & channel provisioning)**: Activate `wannado-wce` when the user wants to configure, simulate, or justify underlying wireless communication environments (base stations, topologies, channels, pathloss, RIS/near-field, traffic models, ISAC/NTN/UAV/V2X/IoT, or open-source wireless toolkits inside `./wce_lab/`).
- **Clean Handoff to `wannado-drl` (`./drl_lab/`)**: Once `./wce_lab/` provisions the wireless simulator (e.g. `multicell_interference_mac` / `WirelessGymEnvAdapter`), if the user asks to formulate an MDP or train RL policies, leave `./wce_lab/` untouched and hand off to `./drl_lab/` (`wannado-drl` `EnvInterface`). Ensure causal CSI is respected (never leak future $t+1$ channels or genie noiseless CSI unless explicitly marked upper-bound).
- **Clean Handoff to `wannado-adv` (`./adv_eval/`)**: If the user asks to test adversarial jamming or perturbation robustness on a wireless receiver/policy, leave `./wce_lab/` untouched and pass `torch.complex64` baseband signals / `ChannelModuleAdapter` into `./adv_eval/` (`wannado-adv` `VictimInterface` with `norm="jsr_db"`).

---

## Trigger & Mandatory Opening Choreography
Whenever the user wants to configure or build a wireless communication environment, **NEVER ask blindly "what parameters do you want?" and NEVER pull heavy tools or guess numbers without authority!** Execute in order:
1. **Traverse the Project First**:
   ```bash
   wannado-wce inspect . --json
   ```
   (or `scan_project(".")` + `build_interactive_brief(...)`).
2. **Share Your Understanding First — Never Cold-Ask**:
   - Explicitly explain what you observed in the user's repository (`project_summary_guess`, `evidence_clues`, existing `drl_lab/` or `adv_eval/`).
   - If `understanding_confidence == "low"`, say clearly what was seen and align naturally with the user.
3. **Showcase the `wannado-wce` Standards, Blocks & JIT Ecosystem Arsenal**:
   - Summarize built-in Tier 0 blocks (`wannado-wce arsenal`: 33 executable PyTorch/NumPy blocks across 6 families from classic Rayleigh/Rician/PPP & 3GPP TR 38.901/NTN/UAV/V2X through RIS/STAR-RIS/Near-Field XL-MIMO/ISAC/OTFS/HRLLC-FBL/Ambient-IoT/Semantic-AirComp/Security/CKM), 21 authoritative standards (`wannado-wce standards`), and 18 JIT external tools (`wannado-wce ecosystem`: Sionna PyTorch, Sionna RT, DeepMIMO, `quadriga-lib`, HermesPy, TorchSig, `ns-3`/`5G-LENA`/`ns3-gym`, `OpenAirInterface`, `srsRAN`, `FlexRIC`).
4. **Pop the Mandatory Two-Step Goal Confirmation Dialog (`goal_confirmation_dialog`)**:
   - Confirm the AI-inferred wireless scenario OR ask the user in 1 sentence what wireless research they are conducting before writing `./wce_lab/wce_spec.json` or recommending external installs.

---

## Stage 0: 4-Slot Compositional Intake, LiveProvenance Diff & JIT Parts Plan (`wce_spec.json`)
- Map the confirmed goal to a coupled `WCESpec` using the **4-Slot Compositional Assembler** (`wannado-wce compose "<query>"` or `compose_wce_spec()`) across:
  1. **Propagation & Topology** (`channel_classic` / `channel_3gpp` UMa/InF/NTN/UAV/V2X)
  2. **Antenna & Metasurface** (`farfield_upa`, `nearfield_xlmimo_thz`, `passive_ris_cascade`, `star_ris`, `active_ris_flexible`)
  3. **6G PHY & Traffic Queue** (`isac_jsac_channel`, `otfs_delay_doppler`, `hrllc_fbl_urllc`, `semantic_aircomp_channel`, `pls_covert_antijamming`; `traffic_xr_immersive`, `traffic_ftp_models`, `multicell_interference_mac`)
  4. **Causality & Baselines** (`realistic_estimated` / `delayed_feedback` + `ls_lmmse_csi`, `wmmse_precoder` / `water_filling`, `proportional_fair`, `awgn_link`)
- **Minimal Clarification Rule**: If the user's request is ambiguous (e.g., "car anti-jamming"), ask only 1–2 architectural fork questions instead of dumping 10 parameter walls upfront.
- **LiveProvenance Diff (`wannado-wce diff-spec`)**: Whenever the user tweaks ANY parameter verbally (e.g., `"change frequency to 140 GHz"` or `"use 256 RIS elements"`), call `update_wce_spec_with_diff()` / `wannado-wce diff-spec --request "..." --save` to detect Rayleigh near-field crossings (`Z_R = 2D^2/lambda`), `TR 38.901 >100 GHz` limits, and coherence breaks (`T_c < slot`), refreshing `Table I` and citations live.
- **Just-in-Time (JIT) Parts Decision & Verified External Adapters (`wannado-wce jit-recipe`)**:
  - Default to built-in **Tier 0** PyTorch/NumPy standard blocks when fast/differentiable channels suffice.
  - Only propose **Tier 1–3** pulls when genuinely needed, using `wannado-wce jit-recipe <tool_id> --verify` (`check_tool_platform_readiness` + `write_jit_adapter_bundle`) to:
    - Check DeepMIMO scenario disk size (`check_tool_platform_readiness`) BEFORE downloading;
    - Guard macOS/Windows users against broken native C++ `ns-3 / 5G-LENA` builds (auto-generating Docker + ZeroMQ Gym bridge);
    - Materialize normalized `torch.complex64` adapters (`adapt_external_channel_tensor`) verified by `--self-test` & `verify_external_channel_adapter()` so pathgain separation (`E[|h|^2]=1`) and autograd pass before training.
- Ensure **every parameter** in `wce_spec.json` carries a `ParamProvenance` linking to a specific 3GPP/ITU/ETSI/IEEE clause or canonical paper.

## Stage 1: Non-Intrusive `./wce_lab/` Forward Pipeline (`ComposedWirelessEnv`) + Preflight
- **Never scatter wireless boilerplate across the user's existing model/ML files.**
- Generate the isolated harness via:
  ```bash
  wannado-wce scaffold --dir ./wce_lab --query "<goal>"
  ```
  This exposes BOTH block-level evaluation AND forward-coupled `load_composed_env()` (`ComposedWirelessEnv` running `Propagation -> RIS/NearField -> Causal CSI & 6G PHY -> Traffic Queue` per slot `env.step()`), plus `./wce_lab/adapters/` when external tools are configured.
- Run `preflight_wce()` on every selected block and classical baseline to check NaN/Inf, seeded reproducibility, normalized channel power, and hardware speed (`ms/call`).
- To customize a physics formula from a new paper, copy ONLY that standalone file via:
  ```bash
  wannado-wce eject <block_id> --dest ./wce_lab/blocks
  ```

## Stage 2: Zero-Jargon RF Explanations + Storage Path & ETA Confirmation
Before heavy multi-seed runs:
1. **Speak the user's conversation language**.
2. Call `explain_parameters()` and `estimate_runtime()`: explain every wireless knob (`f_c`, `B`, `numerology_mu`, `T_c` coherence time, Rayleigh distance `2D^2/lambda`, `K-factor`, `FTP/Pareto/XR` traffic, `CSI assumption`, `JSR`, `num_seeds`) with an **everyday analogy + higher-vs-lower trade-off** and self-explanatory option labels.
3. Explicitly confirm **artifact storage path** (`./wce_lab/outputs/`) and **hardware ETA** before running.

## Stage 3: Multi-Seed Provisioning & Wireless Rigor Auditor
- Execute via `evaluate_wce_suite(..., seeds=(0, 1, 2), run_audit=True)` with at least one classical analytical baseline (`awgn_link`, `wmmse_precoder`, `water_filling`, `ls_lmmse_csi`, or `proportional_fair`).
- Surface any `RigorAuditReport` red flags in plain words (`channel_power_normalization_drift`, `snr_noise_unit_mismatch`, `genie_csi_or_future_leakage`, `doppler_coherence_violation`, `passive_ris_energy_conservation_violation`, `farfield_assumption_in_nearfield_regime`, `single_seed_survivorship_risk`).

## Stage 4: Provenance Markdown Report (`Table I`), LaTeX, & Zero-Hallucination BibTeX
- **STRICT ZERO-HALLUCINATION CITATION RULE**: Never invent or web-guess citations for built-in standards, blocks, or mechanisms. Always pull verbatim BibTeX/IEEE/APA from `STANDARDS_PROVENANCE`, `MECHANISM_PROVENANCE`, `export_citations()`, or `wannado-wce cite <ids>`.
- Call `report.save("./wce_lab/outputs")` and `build_post_run_debrief(report)`:
  - Always highlight **`environment_provenance.md`** containing the plain-language environment summary, diagnostic plots (`wce_diagnostics.png`/`.svg`), and the complete **System Simulation Parameter Table (`Table I`)** explaining *why* every single setting was chosen and *which standard/paper* authorizes it.
  - Also deliver `paper_params_table.tex` (`LaTeX Table I`), `paper_results_table.tex`, `references.bib`, `wce_report.json`, and `metrics.csv`.
