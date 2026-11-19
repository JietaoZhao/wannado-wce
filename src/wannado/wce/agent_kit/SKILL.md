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

## Coding-Agent Execution Workflow (Fast Path + Rigor Gates)
Whenever the user wants to build or verify a wireless communication environment, **do not dump generic parameter questionnaires** and **never install heavy external packages or guess channel parameters without standard provenance**:
1. **Inspect Context First**:
   ```bash
   wannado-wce inspect . --json
   ```
   (or `scan_project(".")`) to detect existing `./wce_lab/`, `./drl_lab/`, `./adv_eval/`, or Python files. If the user's request is already specific (e.g., carrier frequency, topology, antenna/RIS count, or standard), proceed directly; **ask only 1–2 concise architectural clarifying questions if genuinely ambiguous**.
2. **Inspect Built-in Arsenal & Standards**:
   - Use built-in **Tier 0** blocks first (`wannado-wce arsenal`: 33 executable PyTorch/NumPy blocks across 6 families, 21 standards via `wannado-wce standards`, and 18 JIT external tool recipes via `wannado-wce ecosystem`).

---

## Stage 0: 4-Slot Compositional Intake, LiveProvenance Diff & JIT Parts Plan (`wce_spec.json`)
- Map the goal to a coupled `WCESpec` using the **4-Slot Compositional Assembler** (`wannado-wce compose "<query>"` or `compose_wce_spec()`) across:
  1. **Propagation & Topology** (`channel_classic` / `channel_3gpp` UMa/UMi/InF/NTN/UAV/V2X)
  2. **Antenna & Metasurface** (`farfield_upa`, `nearfield_xlmimo_thz`, `passive_ris_cascade`, `star_ris`, `active_ris_flexible`)
  3. **6G PHY & Traffic Queue** (`isac_jsac_channel`, `otfs_delay_doppler`, `hrllc_fbl_urllc`, `semantic_aircomp_channel`, `pls_covert_antijamming`; `traffic_xr_immersive`, `traffic_ftp_models`, `multicell_interference_mac`)
  4. **Causality & Baselines** (`realistic_estimated` / `delayed_feedback` + `ls_lmmse_csi`, `wmmse_precoder` / `water_filling`, `proportional_fair`, `awgn_link`)
- **LiveProvenance Diff (`wannado-wce diff-spec`)**: Whenever the user updates any parameter (e.g., `"change frequency to 140 GHz"` or `"use 256 RIS elements"`), call `update_wce_spec_with_diff()` / `wannado-wce diff-spec --request "..." --save` to flag Rayleigh near-field crossings (`Z_R = 2D^2/lambda`), `TR 38.901 >100 GHz` bounds, and coherence breaks (`T_c < slot`), refreshing `Table I` and citations live.
- **Just-in-Time (JIT) External Simulator Adapters (`wannado-wce jit-recipe`)**:
  - Default to built-in **Tier 0** PyTorch/NumPy standard blocks when fast/differentiable channels suffice.
  - When external simulators (`sionna_rt`, `deepmimo`, `quadriga_lib`, `ns3_5glena_ns3gym`, `hermespy`, `torchsig`) are needed, run `wannado-wce jit-recipe <tool_id> --verify` (`check_tool_platform_readiness` + `write_jit_adapter_bundle` + `verify_jit_adapter_file`). Verification explicitly distinguishes `"synthetic_contract_smoke"` (shape/power/autograd contract check when uninstalled) from `"live_runtime_probe"` (pass `--require-live` when the simulator runtime is required).

## Stage 1: Isolated `./wce_lab/`, Forward `ComposedWirelessEnv`, & Link-Level PHY
- **Never scatter wireless boilerplate across the user's existing ML or app files.**
- Generate the isolated harness via:
  ```bash
  wannado-wce scaffold --dir ./wce_lab --query "<goal>"
  ```
  Exposes both block-level evaluation and `load_composed_env()` (`ComposedWirelessEnv` running `Propagation -> RIS/NearField -> Causal CSI & 6G PHY -> Traffic Queue` per `env.step()`, plus `.as_gym_env(multi_user=...)` and `.as_channel_module(stateless=True)`).
- Run `preflight_wce()` before long sweeps, and use `evaluate_link_level_phy()` when end-to-end coded OFDM/MIMO LDPC BLER, EVM, or pilot-based channel estimation (`LS` / `LMMSE` / `Genie`) is required.
- To customize a physics formula, eject ONLY that single file via `wannado-wce eject <block_id> --dest ./wce_lab/blocks`.

## Stage 2: Standards-Anchored RF Clarifications & Runtime Budgeting
- Call `explain_parameters()` and `estimate_runtime()` when explaining wireless choices (`f_c`, `B`, `mu`, `T_c`, Rayleigh `2D^2/lambda`, `K-factor`, traffic models, `CSI assumption`, `JSR`, `num_seeds`) or estimating wall-clock budget for `./wce_lab/outputs/`.

## Stage 3: Rigor Verification Gate (`wannado-wce verify`) & 10-Check Auditor
- Run the automated physical/statistical gate on any query, spec, or `./wce_lab/` workspace:
  ```bash
  wannado-wce verify --dir ./wce_lab --seeds 0 1 2 --json
  ```
- Execute multi-seed suites via `evaluate_wce_suite(..., seeds=(0, 1, 2), run_audit=True)` with at least one analytical baseline (`awgn_link`, `wmmse_precoder`, `water_filling`, `ls_lmmse_csi`, or `proportional_fair`), checking all 10 `RigorAuditReport` guards (`channel_power_normalization_drift`, `snr_noise_unit_mismatch`, `genie_csi_or_future_leakage`, `doppler_coherence_violation`, `passive_ris_energy_conservation_violation`, `farfield_assumption_in_nearfield_regime`, `single_seed_survivorship_risk`, `zero_variance_deterministic_metric`, `ris_quantization_and_aperture_ok`, `table1_provenance_integrity_ok`).

## Stage 4: Provenance Markdown Report (`Table I`), LaTeX, & Zero-Hallucination BibTeX
- **STRICT ZERO-HALLUCINATION CITATION RULE**: Never invent or web-guess citations for built-in standards, blocks, or mechanisms. Always pull verbatim BibTeX/IEEE/APA from `STANDARDS_PROVENANCE`, `MECHANISM_PROVENANCE`, `export_citations()`, or `wannado-wce cite <ids>`.
- Call `report.save("./wce_lab/outputs")` and highlight:
  - **`environment_provenance.md`** (plain-language environment summary, `wce_diagnostics.png`/`.svg`, and the complete **System Simulation Parameter Table (`Table I`)** linking every parameter to its standard clause or paper),
  - **`paper_params_table.tex`**, **`paper_results_table.tex`**, **`references.bib`**, **`wce_report.json`**, and **`metrics.csv`**.
