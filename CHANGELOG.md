# Changelog

All notable changes to `wannado-wce` will be documented in this file.

## [1.1.3] - 2026-11-17

### Changed
- **English-first demos, UI dialogs & standard naming**: Translated all README CLI demos, CLI `--request` help example, `docs/index.html` feedback portal, Co-Pilot clarifying fork dialogs & labels (`copilot/planner.py`), ITU-R M.2160 scenario names (`standards/itu_imt2030.py`), docstrings, and test fixtures to clean English; preserved silent multilingual NL input keyword matching inside `composer.py` / `planner.py` as an optional parser fallback while ensuring all outputs default cleanly to English.

## [1.1.2] - 2026-11-17

### Added
- **Non-telecom cross-domain bridge primitives (`workspace/cross_domain.py`, `standards/etsi_ieee.py`)**:
  - Added `itur_p618_rain_attenuation_db` (ITU-R P.838-3 + P.618/P.530 weather fade attenuation) with integrated `rain_rate_mm_hr` support inside `ComposedWirelessEnv`.
  - Added `FederatedWirelessRoundModel` & `simulate_federated_round` for Federated Learning / distributed AI synchronous round straggler & uplink energy modeling.
  - Added `AutonomousLinkDelayModel` & `evaluate_link_delivery` for robotics, UAV GIS, and autonomous vehicle V2V 3D position-to-latency/BLER closed-loop checks.
  - Added presets `uav_gis`, `federated_edge`, and `v2v_control` to `build_wce_archetype()` + keyword routing in `compose_wce_spec()`.
- **Live runtime controls in `ComposedWirelessEnv.step()`**: Added causal action execution (`action=0/1/2` or continuous scalars affecting TX power boost, RIS alignment multiplier, and power cost penalty in `as_gym_env()`) and runtime overrides (`distances_m`, `tx_pos_3d`, `rx_pos_3d`, `positions_3d`, `packet_bits`, `ris_scale`, `power_scale`, `rain_rate_mm_hr`).

### Fixed
- **Composer substring misrouting (`intake/composer.py`)**: Replaced bare `"ar"` substring check with word-boundary regex `r"\b(?:xr|vr|ar)\b"` so `"federated learning"`, `"Starlink"`, `"radar"`, and `"lidar"` never misroute to `traffic_xr_immersive`; added `"starlink"` and `"lora"`/`"lorawan"` frequency/bandwidth recognition.
- **Stage 0 & reporter validation**: Tightened `audit_wce_spec()` checks for `min_tx_rx_distance_m <= 0`, invalid `num_tx`/`num_rx`/`numerology_mu`, negative `noise_figure_db`, and extreme terrestrial `tx_power_dbm`; rejected empty `seeds` in `evaluate_wce_suite()`.
- **LaTeX double-unit & escaping fixes (`intake/spec.py`, `reporter/env_reporter.py`)**: Fixed `"100.00 MHz Hz"` double-unit outputs in Table I (`archetypes.py`, `composer.py`, `spec.py`), escaped `&` in `"Benchmark \\& Rigor Audit"` in `to_results_latex()`, and escaped `#`, `$`, `^`, and `~`.

## [1.1.1] - 2026-11-17

### Fixed
- **OTFS delay-Doppler multipath energy conservation (`otfs_delay_doppler`)**: Replaced PyTorch indexed `+=` (which drops colliding indices) with `index_put_(..., accumulate=True)` so colliding multipath taps conserve expected delay-Doppler energy ($\mathbb{E}[E] \approx 1.0$).
- **3GPP TR 38.901 mixed LoS/NLoS shadow fading (`compute_pathloss_tr38901_db`)**: Returned element-wise `sf_std` array for mixed LoS/NLoS link vectors across UMa, UMi, RMa, InH, and InF instead of broadcasting NLoS $\sigma_{\text{SF}}$ to all LoS links, and removed redundant UMa conditional branch.
- **3GPP TR 37.885 V2X highway NLOS (`compute_v2x_tr37885_pathloss`)**: Added distinct highway `NLOS` dual-slope/heavy-pathloss branch rather than silently falling through to LOS.
- **ISAC range CRB consistency (`isac_jsac_channel` & `ComposedWirelessEnv`)**: Unified both modules to use $B_{\text{rms}} = B/\sqrt{12}$ and the two-way round-trip sensitivity factor ($1/2$ for monostatic $\tau = 2R/c$, $1$ for bistatic).
- **ComposedWirelessEnv physical consistency & RIS composer keyword matching**: Used full TR 38.901/TR 37.885 shadow fading std (`1.0 * sf_db`), tied passive/STAR-RIS element scaling dynamically to `spec.num_tx`, added dynamic amplified RIS thermal noise $\sigma_v^2$ to `active_ris_flexible`, prioritized `active_ris_flexible` over `passive_ris_cascade` in `ComposedWirelessEnv`, and prevented substring collisions (`"pass"` in `"passive"`, `"fas"` in `"fast"`) in `compose_wce_spec()`.
- **Suite evaluation & rigor audit (`evaluate_wce_suite` & `audit_wce_rigor`)**: Actually execute deduplicated `baseline_blocks` alongside primary blocks, default `min_seeds=3` to preserve single-seed survivorship checks, forward full `WCESpec` parameters (`num_tx`, `num_rx`, `min_tx_rx_distance_m`, `max_speed_mps`, `numerology_mu`, `slot_duration_ms`) in `build_block_config()`, and avoid selecting deterministic constants as primary audit metrics.
- **Provenance accuracy & cross-platform BLAS cleanliness**: Updated ITU-R P.676 provenance description to simplified spectroscopic surrogate (`2 O2 + 7 H2O` lines), made `awgn_link` transmit-power rationale reflect configured `tx_power_dbm`, synced `problem_name` band tokens during `update_wce_spec_with_diff`, eliminated Accelerate BLAS subnormal `RuntimeWarning` in `ckm_radiomap_bridge`, replaced dynamic `__all__` with static lists, and resolved all `ruff` and `mypy` issues.

### Added
- **`evaluate_wce_suite(spec)` shorthand**: Accepts a `WCESpec` instance directly as its first argument.
- **`quadriga_lib` platform readiness notice**: Warns on macOS / Python 3.13+ where C++ source compilation may be required.
- **Engineering governance**: Added PEP 561 `py.typed` marker, `CITATION.cff`, `CHANGELOG.md`, and GitHub Actions CI matrix across Python 3.10–3.13.

## [1.1.0] - 2026-11-16

### Added
- Initial 1.1.0 release featuring 33 CleanRL-style wireless blocks across 6 families, 21 anchored standards, 18 JIT external tool blueprints, 4-slot compositional assembler, `ComposedWirelessEnv`, `LiveProvenance Diff`, Stage 3 rigor auditor, and Stage 4 LaTeX/BibTeX reporter.
