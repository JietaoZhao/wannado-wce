# Changelog

All notable changes to `wannado-wce` will be documented in this file.

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
