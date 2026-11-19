"""Stage 3: Wireless Physical Consistency & Statistical Rigor Auditor.

Automatically catches 7 fatal wireless communication experimental traps:
1. Channel power normalization drift (unnormalized H inflating SNR artificially).
2. SNR / thermal noise unit mismatch (dB vs linear confusion or non-finite values).
3. Genie / future CSI leakage into causal estimators or schedulers.
4. Doppler / coherence-time violation (lag-1 temporal correlation contradicting speed & slot length).
5. Passive RIS energy conservation violation (|phi_m| > 1 without active amplifier noise model).
6. Far-field planar-wave assumption inside the near-field Rayleigh distance Z_R = 2D^2 / lambda.
7. Single-seed survivorship risk or omission of classical analytical baselines.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from wannado.wce.core.types import ProvisionResult
from wannado.wce.core.utils import aggregate_seed_stats
from wannado.wce.intake.spec import WCESpec


@dataclass
class RigorAuditReport:
    """Structured Stage 3 wireless physical & statistical rigor audit report."""

    passed: bool
    red_flags: list[str] = field(default_factory=list)
    checks: dict[str, bool] = field(default_factory=dict)
    primary_metric_name: str = ""
    seed_stats: dict[str, float] = field(default_factory=dict)
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "red_flags": list(self.red_flags),
            "checks": dict(self.checks),
            "primary_metric_name": self.primary_metric_name,
            "seed_stats": dict(self.seed_stats),
            "recommendations": list(self.recommendations),
        }

    def format_report(self) -> str:
        verdict = "PASS (Physically Consistent & Multi-Seed Verified)" if self.passed else "WARNING (Physical / Statistical Trap Detected)"
        lines = [f"[wannado.wce Rigor Audit: {verdict}]"]
        if self.primary_metric_name:
            lines.append(
                f"  {self.primary_metric_name}: mean={self.seed_stats.get('mean', 0.0):.3f} ± {self.seed_stats.get('std', 0.0):.3f}, "
                f"IQM={self.seed_stats.get('iqm', 0.0):.3f}"
            )
        if self.red_flags:
            lines.append("  Red Flags:")
            lines.extend(f"    - {f}" for f in self.red_flags)
        lines.append("  Checks:")
        for k, v in self.checks.items():
            lines.append(f"    - {k}: {'OK' if v else 'FAIL'}")
        if self.recommendations:
            lines.append("  Recommendations:")
            lines.extend(f"    - {r}" for r in self.recommendations)
        return "\n".join(lines)


def audit_wce_rigor(
    seed_results: Sequence[ProvisionResult],
    *,
    wce_spec: WCESpec | None = None,
    has_classic_baseline: bool = True,
    min_seeds: int = 3,
    primary_metric: str | None = None,
) -> RigorAuditReport:
    """Audit multi-seed ProvisionResults against physical conservation laws, causality, and statistical rigor."""
    if not seed_results:
        return RigorAuditReport(
            passed=False,
            red_flags=["no_runs_provided"],
            checks={"has_runs": False},
            recommendations=["Provision at least 3 random seeds before running Rigor Audit."],
        )

    red_flags: list[str] = []
    recommendations: list[str] = []
    checks: dict[str, bool] = {}

    # Select primary scalar metric for multi-seed IQM summary
    metric_name = primary_metric or ""
    if not metric_name:
        for cand in (
            "mean_spectral_efficiency_bps_hz",
            "shannon_capacity_mbps",
            "wmmse_sum_rate_bps_hz",
            "water_filling_rate_bps_hz",
            "pf_sum_rate_bps_hz",
            "pf_mean_throughput_mbps",
            "fbl_achievable_rate_bps_hz",
            "comm_spectral_efficiency_bps_hz",
            "ckm_rmse_improvement_db",
            "mean_secrecy_rate_bps_hz",
            "active_ris_effective_snr",
            "gain_aligned_ris",
            "mean_dd_energy",
            "mean_farfield_match_corr",
            "farfield_beam_alignment_efficiency",
            "lmmse_nmse_db",
            "outage_probability",
            "device_activation_rate",
            "mean_packet_satisfaction",
            "offered_traffic_mbps",
            "offered_traffic_kbps",
            "coverage_probability",
            "mean_frobenius_sq",
            "csi_temporal_variance",
            "empirical_amount_of_fading",
            "los_power_fraction",
            "autocorr_abs_error",
            "nmse_db",
            "range_rmse_crb_m",
            "sensing_crb_range_m",
            "median_snr_db",
            "p5_edge_snr_db",
            "mean_channel_power",
        ):
            if cand in seed_results[0].metrics:
                metric_name = cand
                break
        if not metric_name and seed_results[0].metrics:
            static_constants = {
                "aperture_m",
                "max_doppler_hz",
                "rayleigh_distance_m",
                "is_in_nearfield",
                "autograd_gradient_valid",
                "num_tx",
                "num_rx",
            }
            stochastic_keys = [k for k in seed_results[0].metrics if k not in static_constants]
            metric_name = stochastic_keys[0] if stochastic_keys else next(iter(seed_results[0].metrics.keys()))

    vals = [float(r.metrics.get(metric_name, 0.0)) for r in seed_results]
    stats = aggregate_seed_stats(vals)

    # Check 1: Channel power normalization drift
    pow_ok = True
    for r in seed_results:
        for pk in ("mean_channel_power", "mean_small_scale_power"):
            if pk in r.metrics:
                p_val = float(r.metrics[pk])
                if p_val < 0.5 or p_val > 1.5:
                    pow_ok = False
        for tk in ("h", "h_small"):
            if tk in r.tensors and isinstance(r.tensors[tk], torch.Tensor):
                t_pow = float(torch.mean(r.tensors[tk].abs().pow(2)).item())
                if t_pow < 0.5 or t_pow > 1.5:
                    pow_ok = False
    checks["channel_power_normalization_ok"] = pow_ok
    if not pow_ok:
        red_flags.append("channel_power_normalization_drift")
        recommendations.append("Ensure E[|h|^2]=1.0 per spatial entry (missing 1/sqrt(L) or 1/sqrt(2) inflates SNR artificially).")

    # Check 2: Unit & Noise / Finiteness consistency
    unit_ok = True
    for r in seed_results:
        if any(not torch.isfinite(torch.as_tensor(v).real).all().item() for v in r.tensors.values() if isinstance(v, torch.Tensor)):
            unit_ok = False
        if "snr_db" in r.metrics and abs(r.metrics["snr_db"]) > 150.0:
            unit_ok = False
        if r.diagnostics.get("unit_mismatch", False):
            unit_ok = False
    checks["snr_and_noise_units_ok"] = unit_ok
    if not unit_ok:
        red_flags.append("snr_noise_unit_mismatch")
        recommendations.append("Verify dB vs linear conversions and k_B * T * B * NF noise floor integration.")

    # Check 3: Genie / Future CSI leakage guard
    csi_ok = True
    for r in seed_results:
        if r.diagnostics.get("has_future_csi_leakage", False):
            csi_ok = False
    if wce_spec is not None and wce_spec.csi_assumption == "ideal_genie_baseline" and not has_classic_baseline:
        csi_ok = False
    checks["causal_csi_no_leakage_ok"] = csi_ok
    if not csi_ok:
        red_flags.append("genie_csi_or_future_leakage")
        recommendations.append("Use causal/noisy CSI (`ls_lmmse_csi` or delayed CSI) and avoid reading t+1 channel realizations.")

    # Check 4: Doppler & coherence time self-consistency
    doppler_ok = True
    for r in seed_results:
        gap = r.diagnostics.get("lag1_autocorr_gap")
        if gap is not None and float(gap) > 0.25:
            doppler_ok = False
        if r.diagnostics.get("doppler_coherence_broken", False):
            doppler_ok = False
    checks["doppler_coherence_ok"] = doppler_ok
    if not doppler_ok:
        red_flags.append("doppler_coherence_violation")
        recommendations.append("Ensure time-series slots satisfy Clarke autocorrelation J0(2*pi*f_d*dt) instead of i.i.d. jumps.")

    # Check 5: Passive RIS energy conservation
    ris_ok = True
    for r in seed_results:
        if r.diagnostics.get("is_passive_ris", False):
            max_mod = float(r.diagnostics.get("max_ris_modulus", 1.0))
            if max_mod > 1.0 + 1e-4:
                ris_ok = False
    checks["passive_ris_energy_conservation_ok"] = ris_ok
    if not ris_ok:
        red_flags.append("passive_ris_energy_conservation_violation")
        recommendations.append("Passive RIS reflection modulus must satisfy |phi_m| <= 1.0; use `active_ris_flexible` with dynamic noise if amplifying.")

    # Check 6: Near-field vs far-field Rayleigh distance regime check
    nearfield_ok = True
    for r in seed_results:
        d_min = r.diagnostics.get("min_tx_rx_distance_m")
        zr = r.diagnostics.get("rayleigh_distance_m")
        if d_min is not None and zr is not None and float(d_min) < float(zr) and r.diagnostics.get("assumed_farfield", False):
            nearfield_ok = False
    if wce_spec is not None:
        phys = wce_spec.derived_physics_summary()
        if phys["is_nearfield_regime"] > 0.5 and "nearfield_xlmimo_thz" not in wce_spec.blocks_selected:
            nearfield_ok = False
    checks["nearfield_rayleigh_regime_ok"] = nearfield_ok
    if not nearfield_ok:
        red_flags.append("farfield_assumption_in_nearfield_regime")
        recommendations.append("Distance < Rayleigh distance Z_R=2D^2/lambda; switch to spherical steering (`nearfield_xlmimo_thz`).")

    # Check 7: Multi-seed count, stochastic variance, & classical baseline inclusion
    seeds_ok = len(seed_results) >= min_seeds and has_classic_baseline
    checks["multi_seed_and_baseline_ok"] = seeds_ok
    if len(seed_results) < min_seeds:
        red_flags.append("single_seed_survivorship_risk")
        recommendations.append(f"Run at least {min_seeds} seeds to compute statistically sound IQM & mean ± std.")
    if not has_classic_baseline:
        red_flags.append("missing_classic_baseline_reference")
        recommendations.append("Always benchmark against at least one classic analytical baseline (AWGN / WMMSE / PF / LS-LMMSE).")

    analytic_whitelist = {"shannon_capacity_mbps", "fbl_achievable_rate_bps_hz", "los_power_fraction"}
    is_analytic = (
        metric_name in analytic_whitelist
        or any(r.diagnostics.get("is_deterministic_analytic", False) for r in seed_results)
    )
    stochastic_ok = True
    if len(seed_results) >= 2 and not is_analytic and stats.get("std", 0.0) <= 1e-12:
        stochastic_ok = False
        red_flags.append("zero_variance_deterministic_metric")
        recommendations.append(
            f"Primary metric '{metric_name}' had zero cross-seed std across {len(seed_results)} seeds; "
            "wire stochastic channel realizations into the metric instead of closed-form mean proxies."
        )
    checks["multi_seed_stochastic_variance_ok"] = stochastic_ok

    # Check 9: RIS phase quantization / physical array aperture self-consistency
    aperture_ok = True
    if wce_spec is not None:
        if wce_spec.num_ris_elements < 0 or wce_spec.ris_phase_bits < 0:
            aperture_ok = False
        if wce_spec.num_tx >= 16 and wce_spec.aperture_m > 0.0:
            lam_m = 2.99792458e8 / max(wce_spec.carrier_fc_ghz * 1e9, 1e6)
            # Even dense 2D sub-arrays have diagonal aperture >= 0.25 * sqrt(Nt) * 0.5 * lam
            min_phys_ap = 0.1 * (wce_spec.num_tx ** 0.5) * 0.5 * lam_m
            if wce_spec.aperture_m < min_phys_ap:
                aperture_ok = False
    if not aperture_ok:
        red_flags.append("ris_or_aperture_physics_mismatch")
        recommendations.append(
            "Verify WCESpec num_ris_elements >= 0, ris_phase_bits >= 0, and physical array aperture_m >= D_min(num_tx, fc_ghz)."
        )
    checks["ris_quantization_and_aperture_ok"] = aperture_ok

    # Check 10: Table I provenance integrity (no contradictory duplicates or missing citation keys)
    table1_ok = True
    if wce_spec is not None and wce_spec.parameter_provenance:
        seen_t1: set[str] = set()
        for row in wce_spec.parameter_provenance:
            pkey = str(row.parameter)
            if pkey in seen_t1 or not str(row.citation_key).strip():
                table1_ok = False
                break
            seen_t1.add(pkey)
    if not table1_ok:
        red_flags.append("table1_provenance_integrity_violation")
        recommendations.append("Ensure Table I parameter_provenance rows have unique parameter names and non-empty citation keys.")
    checks["table1_provenance_integrity_ok"] = table1_ok

    return RigorAuditReport(
        passed=len(red_flags) == 0,
        red_flags=red_flags,
        checks=checks,
        primary_metric_name=metric_name,
        seed_stats=stats,
        recommendations=recommendations,
    )
