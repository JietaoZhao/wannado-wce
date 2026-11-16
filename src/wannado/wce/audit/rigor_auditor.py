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
            "median_snr_db",
            "mean_channel_power",
        ):
            if cand in seed_results[0].metrics:
                metric_name = cand
                break
        if not metric_name and seed_results[0].metrics:
            metric_name = next(iter(seed_results[0].metrics.keys()))

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

    # Check 7: Multi-seed count & classical baseline inclusion
    seeds_ok = len(seed_results) >= min_seeds and has_classic_baseline
    checks["multi_seed_and_baseline_ok"] = seeds_ok
    if len(seed_results) < min_seeds:
        red_flags.append("single_seed_survivorship_risk")
        recommendations.append(f"Run at least {min_seeds} seeds to compute statistically sound IQM & mean ± std.")
    if not has_classic_baseline:
        red_flags.append("missing_classic_baseline_reference")
        recommendations.append("Always benchmark against at least one classic analytical baseline (AWGN / WMMSE / PF / LS-LMMSE).")

    return RigorAuditReport(
        passed=len(red_flags) == 0,
        red_flags=red_flags,
        checks=checks,
        primary_metric_name=metric_name,
        seed_stats=stats,
        recommendations=recommendations,
    )
