"""3GPP TR 38.901 / TR 38.811 / ITU-R P.676 Quantitative Reference Calibration Self-Test Suite."""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from wannado.wce.standards.etsi_ieee import itur_p676_path_absorption_db
from wannado.wce.standards.gpp_38901 import compute_o2i_penetration_db, compute_pathloss_tr38901_db
from wannado.wce.standards.gpp_38901_3d import tr38901_antenna_pattern_db


@dataclass
class CalibrationCheckResult:
    """Single quantitative benchmark check comparing WCE physics against 3GPP/ITU reference bounds."""

    check_id: str
    standard_ref: str
    measured_value: float
    expected_range: tuple[float, float]
    passed: bool
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CalibrationSuiteReport:
    """Full 3GPP / ITU-R physics calibration verification dossier."""

    passed: bool
    num_checks: int
    num_passed: int
    elapsed_seconds: float
    checks: list[CalibrationCheckResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "num_checks": self.num_checks,
            "num_passed": self.num_passed,
            "elapsed_seconds": self.elapsed_seconds,
            "checks": [c.to_dict() for c in self.checks],
        }

    def format_report(self) -> str:
        verdict = "PASS" if self.passed else "FAIL"
        lines = [
            f"[wannado-wce 3GPP/ITU Calibration Suite: {verdict}] ({self.num_passed}/{self.num_checks} passed in {self.elapsed_seconds:.2f}s)"
        ]
        for c in self.checks:
            tag = "OK" if c.passed else "FAIL"
            lines.append(
                f"  - [{tag}] {c.check_id} ({c.standard_ref}): val={c.measured_value:.3f}, "
                f"expected=[{c.expected_range[0]:.2f}, {c.expected_range[1]:.2f}] — {c.detail}"
            )
        return "\n".join(lines)


def run_3gpp_calibration_suite(*, seed: int = 42, num_links: int = 512) -> CalibrationSuiteReport:
    """Run 7 deterministic 3GPP TR 38.901 / TR 38.811 / ITU-R P.676 reference calibration checks."""
    from wannado.wce.blocks.channel_3gpp.tr38811_ntn_leo import provision_ntn_leo_satellite
    from wannado.wce.blocks.channel_3gpp.tr38901_cdl_tdl import provision_tr38901_cdl_tdl
    from wannado.wce.blocks.channel_3gpp.tr38901_macro_micro import provision_tr38901_uma_umi_rma

    t0 = time.perf_counter()
    results: list[CalibrationCheckResult] = []

    # 1. TR 38.901 UMa 3.5 GHz Coupling / Pathloss CDF spread
    r_uma = provision_tr38901_uma_umi_rma(
        scenario="UMa",
        fc_ghz=3.5,
        d_2d_min_m=50.0,
        d_2d_max_m=500.0,
        o2i_fraction=0.0,
        num_links=num_links,
        seed=seed,
    )
    pl_uma = r_uma.tensors["pathloss_db"].detach().cpu().numpy()
    med_pl = float(np.median(pl_uma))
    p5_pl, p95_pl = float(np.percentile(pl_uma, 5.0)), float(np.percentile(pl_uma, 95.0))
    ok1 = (
        (85.0 <= med_pl <= 128.0)
        and (p95_pl > p5_pl + 5.0)
        and (0.8 <= float(r_uma.metrics["mean_small_scale_power"]) <= 1.2)
    )
    results.append(
        CalibrationCheckResult(
            check_id="tr38901_uma_3p5ghz_coupling_loss_cdf",
            standard_ref="3GPP TR 38.901 Table 7.4.1-1 / Annex A",
            measured_value=med_pl,
            expected_range=(85.0, 128.0),
            passed=ok1,
            detail=f"median PL={med_pl:.2f} dB, p5={p5_pl:.2f} dB, p95={p95_pl:.2f} dB",
        )
    )

    # 2. TR 38.901 UMi 28 GHz LoS vs NLoS pathloss & distance monotonicity
    d_grid = np.linspace(20.0, 300.0, 50)
    pl_los, _ = compute_pathloss_tr38901_db("UMi", d_grid, 28.0, is_los=True, h_bs_m=10.0)
    pl_nlos, _ = compute_pathloss_tr38901_db("UMi", d_grid, 28.0, is_los=False, h_bs_m=10.0)
    gap_28 = float(np.median(pl_nlos - pl_los))
    mono_ok = bool(np.all(np.diff(pl_los) >= -1e-6) and np.all(np.diff(pl_nlos) >= -1e-6))
    ok2 = mono_ok and (8.0 <= gap_28 <= 45.0)
    results.append(
        CalibrationCheckResult(
            check_id="tr38901_umi_28ghz_los_nlos_gap",
            standard_ref="3GPP TR 38.901 Table 7.4.1-1 (UMi 28 GHz)",
            measured_value=gap_28,
            expected_range=(8.0, 45.0),
            passed=ok2,
            detail=f"median NLoS-LoS excess={gap_28:.2f} dB, monotonic={mono_ok}",
        )
    )

    # 3. TR 38.901 Table 7.4.3-2 O2I High-Loss (IRR) vs Low-Loss (Standard Glass) at 28 GHz
    o2i_low = compute_o2i_penetration_db(28.0, model="low_loss", d_2d_in_m=5.0, rng=None)
    o2i_high = compute_o2i_penetration_db(28.0, model="high_loss", d_2d_in_m=5.0, rng=None)
    o2i_delta = float(np.mean(o2i_high - o2i_low))
    ok3 = 10.0 <= o2i_delta <= 25.0
    results.append(
        CalibrationCheckResult(
            check_id="tr38901_o2i_high_vs_low_loss_28ghz",
            standard_ref="3GPP TR 38.901 Table 7.4.3-2",
            measured_value=o2i_delta,
            expected_range=(10.0, 25.0),
            passed=ok3,
            detail=f"high_loss={float(np.mean(o2i_high)):.2f} dB vs low_loss={float(np.mean(o2i_low)):.2f} dB",
        )
    )

    # 4. TR 38.901 Clause 7.7.1 CDL-C Empirical RMS Delay Spread vs Scaled DS_ns
    r_cdl = provision_tr38901_cdl_tdl(
        profile="CDL-C",
        batch_size=8,
        num_tx=4,
        num_rx=2,
        num_subcarriers=32,
        num_slots=4,
        delay_spread_ns=100.0,
        seed=seed,
    )
    emp_ds = float(r_cdl.metrics["rms_delay_spread_ns"])
    p_cdl = float(r_cdl.metrics["mean_channel_power"])
    ok4 = (85.0 <= emp_ds <= 125.0) and (0.8 <= p_cdl <= 1.2)
    results.append(
        CalibrationCheckResult(
            check_id="tr38901_cdl_rms_delay_spread_accuracy",
            standard_ref="3GPP TR 38.901 Table 7.7.1-3 (CDL-C)",
            measured_value=emp_ds,
            expected_range=(85.0, 125.0),
            passed=ok4,
            detail=f"RMS DS={emp_ds:.2f} ns (target ~100 ns), E[|H|^2]={p_cdl:.3f}",
        )
    )

    # 5. TR 38.901 Table 7.3-1 3D Directional Antenna Element Pattern (HPBW=65 deg, FBR=30 dB, Gmax=8 dBi)
    g0 = float(tr38901_antenna_pattern_db(90.0, 0.0))
    ghpbw = float(tr38901_antenna_pattern_db(90.0, 32.5))
    gback = float(tr38901_antenna_pattern_db(90.0, 180.0))
    fbr = g0 - gback
    ok5 = math.isclose(g0, 8.0, abs_tol=1e-4) and math.isclose(g0 - ghpbw, 3.0, abs_tol=1e-4) and math.isclose(fbr, 30.0, abs_tol=1e-4)
    results.append(
        CalibrationCheckResult(
            check_id="tr38901_3d_antenna_pattern_fbr",
            standard_ref="3GPP TR 38.901 Table 7.3-1",
            measured_value=fbr,
            expected_range=(29.9, 30.1),
            passed=ok5,
            detail=f"boresight={g0:.2f} dBi, HPBW_delta={g0 - ghpbw:.2f} dB, FBR={fbr:.2f} dB",
        )
    )

    # 6. TR 38.811 LEO 600 km S-Band Elevation SNR Monotonicity (10 deg -> 90 deg)
    ntn_lo = provision_ntn_leo_satellite(orbit_altitude_km=600.0, fc_ghz=2.0, elevation_deg=10.0, num_samples=128, seed=seed)
    ntn_hi = provision_ntn_leo_satellite(orbit_altitude_km=600.0, fc_ghz=2.0, elevation_deg=90.0, num_samples=128, seed=seed)
    snr_gain_el = float(ntn_hi.metrics["median_snr_db"]) - float(ntn_lo.metrics["median_snr_db"])
    ok6 = 6.0 <= snr_gain_el <= 22.0
    results.append(
        CalibrationCheckResult(
            check_id="tr38811_ntn_leo_elevation_monotonicity",
            standard_ref="3GPP TR 38.811 Clause 6.6.2 (LEO 600 km)",
            measured_value=snr_gain_el,
            expected_range=(6.0, 22.0),
            passed=ok6,
            detail=f"SNR(90deg)-SNR(10deg)={snr_gain_el:+.2f} dB",
        )
    )

    # 7. ITU-R P.676-13 Oxygen Absorption Resonance Peak at 60 GHz (> 28 GHz & > 100 GHz)
    a28 = float(np.mean(itur_p676_path_absorption_db(28.0, 1000.0)))
    a60 = float(np.mean(itur_p676_path_absorption_db(60.0, 1000.0)))
    a100 = float(np.mean(itur_p676_path_absorption_db(100.0, 1000.0)))
    ok7 = (a60 >= 10.0) and (a60 > a28 * 5.0) and (a60 > a100 * 3.0)
    results.append(
        CalibrationCheckResult(
            check_id="itur_p676_oxygen_absorption_60ghz_peak",
            standard_ref="ITU-R P.676-13 (60 GHz O2 Band)",
            measured_value=a60,
            expected_range=(10.0, 25.0),
            passed=ok7,
            detail=f"1km atten: 28GHz={a28:.2f} dB, 60GHz={a60:.2f} dB, 100GHz={a100:.2f} dB",
        )
    )

    num_passed = sum(1 for r in results if r.passed)
    return CalibrationSuiteReport(
        passed=(num_passed == len(results)),
        num_checks=len(results),
        num_passed=num_passed,
        elapsed_seconds=round(time.perf_counter() - t0, 4),
        checks=results,
    )


__all__ = [
    "CalibrationCheckResult",
    "CalibrationSuiteReport",
    "run_3gpp_calibration_suite",
]
