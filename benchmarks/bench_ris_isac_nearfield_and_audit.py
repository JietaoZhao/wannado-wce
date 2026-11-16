"""Benchmark 2: Near-Field XL-MIMO, Active/Passive RIS, WMMSE/LMMSE Baselines & Stage 4 Provenance Export."""

from __future__ import annotations

import csv
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wannado.wce.blocks.arrays_and_ris import (
    provision_active_ris_flexible,
    provision_nearfield_xlmimo_thz,
    provision_passive_ris_cascade,
)
from wannado.wce.blocks.baselines import provision_ls_lmmse_csi, provision_wmmse_precoder
from wannado.wce.intake import build_wce_archetype
from wannado.wce.reporter import evaluate_wce_suite


def main() -> None:
    root = Path(__file__).resolve().parent
    assets_dir = root / "assets"
    outputs_dir = root / "outputs"
    assets_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    rows: list[list[object]] = []

    # 1. Near-field Rayleigh distance breakdown at 100 GHz (aperture ~0.19 m -> Z_R ~ 24 m)
    distances = [3.0, 5.0, 10.0, 15.0, 25.0, 40.0, 80.0, 120.0]
    ff_eff: list[float] = []
    zr_val = 0.0
    for d in distances:
        rn = provision_nearfield_xlmimo_thz(num_tx_x=128, num_tx_y=1, fc_ghz=100.0, user_distance_m=d, batch_size=32, seed=42)
        ff_eff.append(rn.metrics["farfield_beam_alignment_efficiency"])
        zr_val = rn.metrics["rayleigh_distance_m"]
        rows.append([f"nearfield_d{int(d)}m", "farfield_alignment_eff", rn.metrics["farfield_beam_alignment_efficiency"]])

    # 2. Passive vs Active RIS SNR across number of elements
    ris_sizes = [16, 32, 64, 128]
    passive_gain: list[float] = []
    active_snr_db_gain: list[float] = []
    for n_ris in ris_sizes:
        rp = provision_passive_ris_cascade(num_ris_elements=n_ris, batch_size=64, seed=42)
        ra = provision_active_ris_flexible(architecture="active_ris", num_ris_elements=n_ris, batch_size=64, seed=42)
        passive_gain.append(rp.metrics["ris_gain_ratio_vs_random"])
        active_snr_db_gain.append(ra.metrics["active_gain_over_passive"])
        rows.append([f"passive_ris_n{n_ris}", "gain_vs_random", rp.metrics["ris_gain_ratio_vs_random"]])
        rows.append([f"active_ris_n{n_ris}", "gain_over_passive", ra.metrics["active_gain_over_passive"]])

    # 3. Analytical Precoding (WMMSE vs ZF vs MRT) & CSI (LMMSE vs LS) across SNR
    snrs = [-5.0, 0.0, 5.0, 10.0, 15.0, 20.0]
    r_wmmse, r_zf, r_mrt = [], [], []
    nmse_ls, nmse_lmmse = [], []
    for snr in snrs:
        rw = provision_wmmse_precoder(batch_size=24, num_tx=8, num_users=4, snr_db=snr, num_iters=25, seed=42)
        rc = provision_ls_lmmse_csi(pilot_snr_db=snr, corr_rho=0.85, batch_size=64, seed=42)
        r_wmmse.append(rw.metrics["wmmse_sum_rate_bps_hz"])
        r_zf.append(rw.metrics["zf_sum_rate_bps_hz"])
        r_mrt.append(rw.metrics["mrt_sum_rate_bps_hz"])
        nmse_ls.append(rc.metrics["ls_nmse_db"])
        nmse_lmmse.append(rc.metrics["lmmse_nmse_db"])
        rows.append([f"wmmse_snr{int(snr)}", "wmmse_sum_rate_bps_hz", rw.metrics["wmmse_sum_rate_bps_hz"]])
        rows.append([f"csi_snr{int(snr)}", "lmmse_nmse_db", rc.metrics["lmmse_nmse_db"]])

    csv_path = outputs_dir / "ris_nearfield_baselines.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["benchmark_case", "metric", "value"])
        writer.writerows(rows)

    # Plot 3-panel publication figure
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), dpi=160)

    axes[0].semilogx(distances, ff_eff, marker="o", lw=2.2, color="#2563eb", label="Far-Field Planar Beam Efficiency")
    axes[0].axvline(zr_val, color="#dc2626", ls="--", lw=1.8, label=f"Rayleigh $Z_R = {zr_val:.1f}$ m")
    axes[0].set_title("Near-Field XL-MIMO (100 GHz, $N_t=128$)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("User Distance $d$ [m]")
    axes[0].set_ylabel("Far-Field Beam Alignment Efficiency")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8.5)

    axes[1].plot(snrs, r_wmmse, marker="o", lw=2.2, color="#16a34a", label="WMMSE (Shi et al. 2011)")
    axes[1].plot(snrs, r_zf, marker="s", lw=2.0, color="#2563eb", label="Zero-Forcing (ZF)")
    axes[1].plot(snrs, r_mrt, marker="^", lw=2.0, ls="--", color="#f59e0b", label="MRT")
    axes[1].set_title("8x4 MU-MISO Sum-Rate Baselines", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Transmit SNR [dB]")
    axes[1].set_ylabel("Sum-Rate [bit/s/Hz]")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8.5)

    axes[2].plot(snrs, nmse_ls, marker="x", lw=2.0, ls="--", color="#dc2626", label="Pilot LS Estimator")
    axes[2].plot(snrs, nmse_lmmse, marker="D", lw=2.2, color="#7c3aed", label="Spatial LMMSE Estimator")
    axes[2].set_title("Pilot CSI Estimation ($\\rho=0.85$)", fontsize=11, fontweight="bold")
    axes[2].set_xlabel("Pilot SNR [dB]")
    axes[2].set_ylabel("Normalized MSE [dB]")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(fontsize=8.5)

    fig.tight_layout()
    fig_path = assets_dir / "ris_nearfield_baselines.png"
    fig.savefig(fig_path, dpi=160)
    plt.close(fig)

    # Full Stage 4 ISAC + RIS Provenance Export (Table I + BibTeX + Rigor Audit)
    spec = build_wce_archetype("imt2030_isac_monostatic")
    report = evaluate_wce_suite(
        spec.blocks_selected + spec.baseline_blocks,
        wce_spec=spec,
        seeds=(0, 1, 2),
        baseline_blocks=spec.baseline_blocks,
    )
    saved = report.save(outputs_dir / "isac_provenance")
    print(f"Saved Figure : {fig_path}")
    print(f"Saved CSV    : {csv_path}")
    print(f"Saved Report : {saved['environment_provenance_md']}")
    print(f"Saved Table I: {saved['paper_params_table_tex']}")


if __name__ == "__main__":
    main()
