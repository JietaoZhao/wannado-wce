"""Benchmark 1: 3GPP TR 38.901 / TR 38.811 NTN / ITU-R IMT-2030 FBL & Multi-Band Validation."""

from __future__ import annotations

import csv
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np

from wannado.wce.blocks.channel_3gpp import (
    provision_ntn_leo_satellite,
    provision_tr38901_inf,
    provision_tr38901_uma_umi_rma,
)
from wannado.wce.blocks.domain_6g import provision_hrllc_fbl_urllc
from wannado.wce.standards.gpp_38901 import compute_pathloss_tr38901_db


def main() -> None:
    root = Path(__file__).resolve().parent
    assets_dir = root / "assets"
    outputs_dir = root / "outputs"
    assets_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    rows: list[list[object]] = []

    # 1. 3GPP TR 38.901 UMa / UMi / RMa analytical pathloss curves across distance
    d_grid = np.geomspace(20.0, 1000.0, 40)
    pl_uma_los, _ = compute_pathloss_tr38901_db("UMa", d_grid, 3.5, is_los=True, h_bs_m=25.0)
    pl_uma_nlos, _ = compute_pathloss_tr38901_db("UMa", d_grid, 3.5, is_los=False, h_bs_m=25.0)
    pl_umi_nlos, _ = compute_pathloss_tr38901_db("UMi", d_grid, 28.0, is_los=False, h_bs_m=10.0)
    pl_rma_los, _ = compute_pathloss_tr38901_db("RMa", d_grid, 0.7, is_los=True, h_bs_m=35.0)

    # Scenario runs across UMa / UMi / RMa / InF
    for scen in ("UMa", "UMi", "RMa"):
        res = provision_tr38901_uma_umi_rma(scenario=scen, num_links=512, seed=42)
        rows.append(["tr38901_" + scen.lower(), "median_snr_db", res.metrics["median_snr_db"]])
        rows.append(["tr38901_" + scen.lower(), "mean_se_bps_hz", res.metrics["mean_spectral_efficiency_bps_hz"]])
    for inf_sub in ("InF-SL", "InF-DL", "InF-DH"):
        res = provision_tr38901_inf(scenario=inf_sub, num_links=512, seed=42)
        rows.append([inf_sub.lower(), "p5_edge_snr_db", res.metrics["p5_edge_snr_db"]])

    # 2. NTN LEO TR 38.811 Elevation Sweep (10..90 deg)
    elevs = [10.0, 20.0, 30.0, 45.0, 60.0, 75.0, 90.0]
    ntn_s_snr: list[float] = []
    ntn_ka_snr: list[float] = []
    for el in elevs:
        rs = provision_ntn_leo_satellite(elevation_deg=el, fc_ghz=2.0, bandwidth_hz=20e6, eirp_dbw=34.0, seed=42)
        rka = provision_ntn_leo_satellite(elevation_deg=el, fc_ghz=20.0, bandwidth_hz=50e6, eirp_dbw=44.0, seed=42)
        ntn_s_snr.append(rs.metrics["median_snr_db"])
        ntn_ka_snr.append(rka.metrics["median_snr_db"])
        rows.append([f"ntn_s_el{int(el)}", "median_snr_db", rs.metrics["median_snr_db"]])
        rows.append([f"ntn_ka_el{int(el)}", "median_snr_db", rka.metrics["median_snr_db"]])

    # 3. IMT-2030 HRLLC Finite Blocklength (Polyanskiy 2010) vs Shannon Capacity
    blocklengths = [64, 128, 256, 512, 1024, 2048, 4096]
    fbl_rates: list[float] = []
    shannon_rates: list[float] = []
    for n_blk in blocklengths:
        rf = provision_hrllc_fbl_urllc(snr_db=10.0, blocklength_n=n_blk, target_bler=1e-5, seed=42)
        fbl_rates.append(rf.metrics["fbl_achievable_rate_bps_hz"])
        shannon_rates.append(rf.metrics["shannon_capacity_bps_hz"])
        rows.append([f"hrllc_n{n_blk}", "fbl_rate_bps_hz", rf.metrics["fbl_achievable_rate_bps_hz"]])

    # Save CSV
    csv_path = outputs_dir / "3gpp_imt2030_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["benchmark_case", "metric", "value"])
        writer.writerows(rows)

    # Plot 3-panel publication figure
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), dpi=160)

    axes[0].semilogx(d_grid, pl_uma_los, label="UMa LOS (3.5 GHz)", lw=2.0)
    axes[0].semilogx(d_grid, pl_uma_nlos, label="UMa NLOS (3.5 GHz)", lw=2.0, ls="--")
    axes[0].semilogx(d_grid, pl_umi_nlos, label="UMi NLOS (28 GHz)", lw=2.0)
    axes[0].semilogx(d_grid, pl_rma_los, label="RMa LOS (0.7 GHz)", lw=2.0, ls="-.")
    axes[0].set_title("3GPP TR 38.901 Pathloss vs Distance", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("2D Distance $d_{2D}$ [m]")
    axes[0].set_ylabel("Pathloss [dB]")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8.5)

    axes[1].plot(elevs, ntn_s_snr, marker="o", lw=2.2, label="LEO 600km S-band (2 GHz, 20 MHz)")
    axes[1].plot(elevs, ntn_ka_snr, marker="s", lw=2.2, label="LEO 600km Ka-band (20 GHz, 50 MHz)")
    axes[1].set_title("3GPP TR 38.811 LEO NTN Median SNR", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Satellite Elevation Angle [deg]")
    axes[1].set_ylabel("Median Rx SNR [dB]")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8.5)

    axes[2].semilogx(blocklengths, shannon_rates, "--", color="#64748b", lw=2.0, label="Shannon Limit ($n \\to \\infty$)")
    axes[2].semilogx(blocklengths, fbl_rates, marker="D", color="#dc2626", lw=2.2, label="Polyanskiy FBL ($\\epsilon = 10^{-5}$)")
    axes[2].set_title("IMT-2030 HRLLC Short-Packet Penalty", fontsize=11, fontweight="bold")
    axes[2].set_xlabel("Blocklength $n$ [channel uses]")
    axes[2].set_ylabel("Achievable Rate [bit/s/Hz]")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(fontsize=8.5)

    fig.tight_layout()
    fig_path = assets_dir / "3gpp_imt2030_channels.png"
    fig.savefig(fig_path, dpi=160)
    plt.close(fig)
    print(f"Saved Figure: {fig_path}")
    print(f"Saved CSV   : {csv_path}")


if __name__ == "__main__":
    main()
