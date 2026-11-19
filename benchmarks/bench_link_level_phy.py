"""Benchmark 3: Pure-PyTorch OFDM-MIMO Link-Level PHY Waterfalls (LDPC Coded BLER, Equalizers, CSI Estimators)."""

from __future__ import annotations

import csv
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt

from wannado.wce.blocks.channel_3gpp import provision_tr38901_cdl_tdl
from wannado.wce.intake.composer import compose_wce_spec
from wannado.wce.workspace.link_level_phy import sweep_link_level_bler


def main() -> None:
    root = Path(__file__).resolve().parent
    assets_dir = root / "assets"
    outputs_dir = root / "outputs"
    assets_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    rows: list[list[object]] = []
    cdl_c = provision_tr38901_cdl_tdl(profile="CDL-C", batch_size=32, num_tx=4, num_rx=2, num_subcarriers=32, seed=42)

    snrs_mod = [2.0, 6.0, 10.0, 14.0, 18.0, 22.0]
    mod_curves: dict[str, list[float]] = {}
    for mod in ("QPSK", "16QAM", "64QAM"):
        wf = sweep_link_level_bler(
            cdl_c,
            snr_db_list=snrs_mod,
            modulation=mod,
            equalizer="mmse",
            estimator="ls",
            coded=True,
            num_blocks=48,
            seed=42,
        )
        mod_curves[mod] = [r.coded_bler for r in wf]
        for r in wf:
            rows.append([f"cdlc_{mod}", f"snr_{int(r.snr_db)}", r.coded_bler, r.uncoded_ber, r.effective_goodput_bps_hz])

    snrs_eq = [4.0, 8.0, 12.0, 16.0, 20.0]
    eq_curves: dict[str, list[float]] = {}
    for label, eq, est in [
        ("MMSE + Genie CSI", "mmse", "genie"),
        ("MMSE + LMMSE Pilot CSI", "mmse", "lmmse"),
        ("MMSE + LS Pilot CSI", "mmse", "ls"),
        ("ZF + LS Pilot CSI", "zf", "ls"),
    ]:
        wf = sweep_link_level_bler(
            cdl_c,
            snr_db_list=snrs_eq,
            modulation="16QAM",
            equalizer=eq,  # type: ignore[arg-type]
            estimator=est,  # type: ignore[arg-type]
            coded=True,
            num_blocks=48,
            seed=42,
        )
        eq_curves[label] = [r.coded_bler for r in wf]
        for r in wf:
            rows.append([label.replace(" ", "_"), f"snr_{int(r.snr_db)}", r.coded_bler, r.uncoded_ber, r.effective_goodput_bps_hz])

    goodput_profiles: dict[str, list[float]] = {}
    for label, prompt in [
        ("28 GHz 64-RIS MIMO", "64-element RIS MIMO at 28 GHz"),
        ("3.5 GHz UMa 16x2 MIMO", "3.5 GHz urban cellular 16x2 MIMO"),
        ("20 GHz Ka-band LEO NTN", "LEO NTN satellite downlink at 20 GHz Ka-band"),
    ]:
        sp, _ = compose_wce_spec(prompt)
        wf = sweep_link_level_bler(
            sp,
            snr_db_list=snrs_eq,
            modulation="QPSK",
            equalizer="mmse",
            estimator="ls",
            coded=True,
            num_blocks=32,
            seed=42,
        )
        goodput_profiles[label] = [r.effective_goodput_bps_hz for r in wf]
        for r in wf:
            rows.append([label.replace(" ", "_"), f"snr_{int(r.snr_db)}", r.coded_bler, r.uncoded_ber, r.effective_goodput_bps_hz])

    csv_path = outputs_dir / "link_level_phy_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["case", "snr", "coded_bler", "uncoded_ber", "effective_goodput_bps_hz"])
        writer.writerows(rows)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), dpi=160)

    for mod, color, marker in [("QPSK", "#16a34a", "o"), ("16QAM", "#2563eb", "s"), ("64QAM", "#dc2626", "^")]:
        axes[0].semilogy(snrs_mod, [max(v, 1e-3) for v in mod_curves[mod]], marker=marker, lw=2.2, color=color, label=f"{mod} (LDPC R=1/2)")
    axes[0].set_title("CDL-C 4x2 MIMO-OFDM Coded BLER", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Receive SNR [dB]")
    axes[0].set_ylabel("LDPC Block Error Rate (BLER)")
    axes[0].set_ylim(8e-4, 1.2)
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=8.5)

    for (label, vals), ls, marker in zip(eq_curves.items(), ["-", "-", "--", ":"], ["o", "s", "D", "x"], strict=True):
        axes[1].semilogy(snrs_eq, [max(v, 1e-3) for v in vals], ls=ls, marker=marker, lw=2.0, label=label)
    axes[1].set_title("Equalizer & Pilot CSI Ablation (16QAM)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Receive SNR [dB]")
    axes[1].set_ylabel("Coded BLER")
    axes[1].set_ylim(8e-4, 1.2)
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend(fontsize=8.0)

    for label, vals in goodput_profiles.items():
        axes[2].plot(snrs_eq, vals, marker="o", lw=2.2, label=label)
    axes[2].set_title("Composed WCE End-to-End Goodput (QPSK)", fontsize=11, fontweight="bold")
    axes[2].set_xlabel("Receive SNR [dB]")
    axes[2].set_ylabel("Effective Goodput [bit/s/Hz]")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(fontsize=8.0)

    fig.tight_layout()
    fig_path = assets_dir / "link_level_ofdm_bler.png"
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Figure: {fig_path}")
    print(f"Saved CSV   : {csv_path}")


if __name__ == "__main__":
    main()
