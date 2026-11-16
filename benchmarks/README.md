# `wannado-wce` Benchmarks & Standards Validation Suite

Reproducible validation scripts verifying physical laws, 3GPP / ITU-R / ETSI standard compliance, near-field vs far-field boundaries, analytical baselines (Water-Filling, WMMSE, LMMSE, Proportional Fair), and Stage 4 provenance artifact generation.

## Scripts

| Script | What It Validates | Output Artifacts |
| :--- | :--- | :--- |
| [bench_3gpp_and_imt2030.py](bench_3gpp_and_imt2030.py) | 3GPP TR 38.901 UMa/UMi/RMa/InF pathloss curves, TR 38.811 LEO NTN elevation SNR sweep (S/Ka bands), Polyanskiy FBL vs Shannon URLLC rate gap | `assets/3gpp_imt2030_channels.png`, `outputs/3gpp_imt2030_metrics.csv` |
| [bench_ris_isac_nearfield_and_audit.py](bench_ris_isac_nearfield_and_audit.py) | 100 GHz XL-MIMO Rayleigh-distance beam efficiency, Passive vs Active RIS gains, WMMSE/ZF/MRT & LMMSE/LS curves, Stage 4 ISAC provenance + LaTeX Table I export | `assets/ris_nearfield_baselines.png`, `outputs/ris_nearfield_baselines.csv`, `outputs/isac_provenance/*` |

## Quick Run

```bash
python3 benchmarks/bench_3gpp_and_imt2030.py
python3 benchmarks/bench_ris_isac_nearfield_and_audit.py
```

All charts under `benchmarks/assets/*.png` are tracked cleanly for README verification, while scratch tables and intermediate reports under `benchmarks/outputs/` are git-ignored.
