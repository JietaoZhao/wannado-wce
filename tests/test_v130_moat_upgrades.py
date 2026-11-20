"""v1.3.0 Structural Moat Tests: RF Impairments + SigMF, QC-LDPC 256QAM + HARQ/AMC, 3D Dual-Pol + MU-MIMO + Spatial Consistency, Differentiable E2E PHY, 3GPP Calibration, O-RAN KPM v3 & Sealed Dossiers."""

from __future__ import annotations

import io
import json
import math
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import torch

from wannado.wce import (
    ChannelModuleAdapter,
    ComposedWirelessEnv,
    DifferentiableRISChannel,
    DifferentiableWirelessLink,
    ORANKPMTelemetryAdapter,
    QCLDPCCodec,
    RFImpairmentChain,
    RFImpairmentConfig,
    SpatialConsistencyField,
    apply_adc_quantization,
    apply_cfo,
    apply_iq_imbalance,
    apply_phase_noise,
    apply_rapp_pa,
    build_qam_constellation,
    build_wce_dossier,
    compare_iq_psd_and_stats,
    compare_wce_verdicts,
    compose_wce_spec,
    evaluate_harq_amc_link,
    evaluate_link_level_phy,
    generate_tr38901_3d_dualpol_channel,
    measure_aclr_db,
    provision_tr38901_cdl_tdl,
    read_sigmf_recording,
    run_3gpp_calibration_suite,
    select_nr_mcs_from_sinr,
    tr38901_antenna_pattern_db,
    verify_wce_dossier_integrity,
    verify_wce_target,
    write_sigmf_recording,
)
from wannado.wce.cli import main as cli_main


class TestV130MoatUpgrades(unittest.TestCase):
    # ------------------------------------------------------------------
    # Gap 1: Differentiable RF Hardware Impairments & Real-World SigMF IQ
    # ------------------------------------------------------------------
    def test_rf_impairments_physics_and_autograd(self) -> None:
        torch.manual_seed(10)
        # Generate bandlimited oversampled OFDM-like complex signal (zero-padded outer subcarriers)
        spec = torch.zeros((4, 256), dtype=torch.complex64)
        spec[:, 96:160] = (torch.randn(4, 64) + 1j * torch.randn(4, 64)) / math.sqrt(2.0)
        x_bandlimited = torch.fft.ifft(torch.fft.fftshift(spec, dim=-1), dim=-1) * math.sqrt(256.0 / 64.0)
        x_req = x_bandlimited.detach().clone().requires_grad_(True)

        # Heavy PA saturation (IBO=0.5 dB) should cause spectral regrowth (drop ACLR vs clean & vs high-IBO 20 dB)
        aclr_clean = measure_aclr_db(x_bandlimited, occupied_fraction=0.25)
        y_sat = apply_rapp_pa(x_req, ibo_db=0.5, smoothness_p=2.0, phase_alpha=0.25)
        y_lin = apply_rapp_pa(x_bandlimited, ibo_db=20.0, smoothness_p=2.0)
        aclr_sat = measure_aclr_db(y_sat.detach(), occupied_fraction=0.25)
        aclr_lin = measure_aclr_db(y_lin, occupied_fraction=0.25)
        self.assertGreater(aclr_clean, 40.0)
        self.assertGreater(aclr_lin, aclr_sat + 8.0)

        # Compose full RF chain (IQ mismatch + PA + CFO + Phase Noise + STE 5-bit ADC) and verify finite autograd
        cfg = RFImpairmentConfig(
            pa_enabled=True,
            pa_ibo_db=3.0,
            phase_noise_enabled=True,
            phase_noise_dbc_hz=-85.0,
            cfo_hz=5000.0,
            iq_gain_imbalance_db=1.5,
            iq_phase_imbalance_deg=6.0,
            adc_bits=5,
        )
        chain = RFImpairmentChain(cfg, seed=11)
        y_chain = chain(x_req)
        y_chain.real.pow(2).sum().backward()
        self.assertIsNotNone(x_req.grad)
        assert x_req.grad is not None
        self.assertTrue(bool(torch.isfinite(x_req.grad.real).all().item()))
        self.assertTrue(bool(torch.isfinite(x_req.grad.imag).all().item()))
        self.assertGreater(float(x_req.grad.abs().sum().item()), 0.0)

        # Unit-power preservation under IQ imbalance
        u_iq = (torch.randn(8, 256) + 1j * torch.randn(8, 256)) / math.sqrt(2.0)
        u_imb = apply_iq_imbalance(u_iq, gain_imbalance_db=3.0, phase_imbalance_deg=12.0)
        self.assertAlmostEqual(
            float(u_iq.abs().pow(2).mean().item()),
            float(u_imb.abs().pow(2).mean().item()),
            delta=0.05,
        )
        # CFO & Phase noise & ADC smoke checks
        self.assertEqual(apply_cfo(u_iq, cfo_hz=2500.0).shape, u_iq.shape)
        self.assertEqual(apply_phase_noise(u_iq, phase_noise_dbc_hz=-90.0, seed=5).shape, u_iq.shape)
        self.assertEqual(apply_adc_quantization(u_iq, num_bits=4).shape, u_iq.shape)

        # ChannelModuleAdapter integrates RFImpairmentConfig seamlessly
        ch = ChannelModuleAdapter(
            lambda b: (torch.ones(b, 2, 2, dtype=torch.complex64) / math.sqrt(2.0)),
            snr_db=20.0,
            rf_impairments=cfg,
        )
        y_ch = ch(torch.ones(4, 2, dtype=torch.complex64, requires_grad=True))
        self.assertEqual(y_ch.shape, (4, 2))
        metrics = chain.measure_distortion(x_bandlimited, occupied_fraction=0.25)
        self.assertGreater(metrics["evm_pct"], 0.1)
        self.assertLess(metrics["iq_irr_db"], 40.0)

    def test_sigmf_roundtrip_tamper_detection_and_psd_comparison(self) -> None:
        torch.manual_seed(12)
        sig = (torch.randn(1024) + 1j * torch.randn(1024)).to(torch.complex64) * 0.25
        with tempfile.TemporaryDirectory() as tmp:
            for dtype in ("cf32_le", "ci16_le"):
                stem = Path(tmp) / f"cap_{dtype}"
                meta_p, data_p = write_sigmf_recording(
                    sig,
                    stem,
                    sample_rate_hz=30.72e6,
                    center_freq_hz=3.5e9,
                    datatype=dtype,
                    description="unit test capture",
                )
                loaded, meta = read_sigmf_recording(meta_p, verify_sha512=True)
                self.assertEqual(meta.num_samples, 1024)
                self.assertAlmostEqual(meta.sample_rate_hz, 30.72e6)
                tol = 1e-5 if dtype == "cf32_le" else 1e-3
                self.assertTrue(torch.allclose(sig, loaded, atol=tol))

                # Tamper 1 byte -> SHA-512 verification must reject
                raw = bytearray(data_p.read_bytes())
                raw[0] ^= 0xFF
                data_p.write_bytes(bytes(raw))
                with self.assertRaises(ValueError):
                    read_sigmf_recording(data_p, verify_sha512=True)

            # Sim-to-Real PSD & statistical comparison
            comp_same = compare_iq_psd_and_stats(sig, sig, nfft=128)
            self.assertGreater(comp_same["psd_pearson_corr"], 0.999)
            self.assertAlmostEqual(comp_same["mean_power_delta_db"], 0.0, places=4)

    # ------------------------------------------------------------------
    # Gap 2: 3GPP QC-LDPC, 256QAM, 2D-DMRS & TS 38.214 AMC + HARQ
    # ------------------------------------------------------------------
    def test_qc_ldpc_multirate_256qam_and_harq_amc(self) -> None:
        codec = QCLDPCCodec(z_lift=16, k_blocks=4, m_blocks=8, seed=2026)
        torch.manual_seed(21)
        info = torch.randint(0, 2, (24, codec.k_info), dtype=torch.int8)
        cw = codec.encode(info)
        h_mat = torch.from_numpy(codec._h_full_np).to(torch.float32)
        synd = (cw.to(torch.float32) @ h_mat.t()).to(torch.int64) & 1
        self.assertEqual(int(synd.sum().item()), 0)

        # Verify coding gain on BPSK AWGN across rates 1/2 and 2/3
        for rate in (0.5, 2.0 / 3.0):
            pos = torch.as_tensor(codec.select_rate_matched_positions(rate, rv_index=0), dtype=torch.long)
            snr_lin = 10.0 ** (4.0 / 10.0)
            tx = 1.0 - 2.0 * cw[:, pos].float()
            rx = tx + torch.randn_like(tx) * math.sqrt(1.0 / (2.0 * snr_lin))
            llr = torch.zeros((24, codec.n_mother), dtype=torch.float32)
            llr[:, pos] = 4.0 * snr_lin * rx
            uncoded = float(((rx[:, : codec.k_info] < 0).int() != info.int()).float().mean().item())
            _, info_hat = codec.decode_min_sum(llr)
            coded = float((info_hat != info).float().mean().item())
            self.assertLessEqual(coded, uncoded)
            self.assertLess(coded, 0.01)

        # Verify 256QAM constellation unit average power & Gray adjacency (adjacent I/Q PAM differs by 1 bit)
        sym256, bits256 = build_qam_constellation("256QAM")
        self.assertEqual(sym256.shape, (256,))
        self.assertEqual(bits256.shape, (256, 8))
        self.assertAlmostEqual(float(sym256.abs().pow(2).mean().item()), 1.0, places=4)

        cdl = provision_tr38901_cdl_tdl(profile="CDL-D", batch_size=8, num_tx=4, num_rx=2, num_subcarriers=32, num_slots=14, seed=22)
        phy256 = evaluate_link_level_phy(cdl, snr_db=28.0, modulation="256QAM", equalizer="mmse", estimator="lmmse", num_blocks=16, seed=22)
        self.assertLess(phy256.coded_ber, 0.02)

        # Verify TS 38.214 AMC MCS ordering
        low_mcs = select_nr_mcs_from_sinr(0.0)
        high_mcs = select_nr_mcs_from_sinr(28.0)
        self.assertEqual(high_mcs["modulation"], "256QAM")
        self.assertGreater(high_mcs["mcs_index"], low_mcs["mcs_index"])

        # Evaluate 2D-DMRS + Chase & IR HARQ across 14-symbol CDL-C MIMO slot
        cdl_c = provision_tr38901_cdl_tdl(profile="CDL-C", batch_size=8, num_tx=4, num_rx=2, num_subcarriers=32, num_slots=14, speed_mps=5.0, seed=22)
        for mode in ("chase", "ir"):
            res = evaluate_harq_amc_link(
                cdl_c,
                snr_db=9.0,
                mcs_index=7,
                max_harq_tx=3,
                harq_mode=mode,
                num_blocks=24,
                seed=23,
            )
            self.assertEqual(res.harq_mode, mode)
            self.assertLess(res.residual_bler, res.initial_bler)
            self.assertGreater(res.effective_throughput_bps_hz, 0.2)

    # ------------------------------------------------------------------
    # Gap 3: 3D Dual-Pol TR 38.901, Spatial Consistency & MU-MIMO Interference
    # ------------------------------------------------------------------
    def test_3d_dualpol_spatial_consistency_and_mu_mimo_interference(self) -> None:
        # 1. Exact Table 7.3-1 directional antenna element pattern
        self.assertAlmostEqual(float(tr38901_antenna_pattern_db(90.0, 0.0)), 8.0, places=4)
        self.assertAlmostEqual(float(tr38901_antenna_pattern_db(90.0, 32.5)), 5.0, places=4)
        self.assertAlmostEqual(float(tr38901_antenna_pattern_db(90.0, 0.0)) - float(tr38901_antenna_pattern_db(90.0, 180.0)), 30.0, places=4)

        # 2. 3D Dual-Pol ±45 deg CDL channel
        ch3d = generate_tr38901_3d_dualpol_channel(
            "CDL-C",
            batch_size=6,
            tx_panel=(2, 2),
            rx_panel=(1, 2),
            dual_pol=True,
            num_subcarriers=24,
            num_slots=14,
            seed=31,
        )
        # Nr_tot = 2*(1*2) = 4, Nt_tot = 2*(2*2) = 8
        self.assertEqual(tuple(ch3d["h_freq"].shape), (6, 4, 8, 24, 14))
        self.assertAlmostEqual(ch3d["mean_channel_power"], 1.0, places=3)

        # 3. SpatialConsistencyField exponential correlation decay
        sf = SpatialConsistencyField(scenario="UMa", seed=32)
        p0 = np.random.default_rng(32).uniform(0.0, 500.0, size=(48, 2))
        p_near = p0 + np.array([[1.0, 0.0]])
        p_far = p0 + np.array([[400.0, 0.0]])
        z0 = sf.step_positions(p0, std=6.0)
        z_near = sf.step_positions(p_near, std=6.0)
        z_far = sf.step_positions(p_far, std=6.0)
        corr_near = float(np.corrcoef(z0, z_near)[0, 1])
        corr_far = abs(float(np.corrcoef(z_near, z_far)[0, 1]))
        self.assertGreater(corr_near, 0.90)
        self.assertLess(corr_far, 0.45)

        # 4. ComposedWirelessEnv with MU-MIMO spatial interference & spatial consistency
        spec, _ = compose_wce_spec("UMa 3.5 GHz 4-antenna MIMO multi-user scheduling")
        env_su = ComposedWirelessEnv(spec, batch_size=4, seed=33, enable_mu_mimo_interference=False)
        env_mu = ComposedWirelessEnv(
            spec,
            batch_size=4,
            seed=33,
            enable_mu_mimo_interference=True,
            enable_spatial_consistency=True,
        )
        out_su = env_su.step()
        out_mu = env_mu.step(
            rx_pos_3d=np.array([[60.0, 0.0, 1.5], [0.0, 80.0, 1.5], [-70.0, 0.0, 1.5], [0.0, -90.0, 1.5]])
        )
        self.assertTrue(out_mu.diagnostics["mu_mimo_interference_enabled"])
        self.assertTrue(out_mu.diagnostics["spatial_consistency_enabled"])
        self.assertIsNotNone(out_mu.diagnostics["mean_inter_user_interference_db"])
        self.assertIsNotNone(out_mu.diagnostics["mu_mimo_sum_rate_bps_hz"])
        self.assertFalse(out_su.diagnostics["mu_mimo_interference_enabled"])
        self.assertGreater(out_mu.diagnostics["jain_fairness"], 0.0)

    # ------------------------------------------------------------------
    # Gap 4: End-to-End PyTorch Differentiable Wireless Link & RIS Autograd
    # ------------------------------------------------------------------
    def test_differentiable_wireless_link_trains_ris_and_constellation(self) -> None:
        torch.manual_seed(41)
        n_ris = 16
        rf = RFImpairmentChain(RFImpairmentConfig(pa_ibo_db=4.5, phase_noise_dbc_hz=-95.0, adc_bits=6), seed=41)
        ris = DifferentiableRISChannel(n_ris, coupling_scale=0.35, learnable_phases=True)
        link = DifferentiableWirelessLink(
            "16QAM",
            equalizer="mmse",
            learnable_constellation=True,
            rf_impairments=rf,
            ris_channel=ris,
        )

        # Deterministic weak direct link + strong RIS paths so optimizing RIS phases materially boosts BICM AIR
        h_d = ((torch.randn(1, 2, 4) + 1j * torch.randn(1, 2, 4)) * 0.1).to(torch.complex64).expand(8, -1, -1)
        h_r = ((torch.randn(1, 2, n_ris) + 1j * torch.randn(1, 2, n_ris)) / math.sqrt(2.0)).to(torch.complex64).expand(8, -1, -1)
        h_g = ((torch.randn(1, n_ris, 4) + 1j * torch.randn(1, n_ris, 4)) / math.sqrt(2.0)).to(torch.complex64).expand(8, -1, -1)
        bits = torch.randint(0, 2, (8, 32, 2, 4))

        opt = torch.optim.Adam(link.parameters(), lr=0.15)
        bce_history: list[float] = []
        for _ in range(12):
            opt.zero_grad()
            res = link(bits, h_direct=h_d, h_ris_rx=h_r, h_bs_ris=h_g, snr_db=6.0)
            loss = res["bce_loss"]
            loss.backward()
            if not bce_history:
                self.assertIsNotNone(ris.ris_phases.grad)
                assert ris.ris_phases.grad is not None
                self.assertGreater(float(ris.ris_phases.grad.abs().sum().item()), 0.0)
                assert link.mapper_demapper.constellation_delta is not None
                self.assertIsNotNone(link.mapper_demapper.constellation_delta.grad)
            opt.step()
            bce_history.append(float(loss.detach().item()))

        self.assertLess(bce_history[-1], bce_history[0])

    # ------------------------------------------------------------------
    # Gap 5: 3GPP Calibration Suite, O-RAN E2SM-KPM v3, Sealed Dossier & CLI
    # ------------------------------------------------------------------
    def test_3gpp_calibration_oran_kpm_dossier_and_cli(self) -> None:
        # 1. 7-check 3GPP/ITU Calibration Suite
        cal = run_3gpp_calibration_suite(seed=42, num_links=256)
        self.assertTrue(cal.passed, cal.format_report())
        self.assertEqual(cal.num_checks, 7)
        self.assertEqual(cal.num_passed, 7)

        # 2. O-RAN WG3 E2SM-KPM v3.0 Telemetry + E2SM-RC Control Loop
        spec, _ = compose_wce_spec("UMa 3.5 GHz multi-user scheduling")
        env = ComposedWirelessEnv(spec, batch_size=4, seed=51, enable_mu_mimo_interference=True)
        ric = ORANKPMTelemetryAdapter(env, cell_id="gNB-51-Cell-0")
        ind0 = ric.reset(seed=51)
        self.assertEqual(ind0.format_version, "E2SM-KPM-v03.00")
        self.assertEqual(len(ind0.ue_measurements), 4)
        self.assertIn("DRB.UEThpDl", ind0.ue_measurements[0])
        self.assertIn("L1.CQI", ind0.ue_measurements[0])

        ind1 = ric.step_indication(rc_control={"prb_allocation": [120, 60, 60, 33], "per_user_power_weights": [2.0, 1.0, 0.5, 0.5]})
        self.assertEqual(ind1.ue_measurements[0]["RRU.PrbUsedDl"], 120)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_p = Path(tmp)
            jsonl_p = ric.export_jsonl(tmp_p / "kpm.jsonl")
            self.assertTrue(jsonl_p.is_file())
            self.assertEqual(len(jsonl_p.read_text(encoding="utf-8").strip().splitlines()), 2)

            # 3. Verify Gate + Sealed Dossier + Regression Comparison + CLI
            dossier_dir = tmp_p / "dossier"
            ver = verify_wce_target(
                query="64-element passive RIS MIMO at 28 GHz",
                seeds=(0, 1, 2),
                phy_snr_db=14.0,
                run_3gpp_calibration=True,
                run_harq_amc=True,
                emit_dossier_dir=dossier_dir,
                signing_key="ci-secret-key",
            )
            self.assertTrue(ver.passed, ver.format_report())
            self.assertTrue((dossier_dir / "wce_dossier.json").is_file())
            self.assertTrue((dossier_dir / "wce_dossier.md").is_file())

            chk_ok = verify_wce_dossier_integrity(dossier_dir / "wce_dossier.json", signing_key="ci-secret-key")
            self.assertTrue(chk_ok["valid"], chk_ok)
            chk_bad_key = verify_wce_dossier_integrity(dossier_dir / "wce_dossier.json", signing_key="wrong-key")
            self.assertFalse(chk_bad_key["valid"])

            # Tamper dossier -> SHA-256 invalid
            raw_d = json.loads((dossier_dir / "wce_dossier.json").read_text(encoding="utf-8"))
            raw_d["overall_passed"] = False
            (dossier_dir / "wce_dossier.json").write_text(json.dumps(raw_d), encoding="utf-8")
            chk_tamper = verify_wce_dossier_integrity(dossier_dir / "wce_dossier.json")
            self.assertFalse(chk_tamper["valid"])

            # PR regression detector catches metric drop >5% or BLER jump >0.03
            base_p = tmp_p / "baseline.json"
            base_p.write_text(json.dumps(ver.to_dict()), encoding="utf-8")
            reg_same = compare_wce_verdicts(base_p, ver)
            self.assertTrue(reg_same["passed"])

            degraded = ver.to_dict()
            degraded["composed_rigor"]["seed_stats"]["mean"] *= 0.80
            reg_bad = compare_wce_verdicts(base_p, degraded, max_se_drop_pct=5.0)
            self.assertFalse(reg_bad["passed"])
            self.assertTrue(any("metric_drop" in r for r in reg_bad["regressions"]))

            # CLI `calibrate --json`
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = cli_main(["calibrate", "--num-links", "128", "--json"])
            self.assertEqual(rc, 0)
            cal_json = json.loads(buf.getvalue())
            self.assertTrue(cal_json["passed"])

            # Stand-alone build_wce_dossier check
            d_obj = build_wce_dossier(ver, signing_key="abc")
            self.assertTrue(verify_wce_dossier_integrity(d_obj, signing_key="abc")["valid"])


if __name__ == "__main__":
    unittest.main()
