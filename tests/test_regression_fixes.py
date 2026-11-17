"""Regression tests for physics, standards fidelity, rigor auditing, and UX fixes in wannado-wce."""

from __future__ import annotations

import platform
import sys
import unittest
import warnings

import numpy as np

from wannado.wce import (
    STANDARDS_PROVENANCE,
    AutonomousLinkDelayModel,
    ComposedWirelessEnv,
    FederatedWirelessRoundModel,
    audit_wce_rigor,
    audit_wce_spec,
    build_block_config,
    build_wce_archetype,
    check_tool_platform_readiness,
    compose_wce_spec,
    evaluate_link_delivery,
    evaluate_wce_suite,
    itur_p618_rain_attenuation_db,
    provision_awgn_link,
    provision_ckm_radiomap_bridge,
    provision_isac_jsac_channel,
    provision_nearfield_xlmimo_thz,
    provision_otfs_delay_doppler,
    simulate_federated_round,
    update_wce_spec_with_diff,
)
from wannado.wce.intake.spec import WCESpec
from wannado.wce.standards.gpp_38901 import compute_pathloss_tr38901_db
from wannado.wce.standards.gpp_extensions import compute_v2x_tr37885_pathloss


class TestWCERegressionFixes(unittest.TestCase):
    def test_otfs_index_put_conserves_multipath_energy(self) -> None:
        res = provision_otfs_delay_doppler(m_delay_bins=8, n_doppler_bins=4, num_paths=64, batch_size=512, seed=42)
        self.assertAlmostEqual(res.metrics["mean_dd_energy"], 1.0, delta=0.05)

    def test_tr38901_per_link_shadow_fading_std_mixed_and_uniform(self) -> None:
        d = np.array([30.0, 80.0, 150.0, 300.0])
        mixed = np.array([True, False, True, False])
        for scen, exp_los, exp_nlos in [
            ("UMa", 4.0, 6.0),
            ("UMi", 4.0, 7.82),
            ("RMa", 4.0, 8.0),
            ("InH", 3.0, 8.03),
            ("InF-SL", 4.3, 5.7),
        ]:
            _, sf_mix = compute_pathloss_tr38901_db(scen, d, 3.5, is_los=mixed)
            self.assertIsInstance(sf_mix, np.ndarray)
            np.testing.assert_allclose(sf_mix, np.array([exp_los, exp_nlos, exp_los, exp_nlos]))

            _, sf_all_los = compute_pathloss_tr38901_db(scen, d, 3.5, is_los=np.ones(4, dtype=bool))
            self.assertIsInstance(sf_all_los, float)
            self.assertAlmostEqual(float(sf_all_los), exp_los)

            _, sf_all_nlos = compute_pathloss_tr38901_db(scen, d, 3.5, is_los=np.zeros(4, dtype=bool))
            self.assertIsInstance(sf_all_nlos, float)
            self.assertAlmostEqual(float(sf_all_nlos), exp_nlos)

    def test_v2x_highway_nlos_distinct_from_los(self) -> None:
        d = np.array([20.0, 60.0, 120.0, 250.0])
        pl_los, sf_los = compute_v2x_tr37885_pathloss("highway", d, 5.9, state="LOS")
        pl_nlos, sf_nlos = compute_v2x_tr37885_pathloss("highway", d, 5.9, state="NLOS")
        self.assertFalse(np.allclose(pl_los, pl_nlos))
        self.assertTrue(np.all(pl_nlos >= pl_los))
        self.assertEqual(sf_los, 3.3)
        self.assertEqual(sf_nlos, 5.2)

    def test_isac_monostatic_vs_bistatic_round_trip_crb(self) -> None:
        mono = provision_isac_jsac_channel(topology="monostatic", target_range_m=50.0, bistatic_rx_range_m=50.0, seed=7)
        bi = provision_isac_jsac_channel(topology="bistatic", target_range_m=50.0, bistatic_rx_range_m=50.0, seed=7)
        # Monostatic two-way time-delay derivative dtau/dR = 2/c halves sigma_R at equal sensing SNR
        self.assertAlmostEqual(bi.metrics["range_rmse_crb_m"] / mono.metrics["range_rmse_crb_m"], 2.0, delta=1e-4)

    def test_composed_env_dynamic_ris_scaling_and_active_noise(self) -> None:
        spec_small, _ = compose_wce_spec("passive RIS at 28 GHz with 8 antennas")
        self.assertIn("passive_ris_cascade", spec_small.blocks_selected)
        self.assertNotIn("active_ris_flexible", spec_small.blocks_selected)
        spec_small.num_tx = 8
        spec_large = WCESpec.from_dict({**spec_small.to_dict(), "num_tx": 32})
        out_small = ComposedWirelessEnv(spec_small, batch_size=8, seed=1).reset()
        out_large = ComposedWirelessEnv(spec_large, batch_size=8, seed=1).reset()
        self.assertGreater(out_large.mean_snr_db, out_small.mean_snr_db + 2.0)

        spec_act, _ = compose_wce_spec("active RIS flexible antenna at 28 GHz")
        self.assertIn("active_ris_flexible", spec_act.blocks_selected)
        env_act = ComposedWirelessEnv(spec_act, batch_size=8, seed=1)
        out_act = env_act.reset()
        self.assertGreater(out_act.max_ris_modulus, 1.0)
        rigor_act = audit_wce_rigor([env_act.to_rigor_result(num_slots=6, seed=s) for s in (0, 1, 2)], wce_spec=spec_act, has_classic_baseline=True)
        self.assertNotIn("passive_ris_energy_conservation_violation", rigor_act.red_flags)

        spec_no_ris = WCESpec.from_dict({**spec_act.to_dict(), "blocks_selected": ["tr38901_uma_umi_rma"]})
        out_no_ris = ComposedWirelessEnv(spec_no_ris, batch_size=8, seed=1).reset()
        # Gain 4.0 alone would add +6.02 dB; dynamic amplified RIS noise keeps net boost strictly smaller
        snr_gain_db = out_act.mean_snr_db - out_no_ris.mean_snr_db
        self.assertGreater(snr_gain_db, 1.0)
        self.assertLess(snr_gain_db, 5.5)

    def test_ckm_radiomap_no_blas_runtime_warning(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            res = provision_ckm_radiomap_bridge(grid_size=24, seed=13)
            self.assertLess(res.metrics["ckm_reconstruction_rmse_db"], res.metrics["blind_statistical_rmse_db"])

    def test_evaluate_wce_suite_accepts_spec_runs_baselines_and_propagates_params(self) -> None:
        spec, _ = compose_wce_spec("28 GHz XL-MIMO near-field at 5 m with 64 antennas")
        cfg = build_block_config("nearfield_xlmimo_thz", wce_spec=spec)
        self.assertIsNotNone(cfg)
        assert cfg is not None
        self.assertEqual(cfg.num_tx_x, spec.num_tx)
        self.assertAlmostEqual(cfg.user_distance_m, spec.min_tx_rx_distance_m)

        report_spec = evaluate_wce_suite(spec, seeds=(0, 1, 2))
        executed_ids = {s.block_id for s in report_spec.block_summaries}
        self.assertIn("nearfield_xlmimo_thz", executed_ids)
        for base in spec.baseline_blocks:
            self.assertIn(base, executed_ids)

        # Single-seed run must flag single_seed_survivorship_risk by default (min_seeds=3)
        single = evaluate_wce_suite(["awgn_link"], baseline_blocks=["awgn_link"], seeds=(0,))
        assert single.block_summaries[0].audit is not None
        self.assertFalse(single.block_summaries[0].audit.passed)
        self.assertIn("single_seed_survivorship_risk", single.block_summaries[0].audit.red_flags)

    def test_rigor_auditor_skips_deterministic_primary_metric(self) -> None:
        nf_runs = [provision_nearfield_xlmimo_thz(seed=s) for s in (0, 1, 2)]
        rep_nf = audit_wce_rigor(nf_runs, has_classic_baseline=True)
        self.assertNotIn(rep_nf.primary_metric_name, {"aperture_m", "rayleigh_distance_m", "max_doppler_hz"})

        otfs_runs = [provision_otfs_delay_doppler(seed=s) for s in (0, 1, 2)]
        rep_otfs = audit_wce_rigor(otfs_runs, has_classic_baseline=True)
        self.assertNotEqual(rep_otfs.primary_metric_name, "max_doppler_hz")

    def test_provenance_diff_and_quadriga_readiness_notes(self) -> None:
        self.assertIn("Simplified Spectroscopic", STANDARDS_PROVENANCE["itur_p676"].section)

        awgn = provision_awgn_link(tx_power_dbm=43.0, seed=1)
        tx_prov = next(p for p in awgn.parameter_provenance if p.parameter == "tx_power_dbm")
        self.assertIn("43 dBm", tx_prov.rationale)
        self.assertNotEqual(tx_prov.rationale, "Typical 3GPP UE max transmit power (23 dBm).")

        spec, _ = compose_wce_spec("urban MIMO", problem_name="28 GHz Study")
        diff = update_wce_spec_with_diff(spec, "把频段改成 140 GHz")
        self.assertIn("140 GHz", diff.updated_spec.problem_name)

        q_ready = check_tool_platform_readiness("quadriga_lib")
        if platform.system() == "Darwin" or sys.version_info >= (3, 13):
            self.assertTrue(any("quadriga-lib" in w for w in q_ready["warnings"]))


class TestReleaseV112Regressions(unittest.TestCase):
    def test_gym_env_actions_and_dynamic_step_overrides(self) -> None:
        spec = build_wce_archetype("imt2030_isac_monostatic")
        env0 = ComposedWirelessEnv(spec, batch_size=4, seed=7)
        env2 = ComposedWirelessEnv(spec, batch_size=4, seed=7)
        g0 = env0.as_gym_env(max_steps=10)
        g2 = env2.as_gym_env(max_steps=10)
        g0.reset(seed=7)
        g2.reset(seed=7)
        obs0, rew0, _, _, info0 = g0.step(0)
        obs2, rew2, _, _, info2 = g2.step(2)
        self.assertGreater(info2["mean_snr_db"], info0["mean_snr_db"] + 3.0)
        self.assertGreater(obs2[0], obs0[0])
        self.assertNotAlmostEqual(rew0, rew2, places=3)

        # Runtime distance, 3D position, packet size, and rain overrides
        env = ComposedWirelessEnv(build_wce_archetype("uav_gis"), batch_size=2, seed=9)
        env.reset(seed=9)
        near = env.step(
            tx_pos_3d=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            rx_pos_3d=[[20.0, 0.0, 50.0], [20.0, 0.0, 50.0]],
            packet_bits=4000.0,
        )
        far_rain = env.step(
            tx_pos_3d=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            rx_pos_3d=[[1200.0, 0.0, 120.0], [1200.0, 0.0, 120.0]],
            packet_bits=5e7,
            rain_rate_mm_hr=60.0,
        )
        self.assertLess(far_rain.mean_snr_db, near.mean_snr_db - 10.0)
        self.assertGreater(far_rain.arrived_bits, near.arrived_bits)

    def test_composer_word_boundaries_starlink_lora_and_validation(self) -> None:
        # "federated learning", "Starlink", "radar" all contain substring "ar" -> must NOT trigger XR!
        spec_fl, _ = compose_wce_spec("federated learning over Starlink satellite with radar sensing")
        self.assertNotIn("traffic_xr_immersive", spec_fl.blocks_selected)
        self.assertIn("ntn_leo_satellite", spec_fl.blocks_selected)
        self.assertIn("semantic_aircomp_channel", spec_fl.blocks_selected)

        # Explicit "AR" word still enables immersive XR traffic
        spec_ar, _ = compose_wce_spec("60 fps AR headset streaming in urban cellular MIMO")
        self.assertIn("traffic_xr_immersive", spec_ar.blocks_selected)

        # LoRaWAN sub-GHz default tuning
        spec_lora, _ = compose_wce_spec("LoRaWAN sensor network in rural deployment")
        self.assertAlmostEqual(spec_lora.carrier_fc_ghz, 0.915)
        self.assertAlmostEqual(spec_lora.bandwidth_hz, 125e3)

        # Strict Stage 0 physical validation
        bad = build_wce_archetype("imt2030_isac_monostatic")
        bad.min_tx_rx_distance_m = -2.0
        bad.num_tx = 0
        bad.numerology_mu = 7
        bad.noise_figure_db = -1.0
        bad.tx_power_dbm = 95.0
        aud = audit_wce_spec(bad)
        self.assertFalse(aud.ok)
        self.assertGreaterEqual(len(aud.errors), 3)
        self.assertGreaterEqual(len(aud.warnings), 2)

        with self.assertRaises(ValueError):
            evaluate_wce_suite(["awgn_link"], seeds=())

    def test_latex_escaping_and_no_double_units(self) -> None:
        for arch in ("imt2030_isac_monostatic", "uav_gis", "federated_edge", "v2v_control"):
            spec = build_wce_archetype(arch)
            self.assertTrue(audit_wce_spec(spec).ok, msg=arch)
            tex1 = spec.format_system_parameter_table_tex()
            self.assertNotIn("MHz Hz", tex1)

        spec_comp, _ = compose_wce_spec(
            "28 GHz urban MIMO",
            problem_name="Study#1_v2^test~100%",
        )
        tex_comp = spec_comp.format_system_parameter_table_tex()
        self.assertNotIn("MHz Hz", tex_comp)
        self.assertIn("\\#", tex_comp)
        self.assertIn("\\^{}", tex_comp)
        self.assertIn("\\~{}", tex_comp)

        rep = evaluate_wce_suite(
            ["awgn_link"],
            baseline_blocks=["awgn_link"],
            seeds=(0, 1, 2),
            task_name="Run#A&B^2~v1",
            run_preflight=False,
        )
        res_tex = rep.to_results_latex()
        self.assertIn("Benchmark \\& Rigor Audit", res_tex)
        self.assertIn("\\#", res_tex)
        self.assertIn("\\&", res_tex)

    def test_rain_federated_and_autonomous_link_extensions(self) -> None:
        zero_rain = itur_p618_rain_attenuation_db(28.0, 1000.0, rain_rate_mm_hr=0.0)
        light_rain = itur_p618_rain_attenuation_db(28.0, 1000.0, rain_rate_mm_hr=10.0)
        heavy_rain = itur_p618_rain_attenuation_db(28.0, 1000.0, rain_rate_mm_hr=50.0)
        self.assertAlmostEqual(float(zero_rain), 0.0)
        self.assertGreater(float(heavy_rain), float(light_rain))
        self.assertGreater(float(light_rain), 0.1)

        fl_near = simulate_federated_round(
            num_clients=6,
            payload_bits_per_client=1e5,
            deadline_sec=0.5,
            client_distances_m=[15.0, 20.0, 25.0, 30.0, 40.0, 50.0],
            seed=3,
        )
        self.assertGreater(fl_near.participation_rate, 0.8)
        self.assertAlmostEqual(float(fl_near.normalized_client_weights.sum()), 1.0, places=5)

        fl_model = FederatedWirelessRoundModel(num_clients=4, payload_bits_per_client=5e8, deadline_sec=0.01, seed=4)
        fl_tight = fl_model.step_round(client_distances_m=[500.0, 600.0, 700.0, 800.0], rain_rate_mm_hr=40.0)
        self.assertGreater(fl_tight.straggler_count, 0)

        link_model = AutonomousLinkDelayModel(preset="v2v_control", latency_budget_ms=10.0, packet_bits=8000.0, seed=5)
        close_res = link_model.evaluate_step(tx_pos_3d=[0.0, 0.0, 1.5], rx_pos_3d=[15.0, 0.0, 1.5])
        far_res = evaluate_link_delivery(
            [0.0, 0.0, 1.5],
            [900.0, 0.0, 1.5],
            packet_bits=5e7,
            latency_budget_ms=5.0,
            preset="v2v_control",
            rain_rate_mm_hr=50.0,
            seed=5,
        )
        self.assertTrue(close_res.connected)
        self.assertTrue(close_res.within_budget)
        self.assertGreater(far_res.total_latency_ms, close_res.total_latency_ms)
        self.assertFalse(far_res.within_budget)


if __name__ == "__main__":
    unittest.main()
