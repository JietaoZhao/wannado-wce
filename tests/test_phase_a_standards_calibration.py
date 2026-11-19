"""Phase A standards-calibration & de-toyification regression tests for wannado-wce."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from wannado.wce import (
    AmbientIoTConfig,
    CDLTDLConfig,
    ComposedWirelessEnv,
    ISACConfig,
    NTNLEOConfig,
    TR38901MacroMicroConfig,
    UAVA2GConfig,
    WirelessSecurityConfig,
    audit_wce_rigor,
    compose_wce_spec,
    evaluate_wce_suite,
    provision_ambient_iot_backscatter,
    provision_isac_jsac_channel,
    provision_ntn_leo_satellite,
    provision_pls_covert_antijamming,
    provision_tr38901_cdl_tdl,
    provision_tr38901_uma_umi_rma,
    provision_uav_air_to_ground,
    update_wce_spec_with_diff,
)
from wannado.wce.blocks.domain_6g.pls_covert_antijamming import META as PLS_META
from wannado.wce.core.types import ProvisionResult
from wannado.wce.standards.etsi_ieee import itur_p676_attenuation_db_per_km
from wannado.wce.standards.gpp_38901 import compute_tr38901_uma_he
from wannado.wce.standards.gpp_extensions import compute_uav_tr36777_los_probability


class TestPhaseAStandardsCalibration(unittest.TestCase):
    def test_itur_p676_van_vleck_weisskopf_band_accuracy(self) -> None:
        g_3p5 = float(np.asarray(itur_p676_attenuation_db_per_km(3.5)).item())
        g_28 = float(np.asarray(itur_p676_attenuation_db_per_km(28.0)).item())
        g_60 = float(np.asarray(itur_p676_attenuation_db_per_km(60.0)).item())
        g_183 = float(np.asarray(itur_p676_attenuation_db_per_km(183.31)).item())

        self.assertLess(g_3p5, 0.02)
        self.assertLess(g_28, 0.25)
        self.assertGreater(g_60, 10.0)
        self.assertLess(g_60, 20.0)
        self.assertGreater(g_183, 20.0)

    def test_tr38901_los_mimo_rank1_dominance_cdl_kfactor_and_uma_he(self) -> None:
        # UMi forced short-range LoS MUST exhibit dominant Rank-1 spatial structure (sigma_1 >> sigma_2)
        umi_los = provision_tr38901_uma_umi_rma(
            TR38901MacroMicroConfig(scenario="UMi", num_links=64, num_tx=8, num_rx=4, d_2d_min_m=10.0, d_2d_max_m=15.0, o2i_fraction=0.0, seed=0)
        )
        self.assertGreater(float(umi_los.tensors["is_los"].float().mean().item()), 0.9)
        los_idx = int(np.where(umi_los.tensors["is_los"].numpy())[0][0])
        svals = np.linalg.svd(umi_los.tensors["h_small"][los_idx].numpy(), compute_uv=False)
        self.assertGreater(float(svals[0] / max(svals[1], 1e-6)), 2.0)

        # CDL-D (K=13.3 dB) & TDL-D maintain unit channel power & strong LoS dominance over CDL-A
        cdl_d = provision_tr38901_cdl_tdl(CDLTDLConfig(profile="CDL-D", batch_size=32, num_tx=8, num_rx=4, num_subcarriers=16, seed=0))
        self.assertAlmostEqual(cdl_d.metrics["mean_channel_power"], 1.0, delta=0.25)
        sv_d = np.linalg.svd(cdl_d.tensors["h_freq"][0, :, :, 0, 0].numpy(), compute_uv=False)
        self.assertGreater(float(sv_d[0] / max(sv_d[1], 1e-6)), 3.0)

        # UMa effective height h_E stays 1.0 at 1.5m and scales > 1.0 at >= 15m UT height
        d2 = np.array([100.0, 300.0])
        np.testing.assert_allclose(compute_tr38901_uma_he(d2, 1.5), np.array([1.0, 1.0]))
        self.assertTrue(bool(np.all(compute_tr38901_uma_he(d2, 18.0) > 1.0)))

    def test_tr36777_uav_and_tr38811_ntn_band_calibration(self) -> None:
        d = np.array([50.0, 200.0, 500.0])
        p_hi = compute_uav_tr36777_los_probability("UMa-AV", d, 150.0)
        p_mid = compute_uav_tr36777_los_probability("UMa_AV", d, 60.0)
        p_rma_hi = compute_uav_tr36777_los_probability("RMa-AV", d, 60.0)
        np.testing.assert_allclose(p_hi, np.ones_like(d))
        np.testing.assert_allclose(p_rma_hi, np.ones_like(d))
        self.assertTrue(bool(np.all(p_mid <= 1.0) and np.all(p_mid > 0.5)))

        uav = provision_uav_air_to_ground(UAVA2GConfig(model_mode="tr36777", tr36777_scenario="UMa-AV", uav_altitude_m=150.0, seed=0))
        self.assertAlmostEqual(uav.metrics["mean_los_prob"], 1.0, delta=1e-6)

        ntn_s = provision_ntn_leo_satellite(NTNLEOConfig(fc_ghz=2.0, elevation_deg=20.0, env_type="urban", seed=0))
        ntn_ka = provision_ntn_leo_satellite(NTNLEOConfig(fc_ghz=20.0, elevation_deg=20.0, env_type="urban", seed=0))
        self.assertGreater(ntn_ka.metrics["mean_clutter_loss_db"], ntn_s.metrics["mean_clutter_loss_db"])
        self.assertGreater(ntn_ka.metrics["atmospheric_gas_loss_db"], ntn_s.metrics["atmospheric_gas_loss_db"] * 5.0)

    def test_isac_stochastic_variance_and_rigor_zero_std_trap(self) -> None:
        isac_runs = [provision_isac_jsac_channel(ISACConfig(seed=s)) for s in (0, 1, 2)]
        comm_vals = [r.metrics["comm_spectral_efficiency_bps_hz"] for r in isac_runs]
        crb_vals = [r.metrics["range_rmse_crb_m"] for r in isac_runs]
        self.assertGreater(float(np.std(comm_vals)), 1e-4)
        self.assertGreater(float(np.std(crb_vals)), 1e-4)
        rep = audit_wce_rigor(isac_runs, has_classic_baseline=True, min_seeds=3)
        self.assertTrue(rep.passed, msg=f"ISAC failed audit: {rep.red_flags}")

        # RigorAuditor catches artificial zero-variance duplicates on stochastic metric names
        fake_dup = [
            ProvisionResult(block_id="dup_block", seed=s, elapsed_seconds=0.01, config={}, metrics={"comm_spectral_efficiency_bps_hz": 5.5, "mean_channel_power": 1.0})
            for s in (0, 1, 2)
        ]
        bad_rep = audit_wce_rigor(fake_dup, has_classic_baseline=True, min_seeds=3)
        self.assertFalse(bad_rep.passed)
        self.assertIn("zero_variance_deterministic_metric", bad_rep.red_flags)

    def test_composer_ris_separation_table1_dedup_and_composed_env_fidelity(self) -> None:
        spec_ris, _ = compose_wce_spec("64-element RIS 2-bit MIMO at 28 GHz")
        self.assertEqual(spec_ris.num_tx, 32)
        self.assertEqual(spec_ris.num_ris_elements, 64)
        self.assertEqual(spec_ris.ris_phase_bits, 2)

        spec_xl, _ = compose_wce_spec("256x4 MIMO at 28 GHz distance 10 m")
        self.assertEqual(spec_xl.num_tx, 256)
        self.assertEqual(spec_xl.num_rx, 4)

        rep = evaluate_wce_suite(spec_ris, seeds=(0, 1, 2))
        md = rep.format_environment_provenance_md()
        bw_rows = [line for line in md.splitlines() if line.startswith("| `bandwidth_hz`")]
        ris_rows = [line for line in md.splitlines() if line.startswith("| `num_ris_elements`")]
        self.assertEqual(len(bw_rows), 1)
        self.assertEqual(len(ris_rows), 1)
        self.assertIn("| `64` |", ris_rows[0])

        # Composed NTN uses orbital slant distance (> 500 km), not terrestrial 150 m
        spec_ntn, _ = compose_wce_spec("LEO NTN satellite downlink at 20 GHz Ka-band elevation 30 deg")
        step_ntn = ComposedWirelessEnv(spec_ntn, batch_size=16, seed=0).step()
        self.assertGreater(step_ntn.diagnostics["mean_distance_m"], 500_000.0)
        self.assertGreater(step_ntn.mean_snr_db, -15.0)

        # Uncapped XL-MIMO tensor dimensions (256 TX)
        step_xl = ComposedWirelessEnv(spec_xl, batch_size=4, seed=0).step()
        self.assertEqual(step_xl.h_small.shape[-1], 256)

        # Causal CSI ordering in ComposedWirelessEnv matrix beamforming
        sp_ideal = update_wce_spec_with_diff(spec_ris, {"csi_assumption": "ideal_genie_baseline"}).updated_spec
        sp_delay = update_wce_spec_with_diff(spec_ris, {"csi_assumption": "delayed_feedback", "max_speed_mps": 60.0}).updated_spec
        se_ideal = ComposedWirelessEnv(sp_ideal, batch_size=16, seed=1).to_rigor_result(num_slots=10, seed=1).metrics["mean_spectral_efficiency_bps_hz"]
        se_real = ComposedWirelessEnv(spec_ris, batch_size=16, seed=1).to_rigor_result(num_slots=10, seed=1).metrics["mean_spectral_efficiency_bps_hz"]
        se_delay = ComposedWirelessEnv(sp_delay, batch_size=16, seed=1).to_rigor_result(num_slots=10, seed=1).metrics["mean_spectral_efficiency_bps_hz"]
        self.assertGreater(se_ideal, se_real)
        self.assertGreater(se_real, se_delay)

        # Gym adapter exact max_steps and arbitrary-batch stateless ChannelModuleAdapter
        env = ComposedWirelessEnv(spec_ris, batch_size=8, seed=0)
        gym_env = env.as_gym_env(max_steps=20)
        gym_env.reset(seed=0)
        steps = 0
        trunc = False
        while not trunc:
            _, _, term, trunc, _ = gym_env.step(1)
            steps += 1
            self.assertFalse(term)
        self.assertEqual(steps, 20)

        ch_mod = env.as_channel_module()
        q_before = float(env._queue_bits.sum())
        x = (torch.randn(64, spec_ris.num_tx) + 1j * torch.randn(64, spec_ris.num_tx)).to(torch.complex64).requires_grad_(True)
        y = ch_mod(x)
        self.assertEqual(tuple(y.shape), (64, spec_ris.num_rx))
        torch.mean(y.abs().pow(2)).backward()
        self.assertIsNotNone(x.grad)
        self.assertAlmostEqual(float(env._queue_bits.sum()), q_before)

    def test_ambient_iot_topology2_asymmetry_and_pls_hook_names(self) -> None:
        aiot_short = provision_ambient_iot_backscatter(AmbientIoTConfig(topology="topology_2", helper_distance_m=8.0, direct_distance_m=40.0, seed=3))
        aiot_far = provision_ambient_iot_backscatter(AmbientIoTConfig(topology="topology_2", helper_distance_m=8.0, direct_distance_m=200.0, seed=3))
        self.assertAlmostEqual(aiot_short.metrics["mean_harvested_uw"], aiot_far.metrics["mean_harvested_uw"], delta=1e-6)
        self.assertGreater(aiot_short.metrics["median_reader_snr_db"], aiot_far.metrics["median_reader_snr_db"] + 10.0)

        pls = provision_pls_covert_antijamming(WirelessSecurityConfig(seed=0))
        self.assertIn("iq_clean", pls.tensors)
        self.assertIn("iq_jammed", pls.tensors)
        self.assertIn("iq_clean", PLS_META.extension_hooks[0])
        self.assertIn("iq_jammed", PLS_META.extension_hooks[0])


if __name__ == "__main__":
    unittest.main()
