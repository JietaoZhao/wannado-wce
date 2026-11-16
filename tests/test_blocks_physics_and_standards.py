"""Physics, standards-fidelity, and conservation-law unit tests across ALL 33 blocks in wannado-wce."""

from __future__ import annotations

import importlib
import unittest

import numpy as np
import torch

from wannado.wce.blocks import BLOCK_ID_TO_RUNNER
from wannado.wce.core.units import rayleigh_distance_m


class TestAll33BlocksPhysics(unittest.TestCase):
    def test_every_registered_block_runs_and_emits_finite_provenance(self) -> None:
        self.assertEqual(len(BLOCK_ID_TO_RUNNER), 33)
        for bid, (mod_path, fn_name) in BLOCK_ID_TO_RUNNER.items():
            mod = importlib.import_module(mod_path)
            fn = getattr(mod, fn_name)
            res = fn(seed=42)
            self.assertEqual(res.block_id, bid)
            self.assertGreater(len(res.metrics), 0)
            self.assertGreater(len(res.parameter_provenance), 0)
            self.assertGreater(len(res.mechanisms_used) + len(res.standard_ids_used), 0)
            for mk, mv in res.metrics.items():
                self.assertTrue(np.isfinite(mv), f"{bid}.{mk} returned non-finite {mv}")
            for tk, tv in res.tensors.items():
                if isinstance(tv, torch.Tensor):
                    self.assertTrue(
                        torch.isfinite(tv.real).all().item(),
                        f"{bid}.{tk} contains non-finite tensor elements",
                    )

    def test_family1_classic_fading_and_ppp_invariants(self) -> None:
        from wannado.wce.blocks.channel_classic import (
            provision_awgn_link,
            provision_nakagami_m,
            provision_ppp_stochastic_geo,
            provision_rayleigh_clarke,
            provision_rician_los,
            provision_saleh_valenzuela,
        )

        # AWGN Shannon & noise power scaling
        awgn = provision_awgn_link(num_symbols=8192, tx_power_dbm=30.0, pathloss_db=100.0, seed=7)
        self.assertAlmostEqual(awgn.metrics["mean_channel_power"], 1.0, delta=0.1)
        self.assertGreater(awgn.metrics["shannon_capacity_mbps"], 1.0)

        # Rayleigh Clarke J0 correlation & unit power E[|h|^2]=1
        ray = provision_rayleigh_clarke(batch_size=256, num_steps=24, speed_mps=15.0, fc_ghz=3.5, seed=7)
        self.assertAlmostEqual(ray.metrics["mean_channel_power"], 1.0, delta=0.12)
        self.assertLess(ray.metrics["autocorr_abs_error"], 0.15)

        # Rician K=15 dB increases LoS power fraction over K=-10 dB
        ric_lo = provision_rician_los(k_factor_db=-10.0, batch_size=64, seed=7)
        ric_hi = provision_rician_los(k_factor_db=15.0, batch_size=64, seed=7)
        self.assertAlmostEqual(ric_hi.metrics["mean_channel_power"], 1.0, delta=0.1)
        self.assertGreater(ric_hi.metrics["los_power_fraction"], ric_lo.metrics["los_power_fraction"])

        # Nakagami m > 1 reduces Amount of Fading (AF = 1/m)
        nak_sev = provision_nakagami_m(m_shape=0.75, batch_size=128, num_links=64, seed=7)
        nak_mild = provision_nakagami_m(m_shape=4.0, batch_size=128, num_links=64, seed=7)
        self.assertLess(nak_mild.metrics["empirical_amount_of_fading"], nak_sev.metrics["empirical_amount_of_fading"])

        # Saleh-Valenzuela normalized Frobenius power E[||H||_F^2] = Nt * Nr
        sv = provision_saleh_valenzuela(batch_size=32, num_tx=32, num_rx=4, seed=7)
        self.assertEqual(tuple(sv.tensors["h"].shape), (32, 4, 32))
        self.assertAlmostEqual(sv.metrics["mean_frobenius_sq"], sv.metrics["target_frobenius_sq"], delta=20.0)

        # PPP cellular coverage matches Andrews (2011) analytical curve
        ppp = provision_ppp_stochastic_geo(num_realizations=1500, pathloss_exponent=4.0, sinr_threshold_db=0.0, seed=7)
        self.assertLess(ppp.metrics["coverage_theory_gap"], 0.06)

    def test_family2_3gpp_standards_invariants(self) -> None:
        from wannado.wce.blocks.channel_3gpp import (
            provision_ntn_leo_satellite,
            provision_tr38901_cdl_tdl,
            provision_tr38901_inf,
            provision_tr38901_uma_umi_rma,
            provision_uav_air_to_ground,
            provision_v2x_vehicular_sl,
        )

        # TR 38.901 UMa O2I increases pathloss
        uma_out = provision_tr38901_uma_umi_rma(scenario="UMa", o2i_fraction=0.0, seed=5)
        uma_o2i = provision_tr38901_uma_umi_rma(scenario="UMa", o2i_fraction=1.0, seed=5)
        self.assertGreater(uma_o2i.metrics["mean_pathloss_db"], uma_out.metrics["mean_pathloss_db"] + 15.0)

        # InF clutter density increases NLOS attenuation / reduces LoS probability
        inf_s = provision_tr38901_inf(scenario="InF-SL", seed=5)
        inf_d = provision_tr38901_inf(scenario="InF-DH", seed=5)
        self.assertLessEqual(inf_d.metrics["los_fraction"], inf_s.metrics["los_fraction"] + 0.05)

        # CDL / TDL generates normalized delay-tap and MIMO frequency responses
        cdl = provision_tr38901_cdl_tdl(profile="CDL-C", batch_size=16, num_tx=8, num_rx=2, num_subcarriers=32, num_slots=4, seed=5)
        self.assertEqual(tuple(cdl.tensors["h_freq"].shape), (16, 2, 8, 32, 4))
        self.assertAlmostEqual(cdl.metrics["mean_channel_power"], 1.0, delta=0.2)

        # NTN LEO higher elevation increases median SNR & shrinks slant distance
        ntn_lo = provision_ntn_leo_satellite(elevation_deg=15.0, fc_ghz=2.0, seed=5)
        ntn_hi = provision_ntn_leo_satellite(elevation_deg=75.0, fc_ghz=2.0, seed=5)
        self.assertLess(ntn_hi.metrics["slant_range_km"], ntn_lo.metrics["slant_range_km"])
        self.assertGreater(ntn_hi.metrics["median_snr_db"], ntn_lo.metrics["median_snr_db"])

        # UAV LoS probability increases with altitude
        uav_low = provision_uav_air_to_ground(uav_altitude_m=20.0, tr36777_scenario="UMa-AV", seed=5)
        uav_hi = provision_uav_air_to_ground(uav_altitude_m=200.0, tr36777_scenario="UMa-AV", seed=5)
        self.assertGreater(uav_hi.metrics["mean_los_prob"], uav_low.metrics["mean_los_prob"])

        # V2X highway relative speed triggers > 1 kHz dual-mobility Doppler
        v2x = provision_v2x_vehicular_sl(scenario="highway", fc_ghz=5.9, speed_tx_kmh=110.0, speed_rx_kmh=-110.0, seed=5)
        self.assertGreater(v2x.metrics["dual_mobility_doppler_hz"], 1000.0)

    def test_family3_arrays_ris_nearfield_conservation_invariants(self) -> None:
        from wannado.wce.blocks.arrays_and_ris import (
            provision_active_ris_flexible,
            provision_farfield_upa,
            provision_nearfield_xlmimo_thz,
            provision_passive_ris_cascade,
            provision_star_ris,
        )

        upa = provision_farfield_upa(tx_nx=8, tx_ny=4, num_rx=1, batch_size=16, seed=3)
        self.assertEqual(int(upa.metrics["num_tx"]), 32)
        self.assertAlmostEqual(upa.metrics["mean_channel_power"], 1.0, delta=0.2)

        # Near-field Rayleigh distance > min distance triggers near-field regime + THz gas attenuation
        nf = provision_nearfield_xlmimo_thz(num_tx_x=128, num_tx_y=1, fc_ghz=100.0, user_distance_m=3.0, seed=3)
        ff = provision_nearfield_xlmimo_thz(num_tx_x=128, num_tx_y=1, fc_ghz=100.0, user_distance_m=200.0, seed=3)
        self.assertGreater(nf.metrics["rayleigh_distance_m"], 20.0)
        self.assertGreater(rayleigh_distance_m(0.19, 100e9), 20.0)
        self.assertGreater(nf.metrics["molecular_absorption_db_per_km"], 0.1)
        self.assertEqual(nf.metrics["is_in_nearfield"], 1.0)
        self.assertGreater(ff.metrics["farfield_beam_alignment_efficiency"], nf.metrics["farfield_beam_alignment_efficiency"] + 0.15)
        self.assertGreater(ff.metrics["farfield_beam_alignment_efficiency"], 0.90)

        # Passive RIS |phi_m| <= 1 and aligned phase strictly beats random phase
        pris = provision_passive_ris_cascade(num_ris_elements=64, batch_size=24, seed=3)
        self.assertLessEqual(pris.metrics["max_ris_element_modulus"], 1.0 + 1e-6)
        self.assertGreater(pris.metrics["gain_aligned_ris"], pris.metrics["gain_random_ris"])

        # STAR-RIS energy splitting conservation: |beta_t|^2 + |beta_r|^2 <= 1
        star = provision_star_ris(num_star_elements=48, batch_size=24, seed=3)
        self.assertLessEqual(star.metrics["max_element_energy_sum"], 1.0 + 1e-6)

        # Active RIS amplification with thermal noise beats passive when overcoming double fading
        aris = provision_active_ris_flexible(num_ris_elements=64, seed=3)
        self.assertGreater(aris.metrics["active_ris_effective_snr"], aris.metrics["passive_ris_equivalent_snr"])

    def test_family4_traffic_and_mac_queue_loop(self) -> None:
        from wannado.wce.blocks.traffic_and_mac import (
            provision_multicell_interference_mac,
            provision_traffic_ftp_models,
            provision_traffic_poisson_pareto_voip,
            provision_traffic_xr_immersive,
        )

        ftp = provision_traffic_ftp_models(num_users=20, num_slots=100, seed=9)
        self.assertGreater(ftp.metrics["offered_traffic_mbps"], 0.0)

        ppv = provision_traffic_poisson_pareto_voip(seed=9)
        self.assertGreater(ppv.metrics["offered_traffic_kbps"], 0.0)

        xr = provision_traffic_xr_immersive(fps=60.0, pdb_ms=10.0, seed=9)
        self.assertLessEqual(xr.metrics["mean_packet_satisfaction"], 1.0)
        self.assertGreater(xr.metrics["total_offered_mbps"], 0.1)

        mac = provision_multicell_interference_mac(num_cells=4, users_per_cell=4, episode_horizon=40, seed=9)
        self.assertGreater(mac.metrics["pf_mean_throughput_mbps"], 0.0)
        self.assertGreaterEqual(mac.metrics["pf_mean_throughput_mbps"], mac.metrics["rr_mean_throughput_mbps"] * 0.9)

    def test_family5_domain6g_invariants(self) -> None:
        from wannado.wce.blocks.domain_6g import (
            provision_ambient_iot_backscatter,
            provision_hrllc_fbl_urllc,
            provision_isac_jsac_channel,
            provision_otfs_delay_doppler,
            provision_pls_covert_antijamming,
            provision_semantic_aircomp_channel,
            provision_wifi_sensing_80211bf,
        )

        isac = provision_isac_jsac_channel(comm_power_split=0.5, seed=11)
        self.assertGreater(isac.metrics["comm_spectral_efficiency_bps_hz"], 0.0)
        self.assertGreater(isac.metrics["range_rmse_crb_m"], 0.0)

        wifi = provision_wifi_sensing_80211bf(seed=11)
        self.assertGreater(wifi.metrics["csi_temporal_variance"], 0.0)

        otfs = provision_otfs_delay_doppler(m_delay_bins=32, n_doppler_bins=16, seed=11)
        self.assertGreater(otfs.metrics["mean_dd_energy"], 0.0)
        self.assertGreater(otfs.metrics["max_doppler_hz"], 0.0)

        # Polyanskiy FBL achievable rate MUST stay strictly below infinite-blocklength Shannon capacity
        fbl = provision_hrllc_fbl_urllc(snr_db=10.0, blocklength_n=200, target_bler=1e-5, seed=11)
        self.assertLess(fbl.metrics["fbl_achievable_rate_bps_hz"], fbl.metrics["shannon_capacity_bps_hz"])
        self.assertGreater(fbl.metrics["shannon_overestimation_pct"], 0.0)

        iot = provision_ambient_iot_backscatter(num_devices=32, seed=11)
        self.assertGreaterEqual(iot.metrics["device_activation_rate"], 0.0)
        self.assertGreater(iot.metrics["mean_harvested_uw"], 0.0)

        # Semantic & AirComp: verify autograd differentiability!
        sem = provision_semantic_aircomp_channel(num_edge_devices=12, batch_size=32, seed=11)
        self.assertEqual(sem.metrics["autograd_gradient_valid"], 1.0)

        # PLS, Covert & JSR-dB jamming
        pls = provision_pls_covert_antijamming(jsr_db=5.0, batch_size=64, seed=11)
        self.assertGreaterEqual(pls.metrics["mean_secrecy_rate_bps_hz"], 0.0)
        self.assertAlmostEqual(pls.metrics["empirical_jsr_db"], 5.0, delta=2.5)

    def test_family6_baselines_invariants(self) -> None:
        from wannado.wce.blocks.baselines import (
            provision_ckm_radiomap_bridge,
            provision_ls_lmmse_csi,
            provision_proportional_fair,
            provision_water_filling,
            provision_wmmse_precoder,
        )

        # Water-filling strictly >= Equal Power Allocation
        wf = provision_water_filling(snr_db=3.0, num_tx=8, num_rx=4, batch_size=64, seed=13)
        self.assertGreaterEqual(wf.metrics["water_filling_rate_bps_hz"], wf.metrics["equal_power_rate_bps_hz"] - 1e-6)

        # WMMSE beats MRT and achieves >= ZF (or ties ZF at high SNR)
        wmmse = provision_wmmse_precoder(batch_size=24, num_tx=8, num_users=4, snr_db=15.0, num_iters=25, seed=13)
        self.assertGreater(wmmse.metrics["wmmse_sum_rate_bps_hz"], wmmse.metrics["mrt_sum_rate_bps_hz"])
        self.assertGreaterEqual(wmmse.metrics["wmmse_sum_rate_bps_hz"], wmmse.metrics["zf_sum_rate_bps_hz"] - 0.1)

        # LMMSE NMSE strictly < LS NMSE when channel has spatial correlation & noise
        csi = provision_ls_lmmse_csi(pilot_snr_db=5.0, corr_rho=0.8, batch_size=64, seed=13)
        self.assertLess(csi.metrics["lmmse_nmse_db"], csi.metrics["ls_nmse_db"])

        # PF achieves higher Jain fairness than Max-SNR
        pf = provision_proportional_fair(num_users=12, num_slots=200, seed=13)
        self.assertGreater(pf.metrics["pf_jains_fairness"], pf.metrics["max_snr_jains_fairness"])

        # CKM reconstruction beats location-blind statistical baseline
        ckm = provision_ckm_radiomap_bridge(grid_size=24, seed=13)
        self.assertLess(ckm.metrics["ckm_reconstruction_rmse_db"], ckm.metrics["blind_statistical_rmse_db"])


if __name__ == "__main__":
    unittest.main()
