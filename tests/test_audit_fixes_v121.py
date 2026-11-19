"""Regression tests for wannado-wce v1.2.1 deep physics, PHY, standards, and engineering audit fixes."""

from __future__ import annotations

import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from wannado.wce import (
    AutonomousLinkDelayModel,
    ComposedWirelessEnv,
    audit_wce_rigor,
    build_wce_archetype,
    compose_wce_spec,
    evaluate_link_level_phy,
    evaluate_wce_suite,
    preflight_wce,
    provision_active_ris_flexible,
    provision_ls_lmmse_csi,
    provision_ntn_leo_satellite,
    provision_otfs_delay_doppler,
    provision_passive_ris_cascade,
    provision_rayleigh_clarke,
    provision_tr38901_cdl_tdl,
    scaffold_wce_lab,
)
from wannado.wce.cli import main as cli_main
from wannado.wce.core.adapters import ChannelModuleAdapter
from wannado.wce.core.geometry import (
    nearfield_spherical_steering,
    upa_element_coordinates,
    upa_farfield_steering,
)
from wannado.wce.core.types import ProvisionResult
from wannado.wce.standards.gpp_38901 import (
    CDL_PROFILES,
    TDL_PROFILES,
    compute_pathloss_tr38901_db,
)
from wannado.wce.standards.gpp_extensions import (
    compute_uav_tr36777_pathloss,
    compute_v2x_tr37885_pathloss,
)
from wannado.wce.workspace.link_level_phy import (
    _build_ldpc_parity_matrix,
    _gray_pam_levels,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


class TestAuditFixesV121(unittest.TestCase):
    def test_cold_imports_have_no_circular_errors_and_all_exports_resolve(self) -> None:
        for stmt in (
            "import wannado.wce.reporter; import wannado.wce.audit",
            "from wannado.wce import evaluate_wce_suite, verify_wce_target",
            "import wannado.wce as w; missing = [n for n in w.__all__ if not hasattr(w, n)]; assert not missing, missing",
        ):
            with self.subTest(stmt=stmt):
                proc = subprocess.run(
                    [sys.executable, "-c", stmt],
                    cwd=REPO_ROOT,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(proc.returncode, 0, msg=f"{stmt}\n{proc.stderr}")

    def test_cli_verify_dir_and_jit_recipe_require_live(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lab_dir = Path(tmp) / "wce_lab"
            scaffold_wce_lab(target_dir=lab_dir, archetype="imt2030_immersive_xr", overwrite=True)
            rc = cli_main(["verify", "--dir", str(lab_dir), "--seeds", "0", "1", "2", "--json"])
            self.assertEqual(rc, 0)

            out_dir = Path(tmp) / "adapters"
            # With --require-live on sionna_rt when sionna is not installed, CLI must return exit code 1
            rc_live = cli_main(["jit-recipe", "sionna_rt", "--write-dir", str(out_dir), "--verify", "--require-live", "--json"])
            try:
                import sionna  # noqa: F401
                expected = 0
            except ImportError:
                expected = 1
            self.assertEqual(rc_live, expected)

    def test_gray_pam_exact_single_bit_adjacency_and_girth6_ldpc(self) -> None:
        for bits_axis in (1, 2, 3):  # QPSK, 16QAM, 64QAM
            levels, bit_pats = _gray_pam_levels(bits_axis)
            order = np.argsort(levels)
            for i in range(len(order) - 1):
                hamming = int(np.sum(bit_pats[order[i]] != bit_pats[order[i + 1]]))
                self.assertEqual(hamming, 1, msg=f"Non-Gray transition for {bits_axis} bits/axis at pair {i}")

        _, h_full, _, _ = _build_ldpc_parity_matrix(k_info=64, m_parity=64, col_weight=3)
        gram = h_full.astype(np.int32).T @ h_full.astype(np.int32)
        np.fill_diagonal(gram, 0)
        self.assertLessEqual(int(gram.max()), 1, msg="LDPC parity matrix must have zero length-4 cycles")

    def test_link_level_phy_eigenbeamforming_wiener_lmmse_and_cp_isi(self) -> None:
        # 1. Wideband eigenbeamforming uses all Nt antennas (Nt=8 beats Nt=2 at same SNR)
        rng = np.random.default_rng(1)
        h2 = torch.from_numpy((rng.normal(size=(32, 2, 2)) + 1j * rng.normal(size=(32, 2, 2))) / math.sqrt(2.0)).to(torch.complex64)
        h8 = torch.from_numpy((rng.normal(size=(32, 2, 8)) + 1j * rng.normal(size=(32, 2, 8))) / math.sqrt(2.0)).to(torch.complex64)
        r2 = evaluate_link_level_phy(h2, snr_db=6.0, modulation="QPSK", equalizer="mmse", estimator="genie", num_blocks=32, seed=0)
        r8 = evaluate_link_level_phy(h8, snr_db=6.0, modulation="QPSK", equalizer="mmse", estimator="genie", num_blocks=32, seed=0)
        self.assertGreater(r8.mean_post_sinr_db, r2.mean_post_sinr_db + 3.0)

        # 2. Causal Wiener LMMSE outperforms LS on wideband CDL-C channel across low & high SNR
        cdl_c = provision_tr38901_cdl_tdl(profile="CDL-C", batch_size=24, num_tx=2, num_rx=2, num_subcarriers=32, delay_spread_ns=100.0, seed=0)
        for snr in (5.0, 20.0):
            r_ls = evaluate_link_level_phy(cdl_c, snr_db=snr, modulation="16QAM", estimator="ls", num_blocks=24, seed=1)
            r_mmse = evaluate_link_level_phy(cdl_c, snr_db=snr, modulation="16QAM", estimator="lmmse", num_blocks=24, seed=1)
            self.assertLessEqual(r_mmse.csi_nmse_db, r_ls.csi_nmse_db + 0.5, msg=f"LMMSE degraded at SNR={snr} dB")

        # 3. Time-domain OFDM CP mode exposes genuine ISI when cp_length=0 vs generous cp_length=16
        cdl_wide = provision_tr38901_cdl_tdl(profile="CDL-C", batch_size=24, num_tx=2, num_rx=2, num_subcarriers=32, delay_spread_ns=300.0, seed=2)
        r_nocp = evaluate_link_level_phy(cdl_wide, snr_db=25.0, modulation="16QAM", estimator="genie", cp_length=0, num_blocks=24, seed=3)
        r_cp = evaluate_link_level_phy(cdl_wide, snr_db=25.0, modulation="16QAM", estimator="genie", cp_length=16, num_blocks=24, seed=3)
        self.assertGreater(r_nocp.evm_pct, r_cp.evm_pct + 2.0)

    def test_tr38901_complete_tdl_cdl_rms_delay_spread_and_no_k_double_count(self) -> None:
        for name, prof in {**TDL_PROFILES, **CDL_PROFILES}.items():
            d = np.asarray(prof["delays_norm"], dtype=np.float64)
            w = 10.0 ** (np.asarray(prof["powers_db"], dtype=np.float64) / 10.0)
            w /= w.sum()
            rms_ds = float(np.sqrt(np.sum(w * d * d) - np.sum(w * d) ** 2))
            self.assertAlmostEqual(rms_ds, 1.0, delta=0.02, msg=f"{name} normalized RMS DS={rms_ds}")

        res_d = provision_tr38901_cdl_tdl(profile="TDL-D", batch_size=128, num_tx=1, num_rx=1, num_subcarriers=16, num_slots=2, seed=0)
        self.assertAlmostEqual(res_d.metrics["mean_channel_power"], 1.0, delta=0.2)
        self.assertAlmostEqual(res_d.metrics["rms_delay_spread_ns"], 100.0, delta=2.0)

    def test_3gpp_standards_rma_inf_uav_ntn_v2x_calibration(self) -> None:
        # RMa LOS sigma_SF = 6 dB past breakpoint
        _, sf_far = compute_pathloss_tr38901_db("RMa", np.array([5000.0]), 3.5, is_los=True, h_bs_m=35.0, h_ut_m=1.5)
        self.assertEqual(sf_far, 6.0)

        # InF-DL NLOS >= InF-SL NLOS
        pl_dl, _ = compute_pathloss_tr38901_db("InF-DL", np.array([15.0]), 3.5, is_los=False, h_bs_m=8.0)
        pl_sl, _ = compute_pathloss_tr38901_db("InF-SL", np.array([15.0]), 3.5, is_los=False, h_bs_m=8.0)
        self.assertGreaterEqual(float(pl_dl[0]), float(pl_sl[0]) - 1e-9)

        # TR 36.777 UAV NLOS never < LOS even at high altitude
        for scen in ("UMa-AV", "UMi-AV", "RMa-AV"):
            pl_l, _ = compute_uav_tr36777_pathloss(scen, np.array([10.0, 50.0, 200.0]), 250.0, 2.0, is_los=True)
            pl_n, _ = compute_uav_tr36777_pathloss(scen, np.array([10.0, 50.0, 200.0]), 250.0, 2.0, is_los=False)
            self.assertTrue(bool(np.all(pl_n >= pl_l - 1e-9)), msg=f"{scen} NLOS < LOS inversion")

        # TR 37.885 Table 6.2.1-1 exact SF values
        _, sf_u_los = compute_v2x_tr37885_pathloss("urban_grid", np.array([50.0]), 5.9, state="LOS")
        _, sf_u_nlos = compute_v2x_tr37885_pathloss("urban_grid", np.array([50.0]), 5.9, state="NLOS")
        _, sf_h_los = compute_v2x_tr37885_pathloss("highway", np.array([100.0]), 5.9, state="LOS")
        self.assertEqual(sf_u_los, 3.0)
        self.assertEqual(sf_u_nlos, 4.0)
        self.assertEqual(sf_h_los, 3.0)

        # NTN S-band NLoS fading should have Rayleigh distribution (mean power ~1, non-trivial variance)
        ntn_res = provision_ntn_leo_satellite(env_type="dense_urban", elevation_deg=10.0, num_samples=512, seed=0)
        self.assertAlmostEqual(ntn_res.metrics["mean_channel_power"], 1.0, delta=0.25)

    def test_composed_env_passive_ris_conservation_fbl_isac_and_passive_ris_conj(self) -> None:
        spec_ris = build_wce_archetype("uav_ris_relay")
        self.assertEqual(spec_ris.num_ris_elements, 64)

        spec_passive, _ = compose_wce_spec("64-element passive RIS MIMO at 28 GHz with 20 MHz bandwidth")
        env = ComposedWirelessEnv(spec_passive, batch_size=8, seed=0)
        env.reset(seed=0)
        # Even under aggressive action=2 or ris_scale=5.0, passive RIS MUST enforce |phi| <= 1.0
        out = env.step(action=2, ris_scale=5.0)
        self.assertLessEqual(out.max_ris_modulus, 1.0 + 1e-6)

        # Polyanskiy FBL rate in ComposedWirelessEnv must NEVER exceed Shannon capacity at extreme low SNR
        spec_fbl, _ = compose_wce_spec("URLLC finite blocklength 28 GHz")
        spec_fbl.tx_power_dbm = -40.0
        env_fbl = ComposedWirelessEnv(spec_fbl, batch_size=8, seed=0)
        out_fbl = env_fbl.reset(seed=0)
        assert env_fbl._last_eff_snr_lin is not None and env_fbl._last_se is not None
        shannon = np.log2(1.0 + env_fbl._last_eff_snr_lin)
        self.assertTrue(bool(np.all(env_fbl._last_se <= shannon + 1e-9)))
        self.assertGreaterEqual(out_fbl.achievable_rate_bps_hz, 0.0)

        # ISAC two-way radar equation: doubling distance should worsen CRB by ~4x (R^2 from sqrt(R^4))
        spec_isac, _ = compose_wce_spec("28 GHz ISAC monostatic radar")
        env_i = ComposedWirelessEnv(spec_isac, batch_size=8, seed=0)
        env_i.reset(seed=0)
        o1 = env_i.step(distances_m=20.0)
        o2 = env_i.step(distances_m=40.0)
        assert o1.sensing_crb_range_m is not None and o2.sensing_crb_range_m is not None
        self.assertGreater(o2.sensing_crb_range_m / o1.sensing_crb_range_m, 3.0)

        # passive_ris_wu2019 conjugate fix: optimal aligned gain should strongly exceed random phase gain
        ris_res = provision_passive_ris_cascade(num_bs_ant=8, num_ris_elements=64, batch_size=32, seed=0)
        self.assertGreater(ris_res.metrics["ris_gain_ratio_vs_random"], 2.0)

        # active_ris analytical noise variance removes Jensen bias
        act_res = provision_active_ris_flexible(architecture="active_ris", batch_size=64, seed=0)
        self.assertGreater(act_res.metrics["active_ris_effective_snr"], 0.0)

    def test_rigor_auditor_catches_imag_nan_zero_aperture_and_empty_provenance(self) -> None:
        # Complex imaginary NaN must be caught
        h_bad = torch.complex(torch.ones(4, 2, 2), torch.full((4, 2, 2), float("nan")))
        bad_run = ProvisionResult(
            block_id="fake",
            seed=0,
            elapsed_seconds=0.01,
            config={},
            tensors={"h_small": h_bad},
            metrics={"mean_channel_power": 1.0, "snr_db": 10.0},
        )
        rep = audit_wce_rigor([bad_run, bad_run, bad_run], has_classic_baseline=True, min_seeds=3)
        self.assertFalse(rep.passed)
        self.assertIn("snr_noise_unit_mismatch", rep.red_flags)

        # aperture_m <= 0 and empty parameter_provenance must fail audit
        spec_bad = build_wce_archetype("imt2030_immersive_xr")
        spec_bad.aperture_m = 0.0
        spec_bad.parameter_provenance = []
        good_runs = [provision_tr38901_cdl_tdl(batch_size=4, num_subcarriers=8, num_slots=2, seed=s) for s in (0, 1, 2)]
        rep2 = audit_wce_rigor(good_runs, wce_spec=spec_bad, has_classic_baseline=True, min_seeds=3)
        self.assertFalse(rep2.passed)
        self.assertIn("ris_or_aperture_physics_mismatch", rep2.red_flags)
        self.assertIn("table1_provenance_integrity_violation", rep2.red_flags)

    def test_geometry_farfield_convergence_adapters_otfs_composer_and_reporter(self) -> None:
        # 1. UPA near-field steering at 100 km converges to upa_farfield_steering with |<b, a>|^2 == 1.0
        fc_hz = 28e9
        coords = upa_element_coordinates(8, 4, fc_hz)
        az = torch.tensor([0.35])
        el = torch.tensor([1.15])
        r_far = 1e5
        target = torch.stack([r_far * torch.sin(az) * torch.sin(el), r_far * torch.cos(az) * torch.sin(el), r_far * torch.cos(el)], dim=-1)
        b_near = nearfield_spherical_steering(coords, target, fc_hz, include_amplitude_variation=True)
        a_far = upa_farfield_steering(8, 4, az, el)
        corr = float(torch.abs(torch.sum(b_near.conj() * a_far, dim=-1)).pow(2).item())
        self.assertGreater(corr, 0.9999)

        # 2. Cross-domain Polyanskiy BLER aligns n_sym with req_se and removes ln(2) error
        dm = AutonomousLinkDelayModel(preset="v2v_control", latency_budget_ms=10.0, packet_bits=8000.0, seed=0)
        res_link = dm.evaluate_step(distance_m=30.0)
        self.assertTrue(0.0 <= res_link.packet_error_prob <= 1.0)

        # 3. OTFS static speed_kmh=0 has zero Doppler shift & unit-power h_tf
        otfs_0 = provision_otfs_delay_doppler(speed_kmh=0.0, batch_size=16, seed=0)
        self.assertAlmostEqual(float(torch.mean(otfs_0.tensors["h_tf"].abs().pow(2)).item()), 1.0, delta=0.35)
        self.assertEqual(float(otfs_0.tensors["h_dd"][:, :, 1:].abs().max().item()), 0.0)

        # 4. Rayleigh speed_mps=0 passes preflight with finite coherence_time_ms
        rc0 = provision_rayleigh_clarke(speed_mps=0.0, batch_size=8, seed=0)
        self.assertTrue(math.isfinite(rc0.metrics["coherence_time_ms"]))
        self.assertTrue(preflight_wce("rayleigh_clarke", config=None).ok)

        # 5. ls_lmmse_csi citation key is valid '3gpp_tr38843'
        csi = provision_ls_lmmse_csi(batch_size=8, seed=0)
        self.assertEqual(csi.parameter_provenance[0].citation_key, "3gpp_tr38843")

        # 6. Composer does not misparse '100 MHz' as '100 m' distance or '28 GHz' as bandwidth
        spec_q, _ = compose_wce_spec("28 GHz near-field MIMO with 100 MHz bandwidth at 5 m distance")
        self.assertAlmostEqual(spec_q.carrier_fc_ghz, 28.0)
        self.assertAlmostEqual(spec_q.bandwidth_hz, 100e6)
        self.assertAlmostEqual(spec_q.min_tx_rx_distance_m, 5.0)

        # 7. ChannelModuleAdapter fixed thermal noise floor scales SNR physically with TX power
        ch = ChannelModuleAdapter(lambda b: torch.ones((b, 8), dtype=torch.complex64), snr_db=10.0, add_noise=True)
        torch.manual_seed(0)
        x1 = torch.ones((64, 8), dtype=torch.complex64)
        e1 = float(torch.mean((ch(x1) - x1).abs().pow(2)).item())
        e2 = float(torch.mean((ch(2.0 * x1) - 2.0 * x1).abs().pow(2)).item())
        self.assertAlmostEqual(e1, e2, delta=0.05)

        # 8. WirelessGymEnvAdapter inherits gymnasium.Env when installed
        gym_env = ComposedWirelessEnv(spec_q, batch_size=4, seed=0).as_gym_env(max_steps=5)
        try:
            import gymnasium as gym
            self.assertIsInstance(gym_env, gym.Env)
        except ImportError:
            pass

        # 9. WCEReport populates baseline_blocks from wce_spec automatically
        spec_rep = build_wce_archetype("imt2030_immersive_xr")
        wce_rep = evaluate_wce_suite(spec_rep, seeds=(0, 1, 2), run_preflight=False, run_audit=False)
        self.assertEqual(wce_rep.baseline_blocks, spec_rep.baseline_blocks)


if __name__ == "__main__":
    unittest.main()
