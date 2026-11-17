"""Unit tests for v1.1 4-Slot Compositional Assembler, LiveProvenance Diff, ComposedWirelessEnv, and JIT Adapter Recipes."""

from __future__ import annotations

import contextlib
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from wannado.wce import (
    ComposedWirelessEnv,
    adapt_external_channel_tensor,
    audit_wce_rigor,
    check_tool_platform_readiness,
    compose_wce_spec,
    recommend_wce_blueprint,
    scaffold_wce_lab,
    update_wce_spec_with_diff,
    verify_external_channel_adapter,
    verify_jit_adapter_file,
    write_jit_adapter_bundle,
)
from wannado.wce.cli import main as cli_main
from wannado.wce.standards.jit_recipes import JIT_RECIPES


class TestCompositionalAndJITAdapters(unittest.TestCase):
    def test_4slot_compositional_assembler_multitech_query(self) -> None:
        query = "我要做 UAV 搭载 STAR-RIS 在 28 GHz 近场辅助 ISAC，且带有 60 fps XR 业务与真实 LMMSE 导频误差"
        spec, plan = compose_wce_spec(query)

        self.assertEqual(spec.archetype, "composed_custom")
        self.assertEqual(spec.carrier_fc_ghz, 28.0)
        self.assertEqual(spec.numerology_mu, 3)
        self.assertAlmostEqual(spec.slot_duration_ms, 0.125)
        self.assertEqual(spec.traffic_model, "xr_60fps")
        self.assertEqual(spec.csi_assumption, "realistic_estimated")

        # Verify all 4 orthogonal slots populated cleanly
        for blk in ("uav_air_to_ground", "star_ris", "nearfield_xlmimo_thz", "isac_jsac_channel", "traffic_xr_immersive"):
            self.assertIn(blk, spec.blocks_selected)
        for base in ("ls_lmmse_csi", "wmmse_precoder", "proportional_fair", "awgn_link"):
            self.assertIn(base, spec.baseline_blocks)
        self.assertEqual(len(plan["slots"]), 4)

        bp = recommend_wce_blueprint(query)
        self.assertEqual(bp["archetype"], "composed_custom")
        self.assertIn("composition_plan", bp)

    def test_live_provenance_diff_triggers_boundary_alerts(self) -> None:
        base_spec, _ = compose_wce_spec("3.5 GHz urban cellular MIMO with UPA and FTP traffic")
        self.assertNotIn("nearfield_xlmimo_thz", base_spec.blocks_selected)

        # Case 1: Shift frequency above TR 38.901 100 GHz ceiling via Chinese NL diff
        diff_thz = update_wce_spec_with_diff(base_spec, "把频段改成 140 GHz，距离 10 m")
        self.assertEqual(diff_thz.updated_spec.carrier_fc_ghz, 140.0)
        self.assertIn("nearfield_xlmimo_thz", diff_thz.auto_promoted_blocks)
        self.assertIn("itur_p676", diff_thz.updated_spec.standard_ids)
        self.assertTrue(any("100 GHz" in a or "140.0 GHz" in a for a in diff_thz.boundary_alerts))
        self.assertIn("itur_p676", diff_thz.updated_table1_md)
        self.assertIn("itur_p676_2022", diff_thz.updated_table1_tex)

        # Case 2: Scale aperture/antennas inside Rayleigh distance
        diff_nf = update_wce_spec_with_diff(base_spec, {"num_tx": 256, "min_tx_rx_distance_m": 4.0})
        self.assertGreater(diff_nf.after_physics["rayleigh_distance_m"], diff_nf.before_physics["rayleigh_distance_m"])
        self.assertIn("nearfield_xlmimo_thz", diff_nf.auto_promoted_blocks)
        self.assertTrue(any("Near-field transition" in a for a in diff_nf.boundary_alerts))

    def test_composed_wireless_env_trajectory_and_rigor_audit(self) -> None:
        query = "UAV + STAR-RIS + 28 GHz ISAC with 60 fps XR traffic and LMMSE CSI"
        spec, _ = compose_wce_spec(query)

        # Run 3 independent seeds and pass through Stage 3 Wireless Rigor Auditor
        rigor_runs = [ComposedWirelessEnv(spec, batch_size=16, seed=s).to_rigor_result(num_slots=12, seed=s) for s in (0, 1, 2)]
        rep = audit_wce_rigor(rigor_runs, wce_spec=spec, has_classic_baseline=True, min_seeds=3)
        self.assertTrue(rep.passed, msg=f"ComposedWirelessEnv failed rigor audit: {rep.red_flags}")
        self.assertIsNotNone(rigor_runs[0].metrics.get("sensing_crb_range_m"))

        # Check ChannelModuleAdapter (for wannado-adv / Semantic AI) & Gym adapter (for wannado-drl)
        env = ComposedWirelessEnv(spec, batch_size=8, seed=42)
        ch_mod = env.as_channel_module(snr_db=12.0)
        x = (torch.randn(8, spec.num_tx) + 1j * torch.randn(8, spec.num_tx)).to(torch.complex64).requires_grad_(True)
        y = ch_mod(x)
        torch.mean(y.abs().pow(2)).backward()
        self.assertIsNotNone(x.grad)

        gym_env = env.as_gym_env(max_steps=10)
        obs, _ = gym_env.reset(seed=0)
        obs2, rew, term, _trunc, _ = gym_env.step(1)
        self.assertEqual(obs.shape, (4,))
        self.assertEqual(obs2.shape, (4,))
        self.assertTrue(np.isfinite(rew))
        self.assertFalse(term)

    def test_scaffold_with_query_and_external_adapters_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            lab_dir = Path(tmpdir) / "wce_lab"
            written = scaffold_wce_lab(
                lab_dir,
                query="UAV 搭载 STAR-RIS 在 28 GHz 做 ISAC 并接入 Sionna RT",
                overwrite=True,
            )
            self.assertIn("adapter_sionna_rt_adapter_py", written)
            self.assertTrue(written["adapter_sionna_rt_adapter_py"].exists())

            proc = subprocess.run(
                [sys.executable, str(written["run_provision_py"]), "--seeds", "0", "1", "--slots", "8"],
                cwd=tmpdir,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, msg=f"stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}")
            self.assertIn("ComposedWirelessEnv", proc.stdout)

    def test_external_channel_adapter_normalization_and_all_jit_recipes(self) -> None:
        # 1. Sionna 7D raw CIR normalization + autograd check
        rng = np.random.default_rng(123)
        raw_cir = (rng.normal(size=(8, 1, 2, 1, 4, 5, 3)) + 1j * rng.normal(size=(8, 1, 2, 1, 4, 5, 3))) * 1e-5
        # Unnormalized raw tensor MUST fail power drift guard
        bad_verdict = verify_external_channel_adapter(torch.from_numpy(raw_cir).to(torch.complex64), expect_normalized=True)
        self.assertFalse(bad_verdict["ok"])

        # Adapted normalized tensor MUST pass with separated pathgain
        h_small, path_gain = adapt_external_channel_tensor(raw_cir, source_layout="sionna_cir", strip_large_scale_gain=True)
        self.assertEqual(tuple(h_small.shape), (8, 2, 4, 3))
        self.assertEqual(tuple(path_gain.shape), (8,))
        good_verdict = verify_external_channel_adapter(h_small, expect_normalized=True)
        self.assertTrue(good_verdict["ok"])
        self.assertTrue(good_verdict["autograd_ok"])

        # 2. Readiness checks for DeepMIMO disk size & ns-3 OS/Docker guard
        dm_ready = check_tool_platform_readiness("deepmimo", deepmimo_scenario="boston5g_28")
        self.assertEqual(dm_ready["estimated_required_gb"], 3.5)
        ns3_ready = check_tool_platform_readiness("ns3_5glena_ns3gym")
        if ns3_ready["os"] != "Linux":
            self.assertGreaterEqual(len(ns3_ready["warnings"]), 1)

        # 3. Materialize & execute --self-test on ALL registered JIT recipes
        with tempfile.TemporaryDirectory() as tmpdir:
            for tid in sorted(JIT_RECIPES.keys()):
                bundle = write_jit_adapter_bundle(tid, dest_dir=tmpdir, overwrite=True)
                self.assertTrue(bundle["adapter_py"].exists())
                res = verify_jit_adapter_file(bundle["adapter_py"])
                self.assertTrue(res["ok"], msg=f"JIT recipe {tid} --self-test failed: {res}")

    def test_v1_1_cli_subcommands(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, contextlib.redirect_stdout(io.StringIO()):
            spec_p = Path(tmpdir) / "composed_spec.json"
            rc_c = cli_main(["compose", "28 GHz UAV STAR-RIS ISAC with XR", "--out", str(spec_p), "--json"])
            self.assertEqual(rc_c, 0)
            self.assertTrue(spec_p.exists())

            rc_d = cli_main(["diff-spec", "--spec", str(spec_p), "--request", "把频段改成 140 GHz", "--save", "--json"])
            self.assertEqual(rc_d, 0)

            rc_j = cli_main(["jit-recipe", "sionna_rt", "--write-dir", str(Path(tmpdir) / "adapters"), "--verify", "--json"])
            self.assertEqual(rc_j, 0)

    def test_stress_regressions_uav_legacy_traps_raytracing_and_140ghz_nearfield(self) -> None:
        from wannado.wce import build_interactive_brief, scan_project

        # 1. Pure UAV query should NOT inject unrequested RIS blocks, while returning UAV clarifying forks & aligned dialog
        uav_bp = recommend_wce_blueprint("我要做 6G 低空无人机通信，用什么信道合适？")
        self.assertIn("uav_air_to_ground", uav_bp["wce_spec"]["blocks_selected"])
        self.assertNotIn("passive_ris_cascade", uav_bp["wce_spec"]["blocks_selected"])
        uav_brief = build_interactive_brief(query="我要做 6G 低空无人机通信，用什么信道合适？")
        self.assertIn("uav_role", [f["fork_id"] for f in uav_brief["clarifying_forks"]])
        self.assertIn("uav_air_to_ground", uav_brief["goal_confirmation_dialog"]["ai_understanding_summary"])

        # 2. Legacy messy script trap detection
        with tempfile.TemporaryDirectory() as tmpdir:
            bad = Path(tmpdir) / "legacy_beam.py"
            bad.write_text("import torch\nH = torch.randn(32, 8)\nphi = torch.randn(64)\nperfect_csi = H\n", encoding="utf-8")
            sc = scan_project(tmpdir)
            trap_ids = {t["trap_id"] for t in sc.detected_traps}
            self.assertIn("unnormalized_channel_power", trap_ids)
            self.assertIn("genie_or_perfect_csi", trap_ids)
            self.assertIn("unseeded_randomness", trap_ids)

        # 3. Real building 28 GHz query triggers Sionna RT & DeepMIMO + readiness + DeepMIMO catalog
        rt_brief = build_interactive_brief(query="普通统计信道不够，我要建真实建筑遮挡的 28 GHz 街区无线仿真环境")
        tool_ids = {t["id"] for t in rt_brief["jit_plan"]["external_tools"]}
        self.assertIn("sionna_rt", tool_ids)
        self.assertIn("deepmimo", tool_ids)
        self.assertTrue(len(rt_brief["jit_plan"]["readiness_reports"]) >= 2)
        self.assertTrue(bool(rt_brief["jit_plan"]["deepmimo_catalog"]))

        # 4. 140 GHz + 256 antennas + 5 m MUST trigger both band ceiling alert AND near-field transition AND direct mu=4
        base, _ = compose_wce_spec("3.5 GHz urban MIMO")
        d_thz = update_wce_spec_with_diff(base, "把频段改成 140 GHz，基站阵列换成 256 个天线，距离 5 m")
        self.assertTrue(any("Near-field transition" in a for a in d_thz.boundary_alerts))
        self.assertTrue(any("100 GHz" in a or "140.0 GHz" in a for a in d_thz.boundary_alerts))
        self.assertEqual(d_thz.updated_spec.numerology_mu, 4)
        self.assertIn("numerology_mu", d_thz.changed_fields)


if __name__ == "__main__":
    unittest.main()
