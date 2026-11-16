"""End-to-end tests for Stage 0 intake, Stage 1 preflight, Stage 2 Co-Pilot, ./wce_lab/ boundary, Stage 3 rigor traps, Stage 4 reporter, and CLI."""

from __future__ import annotations

import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from wannado.wce import (
    WCESpec,
    audit_wce_rigor,
    audit_wce_spec,
    build_interactive_brief,
    build_jit_plan,
    build_post_run_debrief,
    build_wce_archetype,
    eject_block,
    evaluate_wce_suite,
    explain_parameters,
    preflight_wce,
    recommend_wce_blueprint,
    scaffold_wce_lab,
    scan_project,
)
from wannado.wce.cli import main as cli_main
from wannado.wce.core.types import ProvisionResult


class TestWorkflowAuditReporterCLI(unittest.TestCase):
    def test_stage0_spec_roundtrip_table1_and_audit(self) -> None:
        spec = build_wce_archetype("imt2030_isac_monostatic")
        with tempfile.TemporaryDirectory() as tmpdir:
            json_p = Path(tmpdir) / "spec.json"
            yaml_p = Path(tmpdir) / "spec.yaml"
            spec.save(json_p)
            spec.save(yaml_p)
            loaded_j = WCESpec.load(json_p)
            loaded_y = WCESpec.load(yaml_p)
            self.assertEqual(loaded_j.problem_name, spec.problem_name)
            self.assertEqual(loaded_y.carrier_fc_ghz, spec.carrier_fc_ghz)

        md_tbl = spec.format_system_parameter_table_md()
        tex_tbl = spec.format_system_parameter_table_tex()
        self.assertIn("Authority & Clause", md_tbl)
        self.assertIn("\\begin{table}", tex_tbl)
        self.assertIn("\\cite{", tex_tbl)

        ok_audit = audit_wce_spec(spec)
        self.assertTrue(ok_audit.ok)

        # Trap near-field mismatch on Stage 0 audit
        bad_spec = build_wce_archetype("xlmimo_nearfield_thz")
        bad_spec.blocks_selected = ["farfield_upa"]
        bad_audit = audit_wce_spec(bad_spec)
        self.assertTrue(len(bad_audit.warnings) >= 1)

        # Build JIT plan & check tool recommendations + preflight
        jit = build_jit_plan(spec)
        self.assertGreaterEqual(len(jit.builtin_blocks), 1)
        pf = preflight_wce("isac_jsac_channel", wce_spec=spec, seed=0)
        self.assertTrue(pf.ok)

    def test_stage3_rigor_auditor_catches_all_7_planted_traps(self) -> None:
        clean_runs = [
            ProvisionResult(
                block_id="test_block",
                seed=s,
                elapsed_seconds=0.05,
                config={},
                tensors={"h": torch.ones((4, 4), dtype=torch.complex64)},
                metrics={"mean_spectral_efficiency_bps_hz": 5.0 + 0.1 * s, "mean_channel_power": 1.0, "snr_db": 10.0},
                diagnostics={"lag1_autocorr_gap": 0.02, "max_ris_modulus": 1.0, "is_passive_ris": True},
            )
            for s in (0, 1, 2)
        ]
        clean_rep = audit_wce_rigor(clean_runs, has_classic_baseline=True, min_seeds=3)
        self.assertTrue(clean_rep.passed)

        # Planted Trap 1: Channel power drift
        t1 = [ProvisionResult(block_id="b", seed=s, elapsed_seconds=0.01, config={}, metrics={"mean_channel_power": 8.5}) for s in (0, 1, 2)]
        self.assertIn("channel_power_normalization_drift", audit_wce_rigor(t1).red_flags)

        # Planted Trap 2: Non-finite tensor / absurd SNR
        t2 = [
            ProvisionResult(block_id="b", seed=s, elapsed_seconds=0.01, config={}, tensors={"h": torch.tensor([float("nan")])}, metrics={"snr_db": 500.0})
            for s in (0, 1, 2)
        ]
        self.assertIn("snr_noise_unit_mismatch", audit_wce_rigor(t2).red_flags)

        # Planted Trap 3: Future CSI leakage
        t3 = [ProvisionResult(block_id="b", seed=s, elapsed_seconds=0.01, config={}, metrics={"r": 1.0}, diagnostics={"has_future_csi_leakage": True}) for s in (0, 1, 2)]
        self.assertIn("genie_csi_or_future_leakage", audit_wce_rigor(t3).red_flags)

        # Planted Trap 4: Doppler coherence broken
        t4 = [ProvisionResult(block_id="b", seed=s, elapsed_seconds=0.01, config={}, metrics={"r": 1.0}, diagnostics={"lag1_autocorr_gap": 0.8}) for s in (0, 1, 2)]
        self.assertIn("doppler_coherence_violation", audit_wce_rigor(t4).red_flags)

        # Planted Trap 5: Passive RIS |phi| > 1
        t5 = [ProvisionResult(block_id="b", seed=s, elapsed_seconds=0.01, config={}, metrics={"r": 1.0}, diagnostics={"is_passive_ris": True, "max_ris_modulus": 2.4}) for s in (0, 1, 2)]
        self.assertIn("passive_ris_energy_conservation_violation", audit_wce_rigor(t5).red_flags)

        # Planted Trap 6: Far-field assumed inside Rayleigh distance
        t6 = [ProvisionResult(block_id="b", seed=s, elapsed_seconds=0.01, config={}, metrics={"r": 1.0}, diagnostics={"min_tx_rx_distance_m": 5.0, "rayleigh_distance_m": 40.0, "assumed_farfield": True}) for s in (0, 1, 2)]
        self.assertIn("farfield_assumption_in_nearfield_regime", audit_wce_rigor(t6).red_flags)

        # Planted Trap 7: Single seed / missing classical baseline
        t7 = [clean_runs[0]]
        t7_flags = audit_wce_rigor(t7, has_classic_baseline=False, min_seeds=3).red_flags
        self.assertIn("single_seed_survivorship_risk", t7_flags)
        self.assertIn("missing_classic_baseline_reference", t7_flags)

    def test_stage2_copilot_and_isolated_wce_lab_scaffold_and_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            # Create dummy wireless RL project script to test Clue Scanning
            (root / "train_beamforming.py").write_text("import torch\nimport gymnasium as gym\n# RIS near-field MIMO at 28 GHz\n", encoding="utf-8")
            scan = scan_project(root)
            self.assertIn("PyTorch", scan.frameworks)
            self.assertGreaterEqual(len(scan.evidence_clues), 1)

            rec = recommend_wce_blueprint("6G THz near-field XL-MIMO with DRL beamforming")
            self.assertEqual(rec["archetype"], "xlmimo_nearfield_thz")

            brief = build_interactive_brief(project_scan=scan, query="28 GHz ISAC monostatic radar and multi-user MIMO")
            self.assertIn("goal_confirmation_dialog", brief)
            self.assertGreaterEqual(len(brief["questions_to_ask_user"]), 4)
            self.assertGreaterEqual(len(brief["parameter_explanations"]), 4)
            self.assertGreaterEqual(len(explain_parameters()), 4)

            # Scaffold ./wce_lab/, eject a standalone block, run run_provision.py subprocess
            lab_dir = root / "wce_lab"
            written = scaffold_wce_lab(lab_dir, archetype="imt2030_isac_monostatic")
            self.assertTrue(written["wce_spec_json"].exists())
            self.assertTrue(written["interface_py"].exists())
            self.assertTrue(written["run_provision_py"].exists())
            self.assertTrue(written["jit_setup_sh"].exists())

            ejected = eject_block("passive_ris_cascade", dest_dir=lab_dir / "blocks")
            self.assertTrue(ejected.exists())

            proc = subprocess.run(
                [sys.executable, str(written["run_provision_py"]), "--seeds", "0", "1"],
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, msg=f"run_provision stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}")
            out_dir = lab_dir / "outputs"
            self.assertTrue((out_dir / "environment_provenance.md").exists())
            self.assertTrue((out_dir / "paper_params_table.tex").exists())
            self.assertTrue((out_dir / "paper_results_table.tex").exists())
            self.assertTrue((out_dir / "references.bib").exists())
            self.assertTrue((out_dir / "wce_report.json").exists())
            self.assertTrue((out_dir / "metrics.csv").exists())
            self.assertTrue((out_dir / "wce_diagnostics.png").exists() or (out_dir / "wce_diagnostics.svg").exists())

            md_txt = (out_dir / "environment_provenance.md").read_text(encoding="utf-8")
            self.assertIn("Table I", md_txt)
            self.assertIn("3GPP", md_txt)

    def test_stage4_evaluate_across_multiple_archetypes(self) -> None:
        for arch in ("imt2030_immersive_xr", "xlmimo_nearfield_thz", "multicell_ofdma_mac", "jamming_contested_rf"):
            spec = build_wce_archetype(arch)
            blocks = list(dict.fromkeys(spec.blocks_selected + spec.baseline_blocks))
            rep = evaluate_wce_suite(blocks, wce_spec=spec, seeds=(0, 1))
            with tempfile.TemporaryDirectory() as tmpdir:
                saved = rep.save(tmpdir)
                self.assertTrue(saved["environment_provenance_md"].exists())
                self.assertTrue(saved["paper_params_table_tex"].exists())
                debrief = build_post_run_debrief(rep, output_dir=tmpdir)
                self.assertIn("plain_verdict", debrief)

    def test_cli_subcommands(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, contextlib.redirect_stdout(io.StringIO()):
            rc_ars = cli_main(["arsenal", "--json"])
            self.assertEqual(rc_ars, 0)
            rc_std = cli_main(["standards", "--org", "3GPP", "--json"])
            self.assertEqual(rc_std, 0)
            rc_eco = cli_main(["ecosystem", "--probe", "--json"])
            self.assertEqual(rc_eco, 0)
            rc_cat = cli_main(["catalog", "--family", "domain_6g", "--json"])
            self.assertEqual(rc_cat, 0)
            rc_lin = cli_main(["lineage", "--family", "baselines"])
            self.assertEqual(rc_lin, 0)
            rc_rec = cli_main(["recommend", "LEO", "NTN", "satellite", "IoT", "--json"])
            self.assertEqual(rc_rec, 0)
            rc_cite = cli_main(["cite", "isac_jsac_channel", "--format", "bibtex"])
            self.assertEqual(rc_cite, 0)
            spec_out = Path(tmpdir) / "spec.json"
            rc_in = cli_main(["intake", "--archetype", "v2x_highway_urllc", "--out", str(spec_out), "--json"])
            self.assertEqual(rc_in, 0)
            self.assertTrue(spec_out.exists())
            parsed = json.loads(spec_out.read_text(encoding="utf-8"))
            self.assertEqual(parsed["imt2030_scenario"], "HRLLC")


if __name__ == "__main__":
    unittest.main()
