"""Phase B verification tests: OFDM-MIMO Link-Level PHY, LDPC codec, RigorAuditor 2.0, and `wannado-wce verify`."""

from __future__ import annotations

import contextlib
import io
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from scipy import special

from wannado.wce import (
    audit_wce_rigor,
    compose_wce_spec,
    evaluate_link_level_phy,
    provision_tr38901_cdl_tdl,
    scaffold_wce_lab,
    verify_wce_target,
)
from wannado.wce.cli import main as cli_main
from wannado.wce.core.types import ParamProvenance, ProvisionResult
from wannado.wce.workspace.link_level_phy import (
    _get_ldpc,
    build_qam_constellation,
    ldpc_encode,
    sweep_link_level_bler,
)


class TestPhaseBLinkLevelPHYAndVerifyGate(unittest.TestCase):
    def test_qam_unit_power_and_ldpc_parity_invariant(self) -> None:
        for mod in ("QPSK", "16QAM", "64QAM"):
            syms, _ = build_qam_constellation(mod)
            self.assertAlmostEqual(float(torch.mean(syms.abs().pow(2)).item()), 1.0, places=5)

        _, h_full, _, _ = _get_ldpc(64, 64)
        u = np.random.default_rng(123).integers(0, 2, size=(128, 64), dtype=np.int8)
        cw = ldpc_encode(u)
        synd = (cw @ h_full.T) & 1
        self.assertEqual(int(synd.sum()), 0)

    def test_awgn_qpsk_matches_analytical_ber_and_ldpc_waterfall(self) -> None:
        h_awgn = torch.ones((128, 1, 1), dtype=torch.complex64)
        for snr_db in (2.0, 5.0, 8.0):
            res = evaluate_link_level_phy(
                h_awgn,
                snr_db=snr_db,
                modulation="QPSK",
                equalizer="mmse",
                estimator="genie",
                coded=True,
                num_blocks=128,
                seed=7,
            )
            theory_ber = 0.5 * float(special.erfc(math.sqrt((10.0 ** (snr_db / 10.0)) / 2.0)))
            self.assertAlmostEqual(res.uncoded_ber, theory_ber, delta=0.015)

        res_5 = evaluate_link_level_phy(
            h_awgn, snr_db=5.0, modulation="QPSK", equalizer="mmse", estimator="genie", coded=True, num_blocks=128, seed=7
        )
        self.assertLess(res_5.coded_ber, res_5.uncoded_ber * 0.25)

        res_8 = evaluate_link_level_phy(
            h_awgn, snr_db=8.0, modulation="QPSK", equalizer="mmse", estimator="genie", coded=True, num_blocks=128, seed=7
        )
        self.assertLess(res_8.coded_bler, res_5.coded_bler + 1e-6)

    def test_cdlc_mimo_equalizer_estimator_and_modulation_orderings(self) -> None:
        cdl = provision_tr38901_cdl_tdl(profile="CDL-C", batch_size=32, num_tx=4, num_rx=2, num_subcarriers=32, seed=42)

        mmse_genie = evaluate_link_level_phy(cdl, snr_db=14.0, modulation="16QAM", equalizer="mmse", estimator="genie", coded=True, num_blocks=64, seed=42)
        mmse_lmmse = evaluate_link_level_phy(cdl, snr_db=14.0, modulation="16QAM", equalizer="mmse", estimator="lmmse", coded=True, num_blocks=64, seed=42)
        mmse_ls = evaluate_link_level_phy(cdl, snr_db=14.0, modulation="16QAM", equalizer="mmse", estimator="ls", coded=True, num_blocks=64, seed=42)
        zf_ls = evaluate_link_level_phy(cdl, snr_db=14.0, modulation="16QAM", equalizer="zf", estimator="ls", coded=True, num_blocks=64, seed=42)

        # CSI estimation order: Genie <= LMMSE <= LS in NMSE & BER
        self.assertLess(mmse_genie.csi_nmse_db, mmse_lmmse.csi_nmse_db)
        self.assertLess(mmse_lmmse.csi_nmse_db, mmse_ls.csi_nmse_db)
        self.assertLessEqual(mmse_genie.coded_ber, mmse_lmmse.coded_ber + 0.01)
        self.assertLessEqual(mmse_lmmse.coded_ber, mmse_ls.coded_ber + 0.01)

        # Equalizer order: MMSE beats ZF on noisy frequency-selective MIMO
        self.assertLess(mmse_ls.evm_pct, zf_ls.evm_pct)
        self.assertLessEqual(mmse_ls.coded_bler, zf_ls.coded_bler + 0.02)

        # Modulation order: QPSK <= 16QAM <= 64QAM BLER at fixed 14 dB SNR
        qpsk = evaluate_link_level_phy(cdl, snr_db=14.0, modulation="QPSK", equalizer="mmse", estimator="ls", coded=True, num_blocks=48, seed=42)
        qam64 = evaluate_link_level_phy(cdl, snr_db=14.0, modulation="64QAM", equalizer="mmse", estimator="ls", coded=True, num_blocks=48, seed=42)
        self.assertLessEqual(qpsk.coded_bler, mmse_ls.coded_bler)
        self.assertLessEqual(mmse_ls.coded_bler, qam64.coded_bler)

        # Sweep API emits monotonic waterfall goodput
        wf = sweep_link_level_bler(cdl, snr_db_list=[4.0, 12.0, 20.0], modulation="QPSK", num_blocks=24, seed=42)
        self.assertGreater(wf[-1].effective_goodput_bps_hz, wf[0].effective_goodput_bps_hz)

    def test_rigor_auditor_v2_catches_aperture_shrink_and_table1_duplicates(self) -> None:
        spec, _ = compose_wce_spec("64-element RIS MIMO at 28 GHz with 32 antennas")
        runs = [
            ProvisionResult(
                block_id="synth",
                seed=s,
                elapsed_seconds=0.01,
                config={},
                tensors={"h": torch.ones((4, 2, 32), dtype=torch.complex64)},
                metrics={"mean_spectral_efficiency_bps_hz": 4.0 + 0.1 * s, "mean_channel_power": 1.0},
            )
            for s in (0, 1, 2)
        ]
        # Valid baseline check passes RigorAuditor 2.0
        ok_rep = audit_wce_rigor(runs, wce_spec=spec, has_classic_baseline=True, min_seeds=3)
        self.assertTrue(ok_rep.checks["ris_quantization_and_aperture_ok"])
        self.assertTrue(ok_rep.checks["table1_provenance_integrity_ok"])

        # Planted Trap 9: Impossibly shrunken array aperture hiding near-field
        bad_ap_spec = compose_wce_spec("64 antennas at 3.5 GHz")[0]
        bad_ap_spec.aperture_m = 0.001
        rep_ap = audit_wce_rigor(runs, wce_spec=bad_ap_spec, has_classic_baseline=True, min_seeds=3)
        self.assertIn("ris_or_aperture_physics_mismatch", rep_ap.red_flags)

        # Planted Trap 10: Duplicate contradictory Table I row
        bad_tbl_spec = compose_wce_spec("28 GHz MIMO")[0]
        bad_tbl_spec.parameter_provenance.append(
            ParamProvenance("bandwidth_hz", 1e8, "Hz", "dup", "gpp_ts38211", "Clause 4.2", "3gpp_ts38211", "dup")
        )
        rep_tbl = audit_wce_rigor(runs, wce_spec=bad_tbl_spec, has_classic_baseline=True, min_seeds=3)
        self.assertIn("table1_provenance_integrity_violation", rep_tbl.red_flags)

    def test_verify_gate_and_cli_subcommand(self) -> None:
        # 1. Programmatic verify across query & blocks (with + without PHY channel)
        rep_q = verify_wce_target(query="64-element RIS MIMO at 28 GHz", seeds=(0, 1, 2), with_phy_check=True)
        self.assertTrue(rep_q.passed, msg=rep_q.format_report())
        self.assertEqual(rep_q.exit_code, 0)
        assert rep_q.phy_validation is not None
        self.assertTrue(rep_q.phy_validation["phy_applicable"])

        # Traffic + classic scheduler baseline passes without channel PHY check, while single block without baseline fails rigor
        rep_ftp_solo = verify_wce_target(blocks=["traffic_ftp_models"], seeds=(0, 1, 2), with_phy_check=True)
        self.assertFalse(rep_ftp_solo.passed)
        self.assertTrue(any("missing_classic_baseline_reference" in rf for rf in rep_ftp_solo.red_flags))

        rep_ftp = verify_wce_target(blocks=["traffic_ftp_models", "proportional_fair"], seeds=(0, 1, 2), with_phy_check=True)
        self.assertTrue(rep_ftp.passed, msg=rep_ftp.format_report())
        assert rep_ftp.phy_validation is not None
        self.assertFalse(rep_ftp.phy_validation["phy_applicable"])

        # 2. CLI verify on saved spec, scaffolded ./wce_lab/interface.py, and failing broken spec
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            spec_p = root / "spec.json"
            spec, _ = compose_wce_spec("LEO NTN satellite downlink at 20 GHz Ka-band")
            spec.save(spec_p)

            buf_ok = io.StringIO()
            with contextlib.redirect_stdout(buf_ok):
                rc_ok = cli_main(["verify", "--spec", str(spec_p), "--json"])
            self.assertEqual(rc_ok, 0)
            payload = json.loads(buf_ok.getvalue())
            self.assertTrue(payload["passed"])
            self.assertTrue(payload["checks"]["link_level_phy_ok"])

            written = scaffold_wce_lab(root / "wce_lab", query="28 GHz UAV RIS ISAC", overwrite=True)
            with contextlib.redirect_stdout(io.StringIO()):
                rc_iface = cli_main(["verify", "--interface", str(written["interface_py"]), "--json"])
            self.assertEqual(rc_iface, 0)

            bad_spec_p = root / "bad_spec.json"
            spec.num_tx = 0
            spec.save(bad_spec_p)
            buf_fail = io.StringIO()
            with contextlib.redirect_stdout(buf_fail):
                rc_fail = cli_main(["verify", "--spec", str(bad_spec_p), "--json"])
            self.assertEqual(rc_fail, 1)
            fail_payload = json.loads(buf_fail.getvalue())
            self.assertFalse(fail_payload["passed"])
            self.assertFalse(fail_payload["checks"]["stage0_spec_audit_ok"])


if __name__ == "__main__":
    unittest.main()
