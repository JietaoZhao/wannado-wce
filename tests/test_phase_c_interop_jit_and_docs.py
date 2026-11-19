"""Phase C verification tests: JIT honesty modes, multi-user Gym control, complex64 autograd, verify gate, and streamlined Agent Kit."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from wannado.wce import ComposedWirelessEnv, compose_wce_spec
from wannado.wce.agent_kit import install_agent_kit, load_skill_text
from wannado.wce.audit.verify_gate import verify_wce_target
from wannado.wce.standards.jit_recipes import verify_jit_adapter_file, write_jit_adapter_bundle


class TestPhaseCInteropJITAndDocs(unittest.TestCase):
    def test_jit_verification_mode_honesty_and_require_live(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            for tool_id in ("sionna_rt", "deepmimo", "ns3_5glena_ns3gym", "quadriga_lib"):
                bundle = write_jit_adapter_bundle(tool_id, dest_dir=Path(tmp) / tool_id, overwrite=True)
                adapter_py = bundle["adapter_py"]

                smoke = verify_jit_adapter_file(adapter_py, require_live=False)
                self.assertTrue(smoke["ok"], msg=f"{tool_id} contract smoke failed: {smoke}")
                self.assertIn(smoke["verification_mode"], ("synthetic_contract_smoke", "live_runtime_probe"))

                if not smoke["live_library_installed"]:
                    self.assertEqual(smoke["verification_mode"], "synthetic_contract_smoke")
                    strict = verify_jit_adapter_file(adapter_py, require_live=True)
                    self.assertFalse(strict["ok"])
                    self.assertEqual(strict["returncode"], 1)
                    self.assertEqual(strict["verification_mode"], "synthetic_contract_smoke")

    def test_composed_env_multi_user_gym_continuous_allocation(self) -> None:
        spec, _ = compose_wce_spec("64-element passive RIS MIMO at 28 GHz for multi-user beamforming")
        env = ComposedWirelessEnv(spec, batch_size=4, seed=42)
        gym_env = env.as_gym_env(max_steps=15, multi_user=True)

        obs0, _ = gym_env.reset(seed=42)
        self.assertEqual(obs0.shape, (4 * 4 + 4,))
        self.assertEqual(gym_env.action_space.shape, (4,))

        obs_u, rew_u, _, _, _ = gym_env.step(np.zeros(4, dtype=np.float32))
        gym_env.reset(seed=42)
        obs_s, rew_s, _, _, _ = gym_env.step(np.array([4.0, -4.0, -4.0, -4.0], dtype=np.float32))

        self.assertTrue(np.isfinite(rew_u) and np.isfinite(rew_s))
        # Per-user SNR slice starts at index B=4: allocating weight -> user 0 raises SNR0 & drops SNR1
        self.assertGreater(float(obs_s[4] - obs_u[4]), 4.0)
        self.assertLess(float(obs_s[5] - obs_u[5]), -6.0)

        gym_env.reset(seed=42)
        for step_idx in range(1, 16):
            _, _, term, trunc, _ = gym_env.step(np.zeros(4, dtype=np.float32))
            self.assertFalse(term)
            if step_idx < 15:
                self.assertFalse(trunc)
            else:
                self.assertTrue(trunc)

    def test_stateless_channel_module_complex64_autograd(self) -> None:
        spec, _ = compose_wce_spec("64-element passive RIS MIMO at 28 GHz")
        env = ComposedWirelessEnv(spec, batch_size=4, seed=42)
        ch = env.as_channel_module(snr_db=15.0, stateless=True)

        h_snap = ch.channel_sampler(4)
        nt = h_snap.shape[-1] if h_snap.ndim == 3 else 1
        x = torch.randn(4, nt, dtype=torch.complex64, requires_grad=True)
        y = ch(x, h=h_snap)

        self.assertEqual(y.dtype, torch.complex64)
        self.assertTrue(torch.isfinite(y.real).all() and torch.isfinite(y.imag).all())
        loss = y.abs().pow(2).sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad.real).all())

    def test_verify_gate_json_contract_and_10_checks(self) -> None:
        report = verify_wce_target(query="64-element RIS MIMO at 28 GHz", seeds=(0, 1, 2))
        self.assertTrue(report.passed)
        self.assertEqual(report.exit_code, 0)
        payload = report.to_dict()
        checks = (payload.get("composed_rigor") or {}).get("checks", {})
        self.assertEqual(len(checks), 10)
        self.assertTrue(checks.get("ris_quantization_and_aperture_ok"))
        self.assertTrue(checks.get("table1_provenance_integrity_ok"))

        # Single-seed verification must NOT silently bypass single_seed_survivorship_risk
        single = verify_wce_target(query="64-element RIS MIMO at 28 GHz", seeds=(0,))
        self.assertFalse(single.passed)
        self.assertEqual(single.exit_code, 1)
        self.assertTrue(any("single_seed_survivorship_risk" in rf for rf in single.red_flags))

    def test_agent_kit_streamlined_coding_agent_workflow(self) -> None:
        skill = load_skill_text()
        self.assertIn("wannado-wce verify", skill)
        self.assertIn("evaluate_link_level_phy", skill)
        self.assertIn("synthetic_contract_smoke", skill)
        self.assertNotIn("Pop the Mandatory Two-Step Goal Confirmation Dialog", skill)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            install_agent_kit(root, ide="all")
            for rel in ("AGENTS.md", "CLAUDE.md", ".cursor/rules/wannado-wce.mdc"):
                txt = (root / rel).read_text(encoding="utf-8")
                self.assertIn("wannado-wce verify", txt)
                self.assertNotIn("Goal Confirmation Dialog", txt)


if __name__ == "__main__":
    unittest.main()
