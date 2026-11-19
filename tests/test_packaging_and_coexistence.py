"""Unit tests for PEP 420 wannado.* isolation, 3-way multi-package coexistence, catalog integrity, and citations."""

from __future__ import annotations

import importlib
import sys
import tempfile
import unittest
from pathlib import Path

from wannado.wce import __version__, export_citations, install_agent_kit, load_catalog, query_catalog
from wannado.wce.blocks import BLOCK_EXPORTS, BLOCK_ID_TO_RUNNER
from wannado.wce.catalog.build_catalog import build_registry_json
from wannado.wce.copilot import get_sota_lineage
from wannado.wce.core.provenance import BLOCK_DEFAULT_MECHANISMS, MECHANISM_PROVENANCE, STANDARDS_PROVENANCE
from wannado.wce.standards.ecosystem import ECOSYSTEM_REGISTRY

WCE_SRC = Path(__file__).resolve().parents[1] / "src"
WANNA_ROOT = Path(__file__).resolve().parents[2]
DRL_SRC = WANNA_ROOT / "wannado-drl" / "src"
ADV_SRC = WANNA_ROOT / "wannado-adv" / "src"


class TestPackagingAndCoexistence(unittest.TestCase):
    def test_no_wannado_init_py_pep420_namespace(self) -> None:
        self.assertFalse((WCE_SRC / "wannado" / "__init__.py").exists())
        self.assertEqual(__version__, "1.2.1")

    def test_three_way_wannado_namespace_import(self) -> None:
        added: list[str] = []
        for p in (DRL_SRC, ADV_SRC):
            if p.exists() and str(p) not in sys.path:
                sys.path.insert(0, str(p))
                added.append(str(p))
        try:
            import wannado
            import wannado.wce

            self.assertIsNone(getattr(wannado, "__file__", None))
            self.assertEqual(len(BLOCK_ID_TO_RUNNER), 33)
            if DRL_SRC.exists():
                mod_drl = importlib.import_module("wannado.drl")
                self.assertTrue(hasattr(mod_drl, "__version__"))
            if ADV_SRC.exists():
                mod_adv = importlib.import_module("wannado.adv")
                self.assertTrue(hasattr(mod_adv, "__version__"))
        finally:
            for item in added:
                if item in sys.path:
                    sys.path.remove(item)

    def test_agent_kit_marker_isolation_with_drl_and_adv(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            res = install_agent_kit(root, ide="all")
            self.assertTrue((root / "AGENTS.md").exists())
            self.assertTrue((root / "CLAUDE.md").exists())
            self.assertTrue((root / ".agents/skills/wannado-wce/SKILL.md").exists())
            self.assertTrue((root / ".agent/skills/wannado-wce/SKILL.md").exists())
            self.assertTrue((root / ".claude/skills/wannado-wce/SKILL.md").exists())
            self.assertTrue((root / ".cursor/rules/wannado-wce.mdc").exists())
            self.assertGreaterEqual(len(res), 6)

            agents_txt = (root / "AGENTS.md").read_text(encoding="utf-8")
            self.assertIn("<!-- wannado-wce:start -->", agents_txt)
            self.assertIn("<!-- wannado-wce:end -->", agents_txt)

            # Simulate sibling wannado-drl & wannado-adv installs if present
            if DRL_SRC.exists():
                sys.path.insert(0, str(DRL_SRC))
                try:
                    install_drl = importlib.import_module("wannado.drl.agent_kit").install_agent_kit
                    install_drl(root, ide="all")
                finally:
                    sys.path.remove(str(DRL_SRC))
            if ADV_SRC.exists():
                sys.path.insert(0, str(ADV_SRC))
                try:
                    install_adv = importlib.import_module("wannado.adv.agent_kit").install_agent_kit
                    install_adv(root, ide="all")
                finally:
                    sys.path.remove(str(ADV_SRC))

            # Re-install wannado-wce idempotently and confirm sibling blocks preserved
            install_agent_kit(root, ide="codex")
            final_agents = (root / "AGENTS.md").read_text(encoding="utf-8")
            self.assertEqual(final_agents.count("<!-- wannado-wce:start -->"), 1)
            if DRL_SRC.exists():
                self.assertIn("<!-- wannado-drl:start -->", final_agents)
            if ADV_SRC.exists():
                self.assertIn("<!-- wannado-adv:start -->", final_agents)

    def test_catalog_completeness_standards_and_acyclic_lineage(self) -> None:
        cards = load_catalog()
        self.assertEqual(len(cards), 33)
        self.assertEqual(len(STANDARDS_PROVENANCE), 21)
        self.assertEqual(len(ECOSYSTEM_REGISTRY), 18)
        self.assertEqual(len(MECHANISM_PROVENANCE), 29)
        self.assertGreaterEqual(len(BLOCK_EXPORTS), 33)
        self.assertEqual(set(BLOCK_DEFAULT_MECHANISMS.keys()), {c["id"] for c in cards})

        by_id = {c["id"]: c for c in cards}
        for c in cards:
            for sid in c["standard_ids"]:
                self.assertIn(sid, STANDARDS_PROVENANCE)
            for mid in c["default_mechanisms"]:
                self.assertIn(mid, MECHANISM_PROVENANCE)
            # Check parent DAG acyclicity
            seen = {c["id"]}
            cur = c.get("parent_id")
            while cur:
                self.assertNotIn(cur, seen)
                seen.add(cur)
                self.assertIn(cur, by_id)
                cur = by_id[cur].get("parent_id")

        # Rebuild to verify determinism & JSON write
        with tempfile.TemporaryDirectory() as tmpdir:
            reg = build_registry_json(Path(tmpdir) / "registry.json")
            self.assertEqual(reg["count"], 33)
            self.assertEqual(reg["standards_count"], 21)
            self.assertEqual(reg["ecosystem_tools_count"], 18)

        # Query & lineage filters
        self.assertGreaterEqual(len(query_catalog(family="domain_6g")), 7)
        self.assertGreaterEqual(len(query_catalog(standard_id="gpp_tr38901")), 4)
        self.assertTrue(any(item["family"] == "baselines" for item in get_sota_lineage()))

    def test_zero_hallucination_citation_exporter(self) -> None:
        bib = export_citations(["isac_jsac_channel", "gpp_tr38901", "wu_zhang_passive_ris"], fmt="bibtex")
        self.assertIn("@techreport{3gpp_tr38901", bib)
        self.assertIn("@article{liu2022integrated", bib)
        self.assertIn("@article{wu2019intelligent", bib)
        ieee = export_citations(["itur_m2160"], fmt="ieee")
        self.assertIn("ITU-R", ieee)
        apa = export_citations(["itur_m2160"], fmt="apa")
        self.assertIn("2023", apa)
        with self.assertRaises(KeyError):
            export_citations(["nonexistent_fake_paper_xyz"])


if __name__ == "__main__":
    unittest.main()
