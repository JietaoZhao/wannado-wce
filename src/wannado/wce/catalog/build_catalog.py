"""Build and query the wannado.wce standards, blocks, and ecosystem knowledge graph."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any

from wannado.wce.core.provenance import BLOCK_DEFAULT_MECHANISMS, MECHANISM_PROVENANCE, STANDARDS_PROVENANCE
from wannado.wce.core.types import WCEMeta
from wannado.wce.standards.ecosystem import ECOSYSTEM_REGISTRY

WCE_ROOT = Path(__file__).resolve().parent.parent
BLOCKS_DIR = WCE_ROOT / "blocks"
REGISTRY_PATH = Path(__file__).resolve().parent / "registry.json"


def discover_block_metas() -> list[dict[str, Any]]:
    """Scan all CleanRL-style single-file wireless blocks and extract verified WCEMeta cards."""
    cards: list[dict[str, Any]] = []
    for file_path in sorted(BLOCKS_DIR.rglob("*.py")):
        if file_path.name.startswith("_"):
            continue
        rel = file_path.relative_to(WCE_ROOT).with_suffix("")
        module_name = "wannado.wce." + ".".join(rel.parts)
        module = importlib.import_module(module_name)
        meta = getattr(module, "META", None)
        if not isinstance(meta, WCEMeta):
            raise TypeError(f"Block module '{module_name}' must define META as a WCEMeta instance.")
        card = meta.to_dict()
        if not card.get("module_path"):
            card["module_path"] = module_name
        if not card.get("default_mechanisms"):
            card["default_mechanisms"] = list(BLOCK_DEFAULT_MECHANISMS.get(card["id"], ()))
        card["standard_citations"] = [
            STANDARDS_PROVENANCE[sid].to_dict()
            for sid in card["standard_ids"]
            if sid in STANDARDS_PROVENANCE
        ]
        card["mechanism_citations"] = [
            MECHANISM_PROVENANCE[mid].to_dict()
            for mid in card["default_mechanisms"]
            if mid in MECHANISM_PROVENANCE
        ]
        cards.append(card)

    ids = {c["id"] for c in cards}
    for c in cards:
        pid = c.get("parent_id")
        if pid is not None and pid not in ids:
            raise ValueError(f"Block '{c['id']}' declares unknown parent_id='{pid}'.")
        for sid in c.get("standard_ids", []):
            if sid not in STANDARDS_PROVENANCE:
                raise ValueError(f"Block '{c['id']}' declares unknown standard_id='{sid}'.")

    cards.sort(key=lambda c: (c["family"], c["citation"]["year"], c["id"]))
    return cards


def build_registry_json(output_path: Path = REGISTRY_PATH) -> dict[str, Any]:
    """Generate ``registry.json`` directly from source module metadata, standards, and ecosystem tools."""
    cards = discover_block_metas()
    payload = {
        "schema_version": "1.0",
        "count": len(cards),
        "standards_count": len(STANDARDS_PROVENANCE),
        "mechanisms_count": len(MECHANISM_PROVENANCE),
        "ecosystem_tools_count": len(ECOSYSTEM_REGISTRY),
        "blocks": cards,
        "standards": [s.to_dict() for s in STANDARDS_PROVENANCE.values()],
        "ecosystem_tools": [t.to_dict() for t in ECOSYSTEM_REGISTRY.values()],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def load_catalog(rebuild_if_missing: bool = True) -> list[dict[str, Any]]:
    """Load the serialized wireless block knowledge graph."""
    if REGISTRY_PATH.exists():
        data = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        return list(data.get("blocks", []))
    if rebuild_if_missing:
        return discover_block_metas()
    raise FileNotFoundError(f"Catalog registry not found at {REGISTRY_PATH}.")


def query_catalog(
    *,
    family: str | None = None,
    era: str | None = None,
    tier: str | None = None,
    standard_id: str | None = None,
    output_kind: str | None = None,
    supports_torch_complex: bool | None = None,
) -> list[dict[str, Any]]:
    """Filter wireless environment blocks by family, era, tier, standard, or output modality."""
    cards = load_catalog()
    results: list[dict[str, Any]] = []
    for card in cards:
        if family is not None and card["family"] != family:
            continue
        if era is not None and card["era"] != era:
            continue
        if tier is not None and card["tier"] != tier:
            continue
        if output_kind is not None and card["output_kind"] != output_kind:
            continue
        if supports_torch_complex is not None and card["supports_torch_complex"] != supports_torch_complex:
            continue
        if standard_id is not None and standard_id not in card["standard_ids"]:
            continue
        results.append(card)
    return results


def get_taxonomy_tree() -> dict[str, list[dict[str, Any]]]:
    """Group all wireless blocks cleanly by family in classic-to-frontier progression."""
    tree: dict[str, list[dict[str, Any]]] = {}
    for card in load_catalog():
        tree.setdefault(card["family"], []).append(card)
    return tree


def main() -> None:
    parser = argparse.ArgumentParser(description="Build or query the wannado.wce catalog.")
    parser.add_argument("--build", action="store_true", help="Rebuild registry.json from block modules.")
    parser.add_argument("--family", type=str, default=None)
    parser.add_argument("--era", type=str, default=None)
    parser.add_argument("--tier", type=str, default=None)
    parser.add_argument("--standard", type=str, default=None)
    args = parser.parse_args()

    if args.build or not REGISTRY_PATH.exists():
        payload = build_registry_json()
        print(
            f"Built {REGISTRY_PATH} with {payload['count']} blocks, "
            f"{payload['standards_count']} standards, {payload['ecosystem_tools_count']} ecosystem tools."
        )

    matches = query_catalog(
        family=args.family,
        era=args.era,
        tier=args.tier,
        standard_id=args.standard,
    )
    print(json.dumps(matches, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
