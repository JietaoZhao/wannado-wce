"""Knowledge-graph catalog discovery & query API for wannado.wce."""

from wannado.wce.catalog.build_catalog import (
    BLOCKS_DIR,
    REGISTRY_PATH,
    WCE_ROOT,
    build_registry_json,
    discover_block_metas,
    get_taxonomy_tree,
    load_catalog,
    query_catalog,
)

__all__ = [
    "BLOCKS_DIR",
    "REGISTRY_PATH",
    "WCE_ROOT",
    "build_registry_json",
    "discover_block_metas",
    "get_taxonomy_tree",
    "load_catalog",
    "query_catalog",
]
