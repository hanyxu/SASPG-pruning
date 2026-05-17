"""Experiment matrix + recipe metadata (family, upstream_mode) derived from method."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from core.paths import WORKSPACE, experiments_csv_path, release_root

ROOT = release_root()
EXPERIMENTS_CSV = experiments_csv_path()
FAMILY_DEFAULTS = ROOT / "recipes" / "family_defaults.json"

METHOD_TO_FAMILY: dict[str, str] = {
    "native": "native",
    "str_saspg": "saspg",
    "unstr_saspg": "saspg",
    "str_magnitude": "magnitude",
    "unstr_magnitude": "magnitude",
    "str_dphubert": "dphubert",
    "unstr_dphubert": "dphubert",
}


def default_upstream_mode(method: str) -> str:
    if "magnitude" in method:
        return "prune_then_ft"
    return "distill_prune_ft"


def _load_family_defaults() -> dict[str, Any]:
    if not FAMILY_DEFAULTS.exists():
        return {}
    with FAMILY_DEFAULTS.open("r", encoding="utf-8") as f:
        return json.load(f) or {}


def enrich_row(row: dict[str, str]) -> dict[str, str]:
    method = row.get("method", "")
    family = METHOD_TO_FAMILY.get(method, "unknown")
    row = dict(row)
    row["family"] = family
    # Allow YAML override per-family default upstream_mode (future)
    defaults = _load_family_defaults().get(family, {})
    um = defaults.get("upstream_mode_default") if isinstance(defaults, dict) else None
    row["upstream_mode"] = um if um else default_upstream_mode(method)
    return row


def load_experiment_rows() -> list[dict[str, str]]:
    with EXPERIMENTS_CSV.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [enrich_row(r) for r in rows]


def resolve_upstream_script(row: dict[str, str]) -> Path | None:
    if row.get("source_repo") == "TODO" or row.get("entry_script") == "TODO":
        return None
    return Path(WORKSPACE) / row["source_repo"] / row["entry_script"]
