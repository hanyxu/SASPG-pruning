"""Workspace / ROOT resolution for standalone GitHub clones."""

from __future__ import annotations

import os
from pathlib import Path


def release_root() -> Path:
    """Directory containing core/, configs/, recipes/, upstream_* dirs."""
    return Path(__file__).resolve().parent.parent


def experiments_csv_path() -> Path:
    return release_root() / "configs" / "experiments.csv"


def detect_workspace() -> Path:
    """Training script CWD resolution for merged standalone release."""
    env = os.environ.get("SASPG_WORKSPACE", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    root = release_root()
    return root


WORKSPACE = detect_workspace()
