#!/usr/bin/env python3
"""Batch runner: `python3 -m core smoke48` or `bash ./run_smoke48.sh`."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.cli import main


if __name__ == "__main__":
    sys.exit(main(["smoke48"] + sys.argv[1:]))
