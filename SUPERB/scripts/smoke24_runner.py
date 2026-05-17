#!/usr/bin/env python3
"""Backward-compatible entry: delegates to scripts/smoke48_runner.py."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

sys.exit(runpy.run_path(str(Path(__file__).with_name("smoke48_runner.py")), run_name="__main__") or 0)
