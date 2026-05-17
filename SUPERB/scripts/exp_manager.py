#!/usr/bin/env python3
"""Compatibility wrapper: prefer `python3 -m core experiments ...`."""

import sys

from core.experiments import main_experiments

if __name__ == "__main__":
    sys.exit(main_experiments(sys.argv[1:]))
