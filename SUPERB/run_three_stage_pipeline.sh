#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# Canonical name for this release flow:
# stage1+stage2 upstream training -> stage3 downstream SUPERB.
bash "$ROOT/run_two_stage_pipeline.sh" "$@"
