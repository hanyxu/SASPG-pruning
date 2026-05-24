#!/usr/bin/env bash
# SUPERB smoke: hubert-large, SASPG unstr, LibriSpeech train100 — wait bdda7 GPU 2 idle, then run.
set -euo pipefail
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/lib/superb_run_one_smoke.sh" \
  "hubert_large_100_unstr_saspg" "$@"
