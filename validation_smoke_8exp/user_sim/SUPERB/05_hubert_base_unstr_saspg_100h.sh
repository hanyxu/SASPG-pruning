#!/usr/bin/env bash
# SUPERB smoke: hubert-base, SASPG unstr, LibriSpeech train100 — wait bdda7 GPU 2 idle, then run.
set -euo pipefail
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/lib/superb_run_one_smoke.sh" \
  "hubert_base_100_unstr_saspg" "$@"
