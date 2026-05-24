#!/usr/bin/env bash
# SUPERB smoke: hubert-large, MAG str, LibriSpeech train100.
set -euo pipefail
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/lib/superb_run_one_mag_smoke.sh" \
  "hubert_large_100_str_magnitude" "$@"
