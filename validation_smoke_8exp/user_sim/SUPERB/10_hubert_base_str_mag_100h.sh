#!/usr/bin/env bash
# SUPERB smoke: hubert-base, MAG str (head + FFN channel top-k), LibriSpeech train100.
set -euo pipefail
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/lib/superb_run_one_mag_smoke.sh" \
  "hubert_base_100_str_magnitude" "$@"
