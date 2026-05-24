#!/usr/bin/env bash
# ASR demo: HuBERT-large, MAG structured (channel top-k), LibriSpeech 100h
set -euo pipefail
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/asr_run_mag_str_one.sh" \
  "mag_str_hubert_large_100h" hubert "$@"
