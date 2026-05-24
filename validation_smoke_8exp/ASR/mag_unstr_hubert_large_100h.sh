#!/usr/bin/env bash
# ASR demo: HuBERT-large, MAG unstructured (mask×weight), LibriSpeech 100h
set -euo pipefail
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/asr_run_mag_unstr_one.sh" \
  "mag_unstr_hubert_large_100h" hubert "$@"
