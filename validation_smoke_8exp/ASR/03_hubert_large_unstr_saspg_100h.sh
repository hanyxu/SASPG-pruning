#!/usr/bin/env bash
# ASR smoke #3: HuBERT-large, SASPG unstructured, LibriSpeech 100h
set -euo pipefail
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/asr_run_one.sh" \
  "03_hubert_large_unstr_saspg_100h" unstr hubert "$@"
