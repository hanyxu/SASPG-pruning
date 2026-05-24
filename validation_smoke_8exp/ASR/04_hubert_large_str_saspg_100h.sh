#!/usr/bin/env bash
# ASR smoke #4: HuBERT-large, SASPG structured, LibriSpeech 100h
set -euo pipefail
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/asr_run_one.sh" \
  "04_hubert_large_str_saspg_100h" str hubert "$@"
