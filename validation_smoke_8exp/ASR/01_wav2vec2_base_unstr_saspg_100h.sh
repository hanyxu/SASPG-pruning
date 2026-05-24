#!/usr/bin/env bash
# ASR smoke #1: wav2vec2-base, SASPG unstructured, LibriSpeech 100h
set -euo pipefail
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/asr_run_one.sh" \
  "01_wav2vec2_base_unstr_saspg_100h" unstr w2v "$@"
