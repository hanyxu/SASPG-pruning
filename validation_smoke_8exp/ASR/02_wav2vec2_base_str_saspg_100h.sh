#!/usr/bin/env bash
# ASR smoke #2: wav2vec2-base, SASPG structured, LibriSpeech 100h
set -euo pipefail
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/asr_run_one.sh" \
  "02_wav2vec2_base_str_saspg_100h" str w2v "$@"
