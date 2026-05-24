#!/usr/bin/env bash
# ASR demo: wav2vec2-base, MAG structured (channel top-k), LibriSpeech 100h
set -euo pipefail
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/asr_run_mag_str_one.sh" \
  "mag_str_wav2vec2_base_100h" w2v "$@"
