#!/usr/bin/env bash
# ASR demo: wav2vec2-base, NASP structured (7-tier Gumbel), LibriSpeech 100h (str only)
set -euo pipefail
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/asr_run_nasp_str_one.sh" \
  "nasp_str_wav2vec2_base_100h" w2v "$@"
