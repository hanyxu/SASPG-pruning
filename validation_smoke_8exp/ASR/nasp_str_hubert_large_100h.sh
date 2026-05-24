#!/usr/bin/env bash
# ASR demo: HuBERT-large, NASP structured (7-tier Gumbel), LibriSpeech 100h (str only)
set -euo pipefail
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/asr_run_nasp_str_one.sh" \
  "nasp_str_hubert_large_100h" hubert "$@"
