#!/usr/bin/env bash
# Run all 8 SASPG smoke experiments (4 ASR + 4 SUPERB) on LibriSpeech 100h.
#
# ASR: real short training (default MAX_STEPS=500).
# SUPERB: matrix validate + pipeline dry-run by default (SUPERB_SMOKE_DRY_RUN=1).
#   Set SUPERB_SMOKE_DRY_RUN=0 for real upstream launch (training blocks must be enabled).
#
# Example:
#   cd /path/to/SASPG-pruning/validation_smoke_8exp
#   export LIBRISPEECH_AUDIO_ROOT=/data/LibriSpeech
#   export DPHuBERT_PRETRAINED_DIR=/data/pretrained
#   export DPHuBERT_TSV_DIR=/data/librispeech_tsv
#   export SMOKE_GPU_INDEX=0
#   bash ./run_all_8.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common_env.sh
source "${ROOT}/common_env.sh"

chmod +x "${ROOT}"/ASR/*.sh "${ROOT}"/SUPERB/*.sh 2>/dev/null || true

echo "========== SASPG 8-exp smoke (100h) =========="
echo "ASR_ROOT=${ASR_ROOT}"
echo "SUPERB_ROOT=${SUPERB_ROOT}"
echo "MAX_STEPS=${MAX_STEPS}  SUPERB_SMOKE_DRY_RUN=${SUPERB_SMOKE_DRY_RUN}"
echo

_fail=0
_run() {
  local name="$1"
  shift
  echo ""
  echo ">>>>>>>>>> ${name} <<<<<<<<<<"
  if "$@"; then
    echo "[OK] ${name}"
  else
    echo "[FAIL] ${name}" >&2
    _fail=$((_fail + 1))
  fi
}

_run "ASR-01 wav2vec2-base unstr"  bash "${ROOT}/ASR/01_wav2vec2_base_unstr_saspg_100h.sh"
_run "ASR-02 wav2vec2-base str"    bash "${ROOT}/ASR/02_wav2vec2_base_str_saspg_100h.sh"
_run "ASR-03 hubert-large unstr"   bash "${ROOT}/ASR/03_hubert_large_unstr_saspg_100h.sh"
_run "ASR-04 hubert-large str"     bash "${ROOT}/ASR/04_hubert_large_str_saspg_100h.sh"

_run "SUPERB-05 hubert-base unstr"  bash "${ROOT}/SUPERB/05_hubert_base_unstr_saspg_100h.sh"
_run "SUPERB-06 hubert-base str"    bash "${ROOT}/SUPERB/06_hubert_base_str_saspg_100h.sh"
_run "SUPERB-07 hubert-large unstr" bash "${ROOT}/SUPERB/07_hubert_large_unstr_saspg_100h.sh"
_run "SUPERB-08 hubert-large str"   bash "${ROOT}/SUPERB/08_hubert_large_str_saspg_100h.sh"

echo ""
echo "========== SUMMARY: fail=${_fail} / 8 =========="
[[ "$_fail" -eq 0 ]]
