#!/usr/bin/env bash
# Sequential ASR MAG demos (100h): wav2vec2-base + HuBERT-large × unstr/str.
# Mirrors 02_run_asr_rotate_4.sh layout for SASPG: 01 unstr w2v, 02 str w2v, 03 unstr hubert, 04 str hubert.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=paths.env
source "${ROOT}/paths.env"

# shellcheck source=lib/asr_conda_activate.sh
source "${ROOT}/lib/asr_conda_activate.sh"
asr_conda_activate || exit 1

export SASPG_REPO WORK_ROOT MAX_STEPS SMOKE_GPU_INDEX MAG_PRUNE_STEPS
export CONDA_ENV_NAME_CU121 CONDA_ENV_NAME_CU118 ASR_CUDA_VARIANT
export ASR_SMOKE_OUT_ROOT SMOKE_LOG_ROOT
export SMOKE_SPARSITY_UNSTR SMOKE_SPARSITY_STR SMOKE_EVAL_TEST_ONLY
export SMOKE_TINY_BATCH=1 SMOKE_NUM_TRAIN_SAMPLES=64
export LIBRISPEECH_AUDIO_ROOT="${LIBRISPEECH_ROOT}"
export DATALOADER_NUM_WORKERS=0
export PYTHONUNBUFFERED=1

VAL_ROOT="$(cd "${ROOT}/.." && pwd)"
LOG_DIR="${SMOKE_LOG_ROOT:-${WORK_ROOT}/logs}/asr_mag"
mkdir -p "${LOG_DIR}" "${ASR_SMOKE_OUT_ROOT}"

_fail=0
_run() {
  local tag="$1"
  local script="$2"
  local log="${LOG_DIR}/${tag}.log"
  echo ""
  echo ">>>>>>>>>> ASR MAG ${tag} <<<<<<<<<<"
  if bash "${script}" >"${log}" 2>&1; then
    echo "[OK] ${tag} log=${log}"
  else
    echo "[FAIL] ${tag} log=${log}" >&2
    tail -30 "${log}" >&2 || true
    _fail=$((_fail + 1))
  fi
}

_run "01_wav2vec2_base_unstr" "${VAL_ROOT}/ASR/mag_unstr_wav2vec2_base_100h.sh"
_run "02_wav2vec2_base_str"   "${VAL_ROOT}/ASR/mag_str_wav2vec2_base_100h.sh"
_run "03_hubert_large_unstr"  "${VAL_ROOT}/ASR/mag_unstr_hubert_large_100h.sh"
_run "04_hubert_large_str"    "${VAL_ROOT}/ASR/mag_str_hubert_large_100h.sh"

echo ""
echo "========== ASR MAG rotate done: fail=${_fail}/4 =========="
date -Iseconds >"${LOG_DIR}/.rotate_complete"
echo "fail=${_fail}/4" >>"${LOG_DIR}/.rotate_complete"
[[ "${_fail}" -eq 0 ]]
