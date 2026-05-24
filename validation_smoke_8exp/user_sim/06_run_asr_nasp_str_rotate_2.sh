#!/usr/bin/env bash
# Sequential ASR NASP structured demos (100h, str only): wav2vec2-base + HuBERT-large.
# Mirrors user_sim/02_run_asr_rotate_4.sh env; NASP has no unstr slot in this repo.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=paths.env
source "${ROOT}/paths.env"

# shellcheck source=lib/asr_conda_activate.sh
source "${ROOT}/lib/asr_conda_activate.sh"
asr_conda_activate || exit 1

export SASPG_REPO WORK_ROOT MAX_STEPS SMOKE_GPU_INDEX SMOKE_ASR_HOST
export CONDA_ENV_NAME_CU121 CONDA_ENV_NAME_CU118 ASR_CUDA_VARIANT
export ASR_SMOKE_OUT_ROOT SMOKE_LOG_ROOT
export SMOKE_SPARSITY_UNSTR SMOKE_SPARSITY_STR SMOKE_EVAL_TEST_ONLY
export SMOKE_TINY_BATCH=1 SMOKE_NUM_TRAIN_SAMPLES=64
export LIBRISPEECH_AUDIO_ROOT="${LIBRISPEECH_ROOT}"
export DATALOADER_NUM_WORKERS=0
export PYTHONUNBUFFERED=1

VAL_ROOT="$(cd "${ROOT}/.." && pwd)"
# shellcheck source=../ASR/asr_smoke_100h_lib.sh
source "${VAL_ROOT}/ASR/asr_smoke_100h_lib.sh"
asr_smoke_host_preflight || exit 1

LOG_DIR="${SMOKE_LOG_ROOT:-${WORK_ROOT}/logs}/asr_nasp_str"
mkdir -p "${LOG_DIR}" "${ASR_SMOKE_OUT_ROOT}"
echo "========== ASR NASP str (host=$(hostname -s), gpu=${SMOKE_GPU_INDEX}, out=${ASR_SMOKE_OUT_ROOT}) =========="

_fail=0
_run() {
  local tag="$1"
  local script="$2"
  local log="${LOG_DIR}/${tag}.log"
  echo ""
  echo ">>>>>>>>>> ASR NASP str ${tag} <<<<<<<<<<"
  if bash "${script}" >"${log}" 2>&1; then
    echo "[OK] ${tag} log=${log}"
  else
    echo "[FAIL] ${tag} log=${log}" >&2
    tail -30 "${log}" >&2 || true
    _fail=$((_fail + 1))
  fi
}

_run "nasp_str_w2v"   "${VAL_ROOT}/ASR/nasp_str_wav2vec2_base_100h.sh"
_run "nasp_str_hubert" "${VAL_ROOT}/ASR/nasp_str_hubert_large_100h.sh"

echo ""
echo "========== ASR NASP str rotate done: fail=${_fail}/2 =========="
date -Iseconds >"${LOG_DIR}/.rotate_complete"
echo "fail=${_fail}/2" >>"${LOG_DIR}/.rotate_complete"
[[ "${_fail}" -eq 0 ]]
