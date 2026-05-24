#!/usr/bin/env bash
# Sequentially run 4 SUPERB SASPG smoke experiments; wait for SMOKE_GPU_INDEX idle before each job.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUPERB_SCRIPTS="${ROOT}/SUPERB"
# shellcheck source=paths.env
source "${ROOT}/paths.env"
# shellcheck source=lib/gpu_wait_smoke.sh
source "${ROOT}/lib/gpu_wait_smoke.sh"
# shellcheck source=lib/asr_conda_activate.sh
source "${ROOT}/lib/asr_conda_activate.sh"
smoke_conda_activate || exit 1

export SASPG_REPO WORK_ROOT SUPERB_SASPG_WORK_ROOT SMOKE_GPU_INDEX SUPERB_CONDA_ENV CONDA_ENV_NAME SMOKE_DEBUG_LOG
export SUPERB_SMOKE_DRY_RUN=0
export WAIT_GPU_EACH_TASK="${WAIT_GPU_EACH_TASK:-1}"
export CUDA_VISIBLE_DEVICES="${SMOKE_GPU_INDEX}"

LOG_DIR="${SUPERB_SASPG_WORK_ROOT}/logs/superb"
mkdir -p "${LOG_DIR}"

echo "========== SUPERB rotate 4 (gpu=${SMOKE_GPU_INDEX}, idle<=$((GPU_MEM_IDLE_MAX_MB + 1)) MiB) =========="
echo "host=$(hostname) SUPERB_CONDA_ENV=${SUPERB_CONDA_ENV}"
echo "SUPERB_SASPG_WORK_ROOT=${SUPERB_SASPG_WORK_ROOT} LOG_DIR=${LOG_DIR}"

_fail=0
_run() {
  local tag="$1"
  local script="$2"
  local log="${LOG_DIR}/${tag}.log"
  echo ""
  echo ">>>>>>>>>> SUPERB ${tag} <<<<<<<<<<"
  if [[ "${WAIT_GPU_EACH_TASK}" == "1" ]]; then
    wait_for_idle_gpu "${LOG_DIR}/gpu_wait_${tag}.log"
    export WAIT_GPU_EACH_TASK=0
  fi
  if bash "${script}" >"${log}" 2>&1; then
    echo "[OK] ${tag} log=${log}"
  else
    echo "[FAIL] ${tag} log=${log}" >&2
    tail -40 "${log}" >&2 || true
    _fail=$((_fail + 1))
  fi
}

_run "hubert_base_100_unstr_saspg"  "${SUPERB_SCRIPTS}/05_hubert_base_unstr_saspg_100h.sh"
_run "hubert_base_100_str_saspg"    "${SUPERB_SCRIPTS}/06_hubert_base_str_saspg_100h.sh"
_run "hubert_large_100_unstr_saspg" "${SUPERB_SCRIPTS}/07_hubert_large_unstr_saspg_100h.sh"
_run "hubert_large_100_str_saspg"   "${SUPERB_SCRIPTS}/08_hubert_large_str_saspg_100h.sh"

echo ""
echo "========== SUPERB rotate done: fail=${_fail}/4 =========="
date -Iseconds >>"${SMOKE_DEBUG_LOG}"
echo "SUPERB rotate done fail=${_fail}/4 host=$(hostname) gpu=${SMOKE_GPU_INDEX}" >>"${SMOKE_DEBUG_LOG}"
[[ "${_fail}" -eq 0 ]]
