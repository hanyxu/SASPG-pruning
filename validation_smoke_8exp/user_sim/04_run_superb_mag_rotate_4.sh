#!/usr/bin/env bash
# Sequentially run 4 SUPERB MAG smoke experiments (hubert-base/large × unstr/str).
# Mirrors 03_run_superb_rotate_4.sh; MAG skips SASPG distill (prune_mag → final_distill_mag only).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUPERB_MAG="${ROOT}/SUPERB"
# shellcheck source=paths.env
source "${ROOT}/paths.env"
# shellcheck source=lib/gpu_wait_smoke.sh
source "${ROOT}/lib/gpu_wait_smoke.sh"
# shellcheck source=lib/asr_conda_activate.sh
source "${ROOT}/lib/asr_conda_activate.sh"
smoke_conda_activate || exit 1

export SASPG_REPO WORK_ROOT SUPERB_MAG_WORK_ROOT SMOKE_GPU_INDEX SUPERB_CONDA_ENV CONDA_ENV_NAME SMOKE_DEBUG_LOG
export SUPERB_SMOKE_DRY_RUN=0
export WAIT_GPU_EACH_TASK="${WAIT_GPU_EACH_TASK:-1}"
export CUDA_VISIBLE_DEVICES="${SMOKE_GPU_INDEX}"

LOG_DIR="${SUPERB_MAG_WORK_ROOT}/logs/superb_mag"
mkdir -p "${LOG_DIR}"

echo "========== SUPERB MAG rotate 4 (gpu=${SMOKE_GPU_INDEX}) =========="
echo "host=$(hostname) SUPERB_CONDA_ENV=${SUPERB_CONDA_ENV}"
echo "SUPERB_MAG_WORK_ROOT=${SUPERB_MAG_WORK_ROOT} LOG_DIR=${LOG_DIR}"

_fail=0
_run() {
  local tag="$1"
  local script="$2"
  local log="${LOG_DIR}/${tag}.log"
  echo ""
  echo ">>>>>>>>>> SUPERB MAG ${tag} <<<<<<<<<<"
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

_run "hubert_base_100_unstr_mag"  "${SUPERB_MAG}/09_hubert_base_unstr_mag_100h.sh"
_run "hubert_base_100_str_mag"    "${SUPERB_MAG}/10_hubert_base_str_mag_100h.sh"
_run "hubert_large_100_unstr_mag" "${SUPERB_MAG}/11_hubert_large_unstr_mag_100h.sh"
_run "hubert_large_100_str_mag"   "${SUPERB_MAG}/12_hubert_large_str_mag_100h.sh"

echo ""
echo "========== SUPERB MAG rotate done: fail=${_fail}/4 =========="
date -Iseconds >>"${SMOKE_DEBUG_LOG}"
echo "SUPERB MAG rotate done fail=${_fail}/4 host=$(hostname) gpu=${SMOKE_GPU_INDEX}" >>"${SMOKE_DEBUG_LOG}"
[[ "${_fail}" -eq 0 ]]
