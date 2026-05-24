#!/usr/bin/env bash
# Run one SUPERB 100h MAG smoke: wait for idle GPU, then prune_mag + final_distill_mag.
# Usage: superb_run_one_mag_smoke.sh <exp_id>
set -euo pipefail

_exp_id="${1:?exp_id}"

_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_usim_root="$(cd "${_script_dir}/.." && pwd)"
# shellcheck source=../paths.env
source "${_usim_root}/paths.env"
# shellcheck source=gpu_wait_smoke.sh
source "${_script_dir}/gpu_wait_smoke.sh"

export SUPERB_SMOKE_DRY_RUN="${SUPERB_SMOKE_DRY_RUN:-0}"
export CUDA_VISIBLE_DEVICES="${SMOKE_GPU_INDEX}"

_log="${SUPERB_MAG_WORK_ROOT}/logs/superb_mag/${_exp_id}.log"
mkdir -p "$(dirname "${_log}")"

if [[ "${WAIT_GPU_EACH_TASK:-1}" == "1" ]]; then
  wait_for_idle_gpu "${SUPERB_MAG_WORK_ROOT}/logs/superb_mag/gpu_wait_${_exp_id}.log"
fi

echo "[SUPERB MAG smoke] exp_id=${_exp_id} gpu=${SMOKE_GPU_INDEX} conda=${SUPERB_CONDA_ENV} log=${_log}"
bash "${_script_dir}/superb_smoke_mag_from_launcher.sh" "${_exp_id}" 2>&1 | tee "${_log}"
exit "${PIPESTATUS[0]}"
