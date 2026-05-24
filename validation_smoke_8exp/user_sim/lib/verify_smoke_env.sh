#!/usr/bin/env bash
# Verify active conda env supports ASR + SUPERB smoke. Usage: bash lib/verify_smoke_env.sh
set -euo pipefail

_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_usim_root="$(cd "${_script_dir}/.." && pwd)"
# shellcheck source=../paths.env
source "${_usim_root}/paths.env"
# shellcheck source=asr_conda_activate.sh
source "${_script_dir}/asr_conda_activate.sh"

_fail=0
_check() {
  if eval "$2"; then
    echo "[OK] $1"
  else
    echo "[FAIL] $1" >&2
    _fail=$((_fail + 1))
  fi
}

if [[ -n "${CONDA_DEFAULT_ENV:-}" ]]; then
  echo "[verify] using active conda env=${CONDA_DEFAULT_ENV}"
  export CONDA_ENV_NAME="${CONDA_DEFAULT_ENV}"
else
  smoke_conda_activate || exit 1
fi

echo "host=$(hostname -s) env=${CONDA_ENV_NAME} driver=$(asr_nvidia_driver_version 2>/dev/null || echo n/a)"
_check "torch CUDA" 'python -c "import torch; assert torch.version.cuda and torch.cuda.is_available()"'
_check "torchaudio" 'python -c "import torchaudio"'
_check "transformers" 'python -c "import transformers"'
_check "pytorch-lightning (SUPERB)" 'python -c "import pytorch_lightning as pl; from lightning_lite.utilities.rank_zero import _get_rank"'
_check "datasets (ASR)" 'python -c "import datasets"'

[[ "${_fail}" -eq 0 ]] && echo "[OK] smoke env ready for ASR + SUPERB" || {
  echo "Fix: bash 00_create_conda_env_auto.sh  (on the GPU node)" >&2
  exit 1
}
