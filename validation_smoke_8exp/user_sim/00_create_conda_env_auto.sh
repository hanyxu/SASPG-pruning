#!/usr/bin/env bash
# Create unified test_saspg env on *this* GPU node (driver 12.x -> cu121, 11.x -> cu118).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=paths.env
source "${ROOT}/paths.env"
# shellcheck source=lib/asr_conda_activate.sh
source "${ROOT}/lib/asr_conda_activate.sh"

_chosen="$(asr_resolve_conda_env_name)" || exit 1
echo "[auto] NVIDIA driver -> conda env name: ${_chosen}"

if [[ "${_chosen}" == "${CONDA_ENV_NAME_CU118:-test_saspg_cu118}" ]]; then
  exec bash "${ROOT}/00_create_conda_env_cu118.sh"
else
  export CONDA_ENV_NAME="${CONDA_ENV_NAME_CU121:-test_saspg}"
  exec bash "${ROOT}/00_create_conda_env.sh"
fi
