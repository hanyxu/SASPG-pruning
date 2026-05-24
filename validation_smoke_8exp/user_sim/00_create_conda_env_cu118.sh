#!/usr/bin/env bash
# Unified conda test_saspg_cu118 (PyTorch cu118, driver 11.x): ASR + SUPERB smoke.
# Same Python deps as 00_create_conda_env.sh; only the torch wheel index differs.
# Prefer: bash 00_create_conda_env_auto.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=paths.env
source "${ROOT}/paths.env"

export CONDA_ENV_NAME="${CONDA_ENV_NAME_CU118:-test_saspg_cu118}"

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found on PATH" >&2
  exit 1
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

echo "[host] $(hostname)  glibc: $(ldd --version 2>/dev/null | head -1 || echo unknown)"
echo "[conda] target env=${CONDA_ENV_NAME} (PyTorch cu118 for driver 11.x)"

_env_python_ok() {
  conda run -n "${CONDA_ENV_NAME}" python -c 'import sys' >/dev/null 2>&1
}

_recreate_env() {
  if conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV_NAME}"; then
    echo "[conda] removing ${CONDA_ENV_NAME} (recreate on this node) ..."
    conda env remove -y -n "${CONDA_ENV_NAME}"
  fi
  echo "[conda] creating ${CONDA_ENV_NAME} (python=3.10) on $(hostname) ..."
  conda create -y -n "${CONDA_ENV_NAME}" python=3.10 pip
}

if conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV_NAME}"; then
  if [[ "${RECREATE_CONDA:-0}" == "1" ]]; then
    _recreate_env
  elif _env_python_ok; then
    echo "[conda] env ${CONDA_ENV_NAME} exists and runs on this node — reuse"
  else
    echo "[conda] env ${CONDA_ENV_NAME} exists but Python fails here (glibc mismatch?)"
    _recreate_env
  fi
else
  _recreate_env
fi

conda activate "${CONDA_ENV_NAME}"

_torch_has_cuda() {
  python - <<'PY'
import sys
try:
    import torch
except ImportError:
    sys.exit(1)
if torch.version.cuda is None:
    sys.exit(2)
if not torch.cuda.is_available():
    sys.exit(3)
sys.exit(0)
PY
}

_install_pip_torch_cuda() {
  echo "[pip] torch + torchaudio (cu118, download.pytorch.org) ..."
  pip uninstall -y torch torchaudio 2>/dev/null || true
  conda remove -y pytorch pytorch-cuda 2>/dev/null || true
  pip install "torch==2.5.1" "torchaudio==2.5.1" \
    --index-url https://download.pytorch.org/whl/cu118
}

echo "[conda] numpy / scipy / libsndfile ..."
conda install -y -c conda-forge \
  "numpy>=1.24,<2" \
  scipy \
  pyyaml \
  tqdm \
  libsndfile

_torch_rc=0
_torch_has_cuda || _torch_rc=$?
if [[ "${_torch_rc}" -ne 0 ]]; then
  _install_pip_torch_cuda
fi
_torch_has_cuda || {
  echo "ERROR: torch has no working CUDA on $(hostname)." >&2
  exit 1
}

# shellcheck source=lib/smoke_pip_install.sh
source "${ROOT}/lib/smoke_pip_install.sh"
smoke_pip_install "${ROOT}/requirements-smoke.txt"

_torch_has_cuda || _install_pip_torch_cuda

smoke_verify_python_deps
echo "[OK] unified smoke env ${CONDA_ENV_NAME} (ASR+SUPERB) on $(hostname)."
