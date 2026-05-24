#!/usr/bin/env bash
# Unified conda env test_saspg (PyTorch cu121): ASR (HF/transformers) + SUPERB (PyTorch Lightning).
# Create on the GPU node where you train (glibc must match). Prefer: bash 00_create_conda_env_auto.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=paths.env
source "${ROOT}/paths.env"

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found on PATH" >&2
  exit 1
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

echo "[host] $(hostname)  glibc: $(ldd --version 2>/dev/null | head -1 || echo unknown)"

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
    sys.exit(2)  # CPU-only wheel (common if pip overwrote conda)
if not torch.cuda.is_available():
    sys.exit(3)  # CUDA build but driver/runtime not visible
sys.exit(0)
PY
}

_install_pip_torch_cuda() {
  # On some cluster nodes `conda install pytorch-cuda` still resolves cpu_openblas
  # builds from pkgs/main / conda-forge. Official cu121 wheels are reliable.
  echo "[pip] torch + torchaudio (cu121, download.pytorch.org) ..."
  pip uninstall -y torch torchaudio 2>/dev/null || true
  conda remove -y pytorch pytorch-cuda 2>/dev/null || true
  pip install "torch==2.5.1" "torchaudio==2.5.1" \
    --index-url https://download.pytorch.org/whl/cu121
}

echo "[conda] numpy / scipy / libsndfile (no pip compile) ..."
conda install -y -c conda-forge \
  "numpy>=1.24,<2" \
  scipy \
  pyyaml \
  tqdm \
  libsndfile

# CUDA torch must come after conda base libs; never `conda install pytorch` here (CPU builds).
_torch_rc=0
_torch_has_cuda || _torch_rc=$?
if [[ "${_torch_rc}" -ne 0 ]]; then
  _install_pip_torch_cuda
fi
_torch_has_cuda || {
  echo "ERROR: torch has no working CUDA on $(hostname)." >&2
  echo "  python -c \"import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())\"" >&2
  exit 1
}

# ASR/SUPERB smoke do not import pandas — do not pip-install it (pandas 2.3 sdist fails on GCC 4.8)
# shellcheck source=lib/smoke_pip_install.sh
source "${ROOT}/lib/smoke_pip_install.sh"
smoke_pip_install "${ROOT}/requirements-smoke.txt"

# pytorch-lightning / accelerate must not downgrade torch to CPU wheels
_torch_has_cuda || _install_pip_torch_cuda

smoke_verify_python_deps
echo "[OK] unified smoke env ${CONDA_ENV_NAME} (ASR+SUPERB) on $(hostname). Activate: conda activate ${CONDA_ENV_NAME}"
