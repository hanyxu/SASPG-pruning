#!/usr/bin/env bash
# Install / refresh Python deps for unified test_saspg (ASR + SUPERB). Source with env activated.
# Optional: SMOKE_TORCH_INDEX_URL (cu121 default in 00_create_conda_env.sh, cu118 in *_cu118.sh)

smoke_pip_install() {
  local _req="${1:?requirements-smoke.txt path}"

  echo "[pip] smoke deps (ASR + SUPERB) from ${_req} ..."
  pip install -U pip wheel setuptools
  export PIP_ONLY_BINARY=":all:"
  if ! pip install -r "${_req}"; then
    echo "[pip] retry with explicit inline pins ..."
    pip install \
      "transformers==4.45.2" \
      "datasets==2.21.0" \
      "evaluate" \
      "accelerate==0.34.2" \
      "jiwer==2.6.0" \
      "soundfile==0.12.1" \
      "librosa==0.10.2" \
      "click" \
      "pytorch-lightning==2.1.4" \
      "huggingface_hub"
  fi
}

smoke_verify_python_deps() {
  echo "[verify] $(python -c 'import torch, transformers, datasets; print("torch", torch.__version__, "cuda_build", torch.version.cuda, "cuda_ok", torch.cuda.is_available())')"
  python -c 'import torchaudio; print("[verify] torchaudio", torchaudio.__version__)'
  python -c 'import pytorch_lightning as pl; from lightning_lite.utilities.rank_zero import _get_rank; print("[verify] pytorch-lightning", pl.__version__)'
}
