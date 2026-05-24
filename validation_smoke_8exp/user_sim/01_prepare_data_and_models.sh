#!/usr/bin/env bash
# Simulate a new user: wire LibriSpeech + HF weights into cloned SASPG-pruning (no old project symlinks).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=paths.env
source "${ROOT}/paths.env"

# shellcheck source=lib/asr_conda_activate.sh
source "${ROOT}/lib/asr_conda_activate.sh"
smoke_conda_activate || exit 1

ASR_ROOT="${SASPG_REPO}/ASR"
SUPERB_ROOT="${SASPG_REPO}/SUPERB"
_repo_pretrained="${SASPG_REPO}/SUPERB/pretrained"
export DPHUBERT_PRETRAINED_DIR="${WORK_ROOT}/superb/pretrained"
mkdir -p "${DPHUBERT_PRETRAINED_DIR}"
if [[ -d "${_repo_pretrained}" ]]; then
  for _pth in hubert-base-ls960.hf.pth hubert-large-ll60k.hf.pth wav2vec2-base.hf.pth wavlm-base-plus.hf.pth; do
    if [[ -f "${_repo_pretrained}/${_pth}" && ! -e "${DPHUBERT_PRETRAINED_DIR}/${_pth}" ]]; then
      ln -sf "${_repo_pretrained}/${_pth}" "${DPHUBERT_PRETRAINED_DIR}/${_pth}"
    fi
  done
fi
export DPHUBERT_TSV_DIR="${WORK_ROOT}/superb/tsv"

die() { echo "ERROR: $*" >&2; exit 1; }

[[ -d "${HF_MODELS_SRC}" ]] || die "HF_MODELS_SRC missing: ${HF_MODELS_SRC}"
[[ -d "${LIBRISPEECH_ROOT}/train-clean-100" ]] || die "LIBRISPEECH_ROOT missing train-clean-100: ${LIBRISPEECH_ROOT}"
[[ -d "${ASR_ROOT}" ]] || die "ASR repo missing: ${ASR_ROOT}"
[[ -d "${SUPERB_ROOT}" ]] || die "SUPERB repo missing: ${SUPERB_ROOT}"

mkdir -p "${WORK_ROOT}"/{logs/asr,logs/superb,superb/pretrained,superb/tsv}

echo "========== ASR: symlink audio + hf_models (as in DATA.md) =========="
mkdir -p "${ASR_ROOT}/data/librispeech"
rm -rf "${ASR_ROOT}/data/librispeech/audio" "${ASR_ROOT}/hf_models"
ln -sfn "${LIBRISPEECH_ROOT}" "${ASR_ROOT}/data/librispeech/audio"
ln -sfn "${HF_MODELS_SRC}" "${ASR_ROOT}/hf_models"
echo "[OK] audio -> ${LIBRISPEECH_ROOT}"
echo "[OK] hf_models -> ${HF_MODELS_SRC} (bundle; ASR loads via env paths below)"
echo "     ASR W2V:    ${ASR_W2V_MODEL_PATH}"
echo "     ASR HuBERT: ${ASR_HUBERT_MODEL_PATH}"
[[ -d "${ASR_W2V_MODEL_PATH}" ]] || die "ASR W2V missing: ${ASR_W2V_MODEL_PATH}"
[[ -d "${ASR_HUBERT_MODEL_PATH}" ]] || die "ASR HuBERT missing: ${ASR_HUBERT_MODEL_PATH}"

echo "========== SUPERB: LibriSpeech TSV (100h + valid only; fast) =========="
if [[ -f "${DPHUBERT_TSV_DIR}/train100.tsv" && -f "${DPHUBERT_TSV_DIR}/valid.tsv" ]]; then
  echo "[skip] TSV already at ${DPHUBERT_TSV_DIR}"
else
  python "${ROOT}/lib/prepare_librispeech_tsv_100h.py" \
    --data "${LIBRISPEECH_ROOT}" \
    --out "${DPHUBERT_TSV_DIR}"
fi
for _fam in upstream_unstr upstream_str; do
  mkdir -p "${SUPERB_ROOT}/${_fam}/data"
  rm -rf "${SUPERB_ROOT}/${_fam}/data/librispeech"
  ln -sfn "${DPHUBERT_TSV_DIR}" "${SUPERB_ROOT}/${_fam}/data/librispeech"
done
echo "[OK] TSV under ${DPHUBERT_TSV_DIR}"

echo "========== SUPERB: upstream SSL teachers (NOT ASR fine-tuned ckpts) =========="
echo "     SUPERB hubert-base:  ${SUPERB_HF_HUBERT_BASE}"
echo "     SUPERB hubert-large: ${SUPERB_HF_HUBERT_LARGE}"
echo "     SUPERB w2v SSL:      ${SUPERB_HF_W2V_SSL}"
echo "     SUPERB wavlm SSL:    ${SUPERB_HF_WAVLM}"

echo "========== SUPERB: resolve HuBERT-base weights (user bundle may be config-only) =========="
HF_HUBERT_BASE="${SUPERB_HF_HUBERT_BASE}"
if [[ ! -f "${HF_HUBERT_BASE}/pytorch_model.bin" && ! -f "${HF_HUBERT_BASE}/model.safetensors" ]]; then
  echo "[fetch] hubert-base-ls960 weights missing under ${HF_HUBERT_BASE}; downloading (as fetch_hf_models.sh would) ..."
  HF_HUBERT_BASE="${WORK_ROOT}/hf_models/hubert-base-ls960"
  mkdir -p "${HF_HUBERT_BASE}"
  export HF_HUBERT_BASE
  python - <<'PY'
from huggingface_hub import snapshot_download
from pathlib import Path
import os
out = Path(os.environ["HF_HUBERT_BASE"])
snapshot_download("facebook/hubert-base-ls960", local_dir=str(out))
print(f"[OK] downloaded to {out}")
PY
fi
export HF_HUBERT_BASE
export HF_HUBERT_LARGE="${SUPERB_HF_HUBERT_LARGE}"

echo "========== SUPERB: HF -> .hf.pth teacher checkpoints =========="
export HF_MODELS_SRC DPHUBERT_PRETRAINED_DIR HF_HUBERT_BASE HF_HUBERT_LARGE
bash "${ROOT}/lib/convert_hf_to_dphubert.sh" || true

echo "========== SUPERB: fallback copy .hf.pth from DPHuBERT_pretrain_unstr =========="
DPHuBERT_PRETRAINED_SRC="${DPHuBERT_PRETRAINED_SRC:-${SASPG_REPO}/SUPERB/pretrained}"
for _pth in hubert-base-ls960.hf.pth hubert-large-ll60k.hf.pth; do
  if [[ ! -f "${DPHUBERT_PRETRAINED_DIR}/${_pth}" && -f "${DPHuBERT_PRETRAINED_SRC}/${_pth}" ]]; then
    echo "[copy] ${DPHuBERT_PRETRAINED_SRC}/${_pth} -> ${DPHUBERT_PRETRAINED_DIR}/"
    cp "${DPHuBERT_PRETRAINED_SRC}/${_pth}" "${DPHUBERT_PRETRAINED_DIR}/${_pth}"
  fi
done
[[ -f "${DPHUBERT_PRETRAINED_DIR}/hubert-base-ls960.hf.pth" ]] \
  || die "missing ${DPHUBERT_PRETRAINED_DIR}/hubert-base-ls960.hf.pth (run convert or set DPHuBERT_PRETRAINED_SRC)"
[[ -f "${DPHUBERT_PRETRAINED_DIR}/hubert-large-ll60k.hf.pth" ]] \
  || die "missing ${DPHUBERT_PRETRAINED_DIR}/hubert-large-ll60k.hf.pth (run convert or set DPHuBERT_PRETRAINED_SRC)"

echo "========== SUPERB: MAG pruned ckpts from DPHuBERT_pretrain_unstr =========="
for _pruned in pruned_hubert_base.pth pruned_hubert_large.pth; do
  if [[ -f "${DPHuBERT_PRETRAINED_SRC}/${_pruned}" ]]; then
    echo "[copy] ${DPHuBERT_PRETRAINED_SRC}/${_pruned} -> ${DPHUBERT_PRETRAINED_DIR}/"
    cp -f "${DPHuBERT_PRETRAINED_SRC}/${_pruned}" "${DPHUBERT_PRETRAINED_DIR}/${_pruned}"
  else
    echo "[warn] missing ${DPHuBERT_PRETRAINED_SRC}/${_pruned} (MAG smoke may run prune_*_mag locally)"
  fi
done

echo "========== SUPERB: validate experiment matrix =========="
cd "${SUPERB_ROOT}"
python3 -m core experiments validate

echo "[OK] prepare complete. Work dir: ${WORK_ROOT}"
