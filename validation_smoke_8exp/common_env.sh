#!/usr/bin/env bash
# Shared paths for the 8-experiment SASPG smoke grid (ASR + SUPERB).
# Source from individual scripts: source "$(dirname "$0")/../common_env.sh"

: "${BASH_SOURCE[0]:?}"

_smoke_common_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SASPG_REPO_ROOT="$(cd "${_smoke_common_dir}/.." && pwd)"
export SSLPRUNE_WS="${SSLPRUNE_WS:-$(cd "${SASPG_REPO_ROOT}/../.." && pwd)}"

# Optional user-simulation paths (see user_sim/paths.env)
if [[ -f "${_smoke_common_dir}/user_sim/paths.env" ]]; then
  # shellcheck source=user_sim/paths.env
  source "${_smoke_common_dir}/user_sim/paths.env"
fi

# ASR vs SUPERB checkpoints (see user_sim/paths.env)
export HF_MODELS_SRC="${HF_MODELS_SRC:-}"
export ASR_HF_MODELS_ROOT="${ASR_HF_MODELS_ROOT:-${HF_MODELS_SRC}}"
export ASR_W2V_MODEL_PATH="${ASR_W2V_MODEL_PATH:-}"
export ASR_HUBERT_MODEL_PATH="${ASR_HUBERT_MODEL_PATH:-}"
export SUPERB_HF_HUBERT_BASE="${SUPERB_HF_HUBERT_BASE:-${HF_MODELS_SRC}/hubert-base-ls960}"
export SUPERB_HF_HUBERT_LARGE="${SUPERB_HF_HUBERT_LARGE:-${HF_MODELS_SRC}/hubert-large-ll60k}"
export SUPERB_HF_W2V_SSL="${SUPERB_HF_W2V_SSL:-${HF_MODELS_SRC}/wav2vec2-base}"
export SUPERB_HF_WAVLM="${SUPERB_HF_WAVLM:-${HF_MODELS_SRC}/wavlm-base-plus}"

export ASR_ROOT="${ASR_ROOT:-${SASPG_REPO}/ASR}"
export ASR_ROOT="${ASR_ROOT:-${SASPG_REPO_ROOT}/ASR}"
export SUPERB_ROOT="${SUPERB_ROOT:-${SASPG_REPO_ROOT}/SUPERB}"

# --- ASR (LibriSpeech 100h) ---
export LIBRISPEECH_AUDIO_ROOT="${LIBRISPEECH_AUDIO_ROOT:-${LIBRISPEECH_ROOT:-${ASR_ROOT}/data/librispeech/audio}}"
export SSLPRUNE_LIBRISPEECH_CSV_ROOT="${SSLPRUNE_LIBRISPEECH_CSV_ROOT:-${ASR_ROOT}/data/librispeech/csv_metadata}"

# --- SUPERB upstream (DPHuBERT) ---
export SASPG_ROOT="${SASPG_ROOT:-${SSLPRUNE_WS}}"
export WORK_ROOT="${WORK_ROOT:-${SASPG_REPO_ROOT}/validation_smoke_8exp/user_sim/work}"
export ASR_SMOKE_OUT_ROOT="${ASR_SMOKE_OUT_ROOT:-${WORK_ROOT}/asr_smoke_out}"
export DPHuBERT_PRETRAINED_DIR="${DPHuBERT_PRETRAINED_DIR:-${WORK_ROOT}/superb/pretrained}"
export DPHuBERT_TSV_DIR="${DPHuBERT_TSV_DIR:-${WORK_ROOT}/superb/tsv}"

# Smoke defaults (override per job)
export MAX_STEPS="${MAX_STEPS:-500}"
export MAG_PRUNE_STEPS="${MAG_PRUNE_STEPS:-50}"
export SMOKE_SPARSITY="${SMOKE_SPARSITY:-sp50}"
export SMOKE_GPU_INDEX="${SMOKE_GPU_INDEX:-0}"
export PYTHON="${PYTHON:-python3}"
export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-0}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

# SUPERB: 1=dry-run only (validate CSV + launcher path); 0=invoke pipeline (needs training enabled in launcher)
export SUPERB_SMOKE_DRY_RUN="${SUPERB_SMOKE_DRY_RUN:-1}"
