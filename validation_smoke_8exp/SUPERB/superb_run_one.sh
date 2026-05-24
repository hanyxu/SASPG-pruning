#!/usr/bin/env bash
# Internal: validate + smoke-run one SUPERB upstream experiment (100h, SASPG str/unstr).
# Usage: superb_run_one.sh <exp_id> [log_dir]
set -euo pipefail

_exp_id="${1:?exp_id}"
_log_dir="${2:-}"

_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common_env.sh
source "${_script_dir}/../common_env.sh"

cd "$SUPERB_ROOT"

if [[ ! -d "$SUPERB_ROOT/upstream_str" || ! -d "$SUPERB_ROOT/upstream_unstr" ]]; then
  echo "[SUPERB] preparing upstream_str / upstream_unstr ..."
  bash "${SUPERB_ROOT}/prepare_local_dependencies.sh"
fi

if [[ -z "$_log_dir" ]]; then
  _log_dir="${SUPERB_ROOT}/smoke_100h_8exp/logs"
fi
mkdir -p "$_log_dir"
_log_file="${_log_dir}/${_exp_id}.log"

export DPHuBERT_PRETRAINED_DIR
export DPHuBERT_TSV_DIR
export SASPG_ROOT
export CUDA_VISIBLE_DEVICES="${SMOKE_GPU_INDEX}"

_preflight_superb() {
  if [[ ! -d "${DPHuBERT_PRETRAINED_DIR}" ]]; then
    echo "ERROR: DPHuBERT_PRETRAINED_DIR not found: ${DPHuBERT_PRETRAINED_DIR}" >&2
    exit 1
  fi
  local ckpt=""
  case "$_exp_id" in
    hubert_base_*)
      ckpt="hubert-base-ls960.hf.pth"
      ;;
    hubert_large_*)
      ckpt="hubert-large-ll60k.hf.pth"
      ;;
    *)
      echo "ERROR: unknown model family in exp_id=${_exp_id}" >&2
      exit 1
      ;;
  esac
  [[ -f "${DPHuBERT_PRETRAINED_DIR}/${ckpt}" ]] || {
    echo "ERROR: missing teacher checkpoint ${DPHuBERT_PRETRAINED_DIR}/${ckpt}" >&2
    exit 1
  }
  if [[ ! -d "${DPHuBERT_TSV_DIR}" ]]; then
    echo "ERROR: DPHuBERT_TSV_DIR not found: ${DPHuBERT_TSV_DIR}" >&2
    exit 1
  fi
}

_preflight_superb

{
  echo "=== SUPERB smoke: ${_exp_id} ==="
  echo "SUPERB_ROOT=${SUPERB_ROOT}"
  echo "DPHuBERT_PRETRAINED_DIR=${DPHuBERT_PRETRAINED_DIR}"
  echo "DPHuBERT_TSV_DIR=${DPHuBERT_TSV_DIR}"
  echo "SUPERB_SMOKE_DRY_RUN=${SUPERB_SMOKE_DRY_RUN}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  echo

  echo "--- experiments validate ---"
  python3 -m core experiments validate
  echo

  echo "--- experiments list ---"
  python3 -m core experiments list --exp-id "${_exp_id}" --ready-only
  echo

  _dry_flag=()
  if [[ "${SUPERB_SMOKE_DRY_RUN}" == "1" ]]; then
    _dry_flag=(--dry-run)
    echo "--- pipeline (dry-run: launcher path / env only) ---"
  else
    echo "--- pipeline (REAL RUN: ensure distill/prune blocks are enabled in launcher) ---"
  fi

  python3 -m core pipeline --run-upstream-first --exp-id "${_exp_id}" "${_dry_flag[@]+"${_dry_flag[@]}"}"
} 2>&1 | tee "${_log_file}"

echo "[SUPERB] log=${_log_file}"
