#!/usr/bin/env bash
# Internal: run one ASR NASP structured (7-tier Gumbel) smoke cell on LibriSpeech 100h.
# Usage: asr_run_nasp_str_one.sh <slot_name> <w2v|hubert> [out_dir]
set -euo pipefail

_slot="${1:?slot_name}"
_backbone="${2:?w2v|hubert}"
_out="${3:-}"

_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common_env.sh
source "${_script_dir}/../common_env.sh"
# shellcheck source=asr_smoke_100h_lib.sh
source "${_script_dir}/asr_smoke_100h_lib.sh"

cd "$ASR_ROOT"
export PYTHONPATH="${ASR_ROOT}${PYTHONPATH:+:$PYTHONPATH}"

if [[ -z "$_out" ]]; then
  _out="${ASR_SMOKE_OUT_ROOT}/${MAX_STEPS}/${_slot}"
fi

preflight_asr() {
  [[ -f "${ASR_ROOT}/main_prune.py" ]] || { echo "ERROR: missing ${ASR_ROOT}/main_prune.py" >&2; exit 1; }
  local _audio="${LIBRISPEECH_AUDIO_ROOT:-${ASR_ROOT}/data/librispeech/audio}"
  [[ -d "${_audio}/train-clean-100" ]] || {
    echo "ERROR: LibriSpeech 100h not found under: ${_audio}" >&2
    echo "  export LIBRISPEECH_AUDIO_ROOT=/path/to/LibriSpeech" >&2
    echo "  or: ln -sfn /path/to/LibriSpeech ${ASR_ROOT}/data/librispeech/audio" >&2
    exit 1
  }
  export LIBRISPEECH_AUDIO_ROOT="${_audio}"
  case "$_backbone" in
    w2v)
      [[ -d "${ASR_W2V_MODEL_PATH:-}" ]] || {
        echo "ERROR: ASR_W2V_MODEL_PATH missing: ${ASR_W2V_MODEL_PATH:-unset}" >&2
        exit 1
      }
      echo "[ASR] W2V checkpoint: ${ASR_W2V_MODEL_PATH}"
      ;;
    hubert)
      [[ -d "${ASR_HUBERT_MODEL_PATH:-}" ]] || {
        echo "ERROR: ASR_HUBERT_MODEL_PATH missing: ${ASR_HUBERT_MODEL_PATH:-unset}" >&2
        exit 1
      }
      echo "[ASR] HuBERT checkpoint: ${ASR_HUBERT_MODEL_PATH}"
      ;;
  esac
}

preflight_asr
asr_smoke_host_preflight || exit 1
export CUDA_VISIBLE_DEVICES="${SMOKE_GPU_INDEX}"
asr_smoke_cuda_preflight || exit 1

_sp="${SMOKE_SPARSITY:-sp50}"
if [[ -n "${SMOKE_SPARSITY_STR:-}" ]]; then
  _sp="${SMOKE_SPARSITY_STR}"
fi

echo "[ASR] slot=${_slot} method=nasp_str backbone=${_backbone} sp=${_sp} type=100 max_steps=${MAX_STEPS} out=${_out}"
run_asr_smoke_nasp_str_100h "$_out" "$_backbone" "$_sp"
