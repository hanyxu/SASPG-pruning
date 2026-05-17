#!/usr/bin/env bash
# Resolve SSLprune workspace and pretrained checkpoint directory for upstream launchers.
# Caller must set SCRIPT_DIR to the directory containing the launcher script, then:
#   # shellcheck source=/dev/null
#   source "${SCRIPT_DIR}/../scripts/source_dphubert_env.sh"

: "${SCRIPT_DIR:?Set SCRIPT_DIR to launcher dir before sourcing source_dphubert_env.sh}"

_SSLPRUNE_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
export SASPG_ROOT="${SASPG_ROOT:-${_SSLPRUNE_ROOT}}"

if [[ -n "${DPHuBERT_PRETRAINED_DIR:-}" ]]; then
  export DPHuBERT_PRETRAINED_ROOT="${DPHuBERT_PRETRAINED_DIR}"
else
  unset DPHuBERT_PRETRAINED_ROOT
  for _cand in \
    "${SCRIPT_DIR}/pretrained" \
    "${_SSLPRUNE_ROOT}/DPHuBERT_pretrain/pretrained" \
    "${_SSLPRUNE_ROOT}/DPHuBERT_pretrain_unstr/pretrained"; do
    if [[ -d "${_cand}" ]]; then
      export DPHuBERT_PRETRAINED_ROOT="${_cand}"
      break
    fi
  done
  : "${DPHuBERT_PRETRAINED_ROOT:=${SCRIPT_DIR}/pretrained}"
fi

# LibriSpeech TSV directory (same layout as DPHuBERT_pretrain_unstr: data/librispeech).
# Launchers should use: tsv_dir=${DPHuBERT_TSV_DIR:-data/librispeech}
export DPHuBERT_PRETRAIN_UNSTR_ROOT="${_SSLPRUNE_ROOT}/DPHuBERT_pretrain_unstr"
if [[ -z "${DPHuBERT_TSV_DIR:-}" && -d "${DPHuBERT_PRETRAIN_UNSTR_ROOT}/data/librispeech" ]]; then
  export DPHuBERT_TSV_DIR="${DPHuBERT_PRETRAIN_UNSTR_ROOT}/data/librispeech"
fi
