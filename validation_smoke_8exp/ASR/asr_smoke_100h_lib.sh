#!/usr/bin/env bash
# ASR smoke helpers on LibriSpeech 100h (--type 100): SASPG, MAG (unstr+str), NASP str.
# Source from validation_smoke_8exp/ASR/*.sh

# Require GPU node (see paths.env SMOKE_ASR_HOST). Empty SMOKE_ASR_HOST skips host check.
asr_smoke_host_preflight() {
  local required="${SMOKE_ASR_HOST:-}"
  if [[ -z "$required" ]]; then
    return 0
  fi
  local host
  host="$(hostname -s 2>/dev/null || hostname)"
  if [[ "$host" == "$required" || "$host" == *"${required}."* ]]; then
    echo "[ASR] host=${host} (matches SMOKE_ASR_HOST=${required})"
    return 0
  fi
  echo "ERROR: ASR smoke SMOKE_ASR_HOST=${required} but current host is ${host}." >&2
  echo "  Unset SMOKE_ASR_HOST to use driver-based conda (cu121/cu118), or ssh ${required}." >&2
  return 1
}

asr_smoke_cuda_preflight() {
  if ! "${PYTHON:?PYTHON unset}" - <<'PY'
import os
import sys

import torch

host = os.environ.get("HOSTNAME", os.uname().nodename)
cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
if torch.version.cuda is None:
    print("ERROR: CPU-only PyTorch in this env (torch.version.cuda is None).", file=sys.stderr)
    print("  Re-run: bash user_sim/00_create_conda_env.sh  (on the GPU node)", file=sys.stderr)
    sys.exit(1)
if not torch.cuda.is_available():
    print("ERROR: torch.cuda.is_available() is False with CUDA_VISIBLE_DEVICES set.", file=sys.stderr)
    print(f"  host={host} CUDA_VISIBLE_DEVICES={cvd}", file=sys.stderr)
    print(f"  PyTorch {torch.__version__} built for CUDA {torch.version.cuda}", file=sys.stderr)
    print("  If driver is 11.x: bash user_sim/00_create_conda_env_cu118.sh then re-run (auto picks test_saspg_cu118).", file=sys.stderr)
    print("  If driver is 12.x: use test_saspg (00_create_conda_env.sh). Force: ASR_CUDA_VARIANT=cu121|cu118", file=sys.stderr)
    sys.exit(1)
print(f"[ASR] torch {torch.__version__} cuda={torch.version.cuda} device={torch.cuda.get_device_name(0)}")
PY
  then
    return 1
  fi
}

smoke_sparsity_ratios() {
  local sp="$1"
  if [[ "$sp" =~ ^sp([0-9]+)$ ]]; then
    local pct="${BASH_REMATCH[1]}"
  else
    echo "ERROR: unknown sparsity tag: $sp (use spXX, e.g. sp50)" >&2
    return 1
  fi
  if (( pct < 1 || pct > 99 )); then
    echo "ERROR: sparsity percent must be 1-99, got sp${pct}" >&2
    return 1
  fi
  # Supported production grid: 50/60/70/80/90 (smoke defaults: unstr=sp50, str=sp90 in paths.env)
  SMOKE_MAX_PRUNE=$(awk -v p="$pct" 'BEGIN { printf "%.6f", p / 100 }')
  if (( pct <= 50 )); then
    SMOKE_MIN_PRUNE=$(awk -v m="$SMOKE_MAX_PRUNE" 'BEGIN { printf "%.6f", m - 0.02 }')
  else
    SMOKE_MIN_PRUNE=$(awk -v m="$SMOKE_MAX_PRUNE" 'BEGIN { printf "%.6f", m - 0.002 }')
  fi
}

smoke_reg_type_alias() {
  local mode="$1"
  local backbone="$2"
  echo "saspg_${mode}"
}

smoke_train_batch() {
  local backbone="$1"
  case "$backbone" in
    w2v)    SMOKE_BS=16; SMOKE_GACC=2 ;;
    hubert) SMOKE_BS=4;  SMOKE_GACC=8 ;;
    *) echo "ERROR: backbone must be w2v or hubert" >&2; return 1 ;;
  esac
}

smoke_model_name() {
  local backbone="$1"
  case "$backbone" in
    w2v)    echo wav2vec2 ;;
    hubert) echo hubert ;;
    *) return 1 ;;
  esac
}

run_asr_smoke_saspg_100h() {
  local out="$1" mode="$2" backbone="$3" sp="${4:-sp50}"
  smoke_sparsity_ratios "$sp" || return 1
  smoke_train_batch "$backbone" || return 1
  local mname reg
  mname="$(smoke_model_name "$backbone")" || return 1
  reg="$(smoke_reg_type_alias "$mode" "$backbone")"

  local -a extra=()
  if [[ "$mode" == "str" ]]; then
    extra+=(--channel-pruning)
  fi

  local -a tiny=()
  if [[ "${SMOKE_TINY_BATCH:-0}" == "1" ]]; then
    case "$backbone" in
      w2v)    SMOKE_BS=2;  SMOKE_GACC=8 ;;
      hubert) SMOKE_BS=1;  SMOKE_GACC=16 ;;
    esac
  fi
  if [[ -n "${SMOKE_NUM_TRAIN_SAMPLES:-}" ]]; then
    tiny+=(--num-train-samples "$SMOKE_NUM_TRAIN_SAMPLES")
  fi
  if [[ -n "${SMOKE_NUM_VAL_SAMPLES:-}" ]]; then
    tiny+=(--num-val-samples "$SMOKE_NUM_VAL_SAMPLES")
  fi

  mkdir -p "$out"
  "$PYTHON" "$ASR_ROOT/main_prune.py" train-pruned \
    --cuda --type 100 \
    --logging-first-step --hard --decay-tau --only-size-hard \
    --max-prune-ratio "$SMOKE_MAX_PRUNE" --min-prune-ratio "$SMOKE_MIN_PRUNE" \
    --no-weight-prune --batch-size "$SMOKE_BS" --seed 1000 --model-name "$mname" \
    --learning-rate 2e-4 --gating-lambda $([[ "$mode" == "str" ]] && echo 4e-5 || echo 2e-5) \
    --gating-dis 0. --reg-type "$reg" --num-epochs 0 --output-dir "$out" \
    --gradient-accumulation-steps "$SMOKE_GACC" --save-model \
    --save-steps "$MAX_STEPS" --save-total-limit 1 --eval-steps "$MAX_STEPS" \
    --metric-for-best-model eval_wer \
    --load-best-model-at-end \
    --eval-strategy $([[ "${SMOKE_SKIP_EVAL:-0}" == "1" ]] && echo no || echo steps) \
    --max-steps "$MAX_STEPS" \
    ${extra[@]+"${extra[@]}"} ${tiny[@]+"${tiny[@]}"}
}

# --- MAG: prune-first (no CTC) -> single finetune (does not change SASPG / NASP) ---
_asr_mag_baseline_dir() {
  local backbone="$1"
  case "$backbone" in
    w2v)    echo "${ASR_W2V_MODEL_PATH}" ;;
    hubert) echo "${ASR_HUBERT_MODEL_PATH}" ;;
    *) return 1 ;;
  esac
}

_run_asr_mag_prune_export_100h() {
  local out="$1" backbone="$2" mode="$3"
  local mname baseline
  mname="$(smoke_model_name "$backbone")" || return 1
  baseline="$(_asr_mag_baseline_dir "$backbone")" || return 1
  [[ -d "$baseline" ]] || {
    echo "FAILED mag_${mode}: missing baseline checkpoint: $baseline" >&2
    return 1
  }
  mkdir -p "$out"
  "$PYTHON" "$ASR_ROOT/mag_asr_prune_export.py" \
    --mode "$mode" --model-name "$mname" \
    --baseline-dir "$baseline" --output-dir "$out" \
    --prune-ratio "$SMOKE_MAX_PRUNE"
}

_run_asr_mag_finetune_100h() {
  local out="$1" backbone="$2" mode="$3"
  local mname reg pruned_ckpt
  mname="$(smoke_model_name "$backbone")" || return 1
  pruned_ckpt="$out/out/pruned"
  [[ -d "$pruned_ckpt" ]] || {
    echo "FAILED mag_${mode}: missing pruned dir: $pruned_ckpt" >&2
    return 1
  }
  reg="mag_unstr"
  local -a mag_extra=(--mag-prune-first)
  if [[ "$mode" == "unstr" ]]; then
    mag_extra+=(--mag-prune --hand-ratio "$SMOKE_MAX_PRUNE")
  else
    mag_extra+=(--fix-prob)
  fi

  local -a tiny=()
  if [[ "${SMOKE_TINY_BATCH:-0}" == "1" ]]; then
    case "$backbone" in
      w2v)    SMOKE_BS=2;  SMOKE_GACC=8 ;;
      hubert) SMOKE_BS=1;  SMOKE_GACC=16 ;;
    esac
  fi
  if [[ -n "${SMOKE_NUM_TRAIN_SAMPLES:-}" ]]; then
    tiny+=(--num-train-samples "$SMOKE_NUM_TRAIN_SAMPLES")
  fi
  if [[ -n "${SMOKE_NUM_VAL_SAMPLES:-}" ]]; then
    tiny+=(--num-val-samples "$SMOKE_NUM_VAL_SAMPLES")
  fi

  mkdir -p "$out"
  "$PYTHON" "$ASR_ROOT/main_prune.py" train-pruned \
    --cuda --type 100 \
    --logging-first-step --hard --decay-tau --only-size-hard \
    --max-prune-ratio "$SMOKE_MAX_PRUNE" --min-prune-ratio "$SMOKE_MIN_PRUNE" \
    --no-weight-prune --batch-size "$SMOKE_BS" --seed 1000 --model-name "$mname" \
    --model-path "$pruned_ckpt" --learning-rate 2e-4 --gating-lambda 2e-5 --gating-dis 0. \
    --reg-type "$reg" --num-epochs 0 --output-dir "$out" \
    --gradient-accumulation-steps "$SMOKE_GACC" --save-model \
    --save-steps "$MAX_STEPS" --save-total-limit 1 --eval-steps "$MAX_STEPS" \
    --metric-for-best-model eval_wer --load-best-model-at-end \
    --greater-is-better False \
    --eval-strategy $([[ "${SMOKE_SKIP_EVAL:-0}" == "1" ]] && echo no || echo steps) \
    --max-steps "$MAX_STEPS" \
    ${mag_extra[@]+"${mag_extra[@]}"} ${tiny[@]+"${tiny[@]}"}
}

run_asr_smoke_mag_unstr_100h() {
  local out="$1" backbone="$2" sp="${3:-sp50}"
  smoke_sparsity_ratios "$sp" || return 1
  smoke_train_batch "$backbone" || return 1
  _run_asr_mag_prune_export_100h "$out" "$backbone" unstr || return 1
  _run_asr_mag_finetune_100h "$out" "$backbone" unstr || return 1
}

run_asr_smoke_mag_str_100h() {
  local out="$1" backbone="$2" sp="${3:-sp50}"
  smoke_sparsity_ratios "$sp" || return 1
  smoke_train_batch "$backbone" || return 1
  _run_asr_mag_prune_export_100h "$out" "$backbone" str || return 1
  _run_asr_mag_finetune_100h "$out" "$backbone" str || return 1
}

# --- NASP structured (7-tier Gumbel ladder; str only) ---
run_asr_smoke_nasp_str_100h() {
  local out="$1" backbone="$2" sp="${3:-sp50}"
  smoke_sparsity_ratios "$sp" || return 1
  smoke_train_batch "$backbone" || return 1
  local mname reg
  mname="$(smoke_model_name "$backbone")" || return 1
  reg="nasp_str"

  local -a tiny=()
  if [[ "${SMOKE_TINY_BATCH:-0}" == "1" ]]; then
    case "$backbone" in
      w2v)    SMOKE_BS=2;  SMOKE_GACC=8 ;;
      hubert) SMOKE_BS=1;  SMOKE_GACC=16 ;;
    esac
  fi
  if [[ -n "${SMOKE_NUM_TRAIN_SAMPLES:-}" ]]; then
    tiny+=(--num-train-samples "$SMOKE_NUM_TRAIN_SAMPLES")
  fi
  if [[ -n "${SMOKE_NUM_VAL_SAMPLES:-}" ]]; then
    tiny+=(--num-val-samples "$SMOKE_NUM_VAL_SAMPLES")
  fi

  mkdir -p "$out"
  "$PYTHON" "$ASR_ROOT/main_prune.py" train-pruned \
    --cuda --type 100 \
    --logging-first-step --channel-pruning --hard \
    --value-1 0.3 --value-075 0.2 --value-05 0.2 --value-025 0.2 --value-0125 0.1 \
    --value-01 0.1 --value-0075 0.1 \
    --decay-tau --only-size-hard \
    --max-prune-ratio "$SMOKE_MAX_PRUNE" --min-prune-ratio "$SMOKE_MIN_PRUNE" \
    --no-weight-prune --batch-size "$SMOKE_BS" --seed 1000 --model-name "$mname" \
    --learning-rate 2e-4 --gating-lambda 4e-5 --gating-dis 0. --reg-type "$reg" \
    --num-epochs 0 --output-dir "$out" --gradient-accumulation-steps "$SMOKE_GACC" \
    --save-model --save-steps "$MAX_STEPS" --save-total-limit 1 --eval-steps "$MAX_STEPS" \
    --metric-for-best-model eval_wer --load-best-model-at-end \
    --eval-strategy $([[ "${SMOKE_SKIP_EVAL:-0}" == "1" ]] && echo no || echo steps) \
    --max-steps "$MAX_STEPS" \
    ${tiny[@]+"${tiny[@]}"}
}
