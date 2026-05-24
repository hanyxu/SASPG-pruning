#!/usr/bin/env bash
# Shared helpers for the 20-slot smoke grid.
# Source from smoke_20exp_500steps.sh (do not execute standalone).
#
# Factors: METHOD × PRUNE_MODE × BACKBONE × SPARSITY
#   METHOD     = saspg | mag | nasp
#   PRUNE_MODE = unstr | str
#   BACKBONE   = w2v | hubert
#   SPARSITY   = spXX  (XX = target prune ratio in percent, e.g. sp50 -> max_prune 0.5)

# --- Sparsity: target prune ratios used by training (arbitrary spXX) ---
smoke_sparsity_ratios() {
  local sp="$1"
  if [[ "$sp" =~ ^sp([0-9]+)$ ]]; then
    local pct="${BASH_REMATCH[1]}"
  else
    echo "ERROR: unknown sparsity tag: $sp (use spXX, e.g. sp50 sp75 sp90)" >&2
    return 1
  fi
  if (( pct < 1 || pct > 99 )); then
    echo "ERROR: sparsity percent must be 1-99, got sp${pct}" >&2
    return 1
  fi
  SMOKE_MAX_PRUNE=$(awk -v p="$pct" 'BEGIN { printf "%.6f", p / 100 }')
  # Match legacy schedule: looser floor below 50%, tight floor above.
  if (( pct <= 50 )); then
    SMOKE_MIN_PRUNE=$(awk -v m="$SMOKE_MAX_PRUNE" 'BEGIN { printf "%.6f", m - 0.02 }')
  else
    SMOKE_MIN_PRUNE=$(awk -v m="$SMOKE_MAX_PRUNE" 'BEGIN { printf "%.6f", m - 0.002 }')
  fi
}

# --- Clear --reg-type aliases (resolved in main_prune.resolve_smoke_reg_type) ---
smoke_reg_type_alias() {
  local method="$1" mode="$2" backbone="$3"
  echo "${method}_${mode}"
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

# --- SASPG ---
run_smoke_saspg() {
  local out="$1" method="$2" mode="$3" backbone="$4" sp="$5"
  smoke_sparsity_ratios "$sp" || return 1
  smoke_train_batch "$backbone" || return 1
  local mname reg
  mname="$(smoke_model_name "$backbone")" || return 1
  reg="$(smoke_reg_type_alias saspg "$mode" "$backbone")"

  local -a extra=()
  if [[ "$mode" == "str" ]]; then
    # SASPG str: channel gates only — no NASP Gumbel ratio ladder (--value-*).
    extra+=(--channel-pruning)
  fi

  mkdir -p "$out"
  "$PYTHON" "$ROOT/main_prune.py" train-pruned \
    --cuda --logging-first-step --hard --decay-tau --only-size-hard \
    --max-prune-ratio "$SMOKE_MAX_PRUNE" --min-prune-ratio "$SMOKE_MIN_PRUNE" \
    --no-weight-prune --batch-size "$SMOKE_BS" --seed 1000 --model-name "$mname" \
    --learning-rate 2e-4 --gating-lambda $([[ "$mode" == "str" ]] && echo 4e-5 || echo 2e-5) \
    --gating-dis 0. --reg-type "$reg" --num-epochs 0 --output-dir "$out" \
    --gradient-accumulation-steps "$SMOKE_GACC" --save-model \
    --save-steps "$MAX_STEPS" --save-total-limit 1 --eval-steps "$MAX_STEPS" \
    --metric-for-best-model eval_wer --load-best-model-at-end \
    --greater-is-better False --eval-strategy steps --max-steps "$MAX_STEPS" \
    ${extra[@]+"${extra[@]}"}
}

# --- NASP (str only) ---
run_smoke_nasp() {
  local out="$1" backbone="$2" sp="$3"
  smoke_sparsity_ratios "$sp" || return 1
  smoke_train_batch "$backbone" || return 1
  local mname
  mname="$(smoke_model_name "$backbone")" || return 1
  local reg
  reg="$(smoke_reg_type_alias nasp str "$backbone")"

  mkdir -p "$out"
  # NASP str: 7-tier Gumbel candidates (1, 0.75, 0.5, 0.25, 0.125, 0.1, 0.075) via --value-*.
  "$PYTHON" "$ROOT/main_prune.py" train-pruned \
    --cuda --logging-first-step --channel-pruning --hard \
    --value-1 0.3 --value-075 0.2 --value-05 0.2 --value-025 0.2 --value-0125 0.1 \
    --value-01 0.1 --value-0075 0.1 \
    --decay-tau --only-size-hard \
    --max-prune-ratio "$SMOKE_MAX_PRUNE" --min-prune-ratio "$SMOKE_MIN_PRUNE" \
    --no-weight-prune --batch-size "$SMOKE_BS" --seed 1000 --model-name "$mname" \
    --learning-rate 2e-4 --gating-lambda 4e-5 --gating-dis 0. --reg-type "$reg" \
    --num-epochs 0 --output-dir "$out" --gradient-accumulation-steps "$SMOKE_GACC" \
    --save-model --save-steps "$MAX_STEPS" --save-total-limit 1 --eval-steps "$MAX_STEPS" \
    --metric-for-best-model eval_wer --load-best-model-at-end \
    --greater-is-better False --eval-strategy steps --max-steps "$MAX_STEPS"
}

# --- Magnitude: prune-first (mag_asr_prune_export) -> single finetune (mag_prune_first) ---
_smoke_mag_baseline_dir() {
  local backbone="$1"
  case "$backbone" in
    w2v)    echo "${ASR_W2V_MODEL_PATH:-${ROOT}/hf_models/wav2vec2-base-100h}" ;;
    hubert) echo "${ASR_HUBERT_MODEL_PATH}" ;;
    *) return 1 ;;
  esac
}

_smoke_mag_prune_export() {
  local out="$1" backbone="$2" sp="$3" mode="$4"
  smoke_sparsity_ratios "$sp" || return 1
  local mname baseline
  mname="$(smoke_model_name "$backbone")" || return 1
  baseline="$(_smoke_mag_baseline_dir "$backbone")" || return 1
  [[ -d "$baseline" ]] || { echo "FAILED mag_${mode}: missing baseline $baseline" >&2; return 1; }
  mkdir -p "$out"
  "$PYTHON" "$ROOT/mag_asr_prune_export.py" \
    --mode "$mode" --model-name "$mname" \
    --baseline-dir "$baseline" --output-dir "$out" \
    --prune-ratio "$SMOKE_MAX_PRUNE"
}

_smoke_mag_finetune() {
  local out="$1" backbone="$2" sp="$3" mode="$4"
  smoke_sparsity_ratios "$sp" || return 1
  smoke_train_batch "$backbone" || return 1
  local mname reg pruned_ckpt
  mname="$(smoke_model_name "$backbone")" || return 1
  reg="$(smoke_reg_type_alias mag unstr "$backbone")"
  pruned_ckpt="$out/out/pruned"
  [[ -d "$pruned_ckpt" ]] || { echo "FAILED mag_${mode}: missing $pruned_ckpt" >&2; return 1; }

  local -a mag_extra=(--mag-prune-first)
  if [[ "$mode" == "unstr" ]]; then
    mag_extra+=(--mag-prune --hand-ratio "$SMOKE_MAX_PRUNE")
  else
    mag_extra+=(--fix-prob)
  fi

  mkdir -p "$out"
  "$PYTHON" "$ROOT/main_prune.py" train-pruned \
    --cuda --logging-first-step --hard --decay-tau --only-size-hard \
    --max-prune-ratio "$SMOKE_MAX_PRUNE" --min-prune-ratio "$SMOKE_MIN_PRUNE" \
    --no-weight-prune --batch-size "$SMOKE_BS" --seed 1000 --model-name "$mname" \
    --model-path "$pruned_ckpt" --learning-rate 2e-4 --gating-lambda 2e-5 --gating-dis 0. \
    --reg-type "$reg" --num-epochs 0 --output-dir "$out" \
    --gradient-accumulation-steps "$SMOKE_GACC" --save-model \
    --save-steps "$MAX_STEPS" --save-total-limit 1 --eval-steps "$MAX_STEPS" \
    --metric-for-best-model eval_wer --load-best-model-at-end \
    --greater-is-better False --eval-strategy steps --max-steps "$MAX_STEPS" \
    ${mag_extra[@]+"${mag_extra[@]}"}
}

run_smoke_mag_unstr() {
  local out="$1" backbone="$2" sp="$3"
  _smoke_mag_prune_export "$out" "$backbone" "$sp" unstr || return 1
  _smoke_mag_finetune "$out" "$backbone" "$sp" unstr
}

run_smoke_mag_str() {
  local out="$1" backbone="$2" sp="$3"
  _smoke_mag_prune_export "$out" "$backbone" "$sp" str || return 1
  _smoke_mag_finetune "$out" "$backbone" "$sp" str
}

# Dispatch one cell: METHOD MODE BACKBONE SPARSITY_TAG OUT_DIR
run_smoke_cell() {
  local method="$1" mode="$2" backbone="$3" sp="$4" out="$5"
  case "${method}_${mode}" in
    saspg_unstr|saspg_str) run_smoke_saspg "$out" "$method" "$mode" "$backbone" "$sp" ;;
    mag_unstr) run_smoke_mag_unstr "$out" "$backbone" "$sp" ;;
    mag_str)   run_smoke_mag_str "$out" "$backbone" "$sp" ;;
    nasp_str)  run_smoke_nasp "$out" "$backbone" "$sp" ;;
    nasp_unstr)
      echo "ERROR: NASP is str-only (no nasp_unstr slot)" >&2
      return 1
      ;;
    *)
      echo "ERROR: unknown method=$method mode=$mode" >&2
      return 1
      ;;
  esac
}
