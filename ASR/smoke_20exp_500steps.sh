#!/usr/bin/env bash
# Smoke-run the 20 ASR pruning experiment slots (500 optimizer steps each) with all
# outputs under this release directory. Requires a vendored full training tree: place
# main_prune.py and the complete models/ package next to this script (see preflight).
#
# Experiment grid (20 slots = 3 methods x 2 modes x 2 backbones x 2 sparsities, NASP str-only):
#
#   Factors in each folder name:  {saspg|mag|nasp}_{unstr|str}_{w2v|hubert}_sp{50|90}
#
#   unstr — no physical weight shape change. Inference = weight * mask (mask=0 disables
#           connections). Deploy/train must keep mask (or equivalent) with full-shaped weights.
#   str   — physical shape shrink after prune. Smaller pytorch_model.bin; inference uses
#           shrunk tensors directly, no mask at inference.
#
#   - SASPG (8): 2 backbones x sp50/sp90 x {unstr, str}
#       * unstr: one training stage; mask x weight each forward (--reg-type saspg_unstr)
#       * str:   one training stage; channel gates (no Gumbel ladder) + structural export (--reg-type saspg_str)
#   - Magnitude (8): same 8 cells
#       * unstr: stage1 (MAG_PRUNE_STEPS) -> stage2 finetune with fixed mag_mask x weight each
#                forward (mask=0 gets zero grad); load stage1 ckpt weights + external masks;
#                NO prune_ASR / no smaller shapes in checkpoint-*/pruned/
#       * str:   stage1 -> prune_ASR_*_mag.py -> checkpoint-*/pruned/ (smaller shapes) -> stage2
#   - NASP (4): str only; Gumbel 7-tier ladder (--value-*, NASP only); structural export
#
# Runners: smoke_experiment_lib.sh (run_smoke_cell METHOD MODE BACKBONE SPARSITY OUT_DIR).
# Details: smoke_experiment_matrix.md
#
# Environment:
#   MAX_STEPS       default 500
#   MAG_PRUNE_STEPS default 50   (offline pruning stage for Magnitude)
#   SKIP_MAGNITUDE  if set to 1, skip the 8 magnitude runs (still counts as skipped in the summary)
#   PYTHON          python interpreter to use
#   SMOKE_GPU_INDEX default 2        (physical GPU id for nvidia-smi -i and CUDA_VISIBLE_DEVICES)
#   GPU_MEM_IDLE_MAX_MB default 99   (idle when memory.used MiB <= this; default 99 => run when used < 100 MiB)
#   GPU_POLL_SEC      default 30    (seconds between checks while waiting)
#   SMOKE_CMD         default main; set to mag_w2v_retry_4 to re-run only the four failed Wav2Vec2
#                     magnitude slots (09,10,13,14); see rerun_smoke_mag_w2v_failed_4.sh
#   DATALOADER_NUM_WORKERS default 4 in this script (override e.g. 32 for local SSD). Passed to main_prune.py via env.
#   PYTHONUNBUFFERED default 1 so retry logs flush promptly under NFS.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
# Smoke/NFS: many DataLoader workers often stall after "Loading cached dataset"; logs stay silent without unbuffered stdio.
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-0}"

MAX_STEPS="${MAX_STEPS:-500}"
MAG_PRUNE_STEPS="${MAG_PRUNE_STEPS:-50}"
SKIP_MAGNITUDE="${SKIP_MAGNITUDE:-0}"
PYTHON="${PYTHON:-python}"
SMOKE_GPU_INDEX="${SMOKE_GPU_INDEX:-2}"
GPU_MEM_IDLE_MAX_MB="${GPU_MEM_IDLE_MAX_MB:-99}"
GPU_POLL_SEC="${GPU_POLL_SEC:-30}"
WAIT_GPU_EACH_TASK="${WAIT_GPU_EACH_TASK:-0}"

die() { echo "ERROR: $*" >&2; exit 1; }

need_file() { [[ -f "$1" ]] || die "Missing required file: $1 (copy from full SSLprune / IS25_code tree into this release)."; }

preflight() {
  need_file "$ROOT/main_prune.py"
  # Minimal modules touched by this smoke grid (wav2vec2 + HuBERT paths).
  # Vendor the full models/ and utils/ tree expected by your main_prune.py
  # (e.g. copy from SSLprune/IS25_code next to this folder into $ROOT).
  for f in \
    models/pruned_wav2vec2_fln_prune.py \
    models/pruned_wav2vec2_fln_channel_prune.py \
    models/pruned_wav2vec2_fln_prune_mag.py \
    models/pruned_hubert_fln_prune.py \
    models/pruned_hubert_fln_channel_prune.py \
    models/pruned_hubert_fln_prune_mag.py \
    smoke_experiment_lib.sh \
    smoke_experiment_matrix.md; do
    need_file "$ROOT/$f"
  done
  # Structural export (magnitude str only; slots 13-16):
  for f in prune_ASR_w2v_mag.py prune_ASR_hubert_mag.py; do
    need_file "$ROOT/$f"
  done
}

# --- GPU 2 idle gate: only start the rotation when selected GPU has no (or negligible) VRAM in use ---
gpu_used_mib() {
  local used
  used="$(nvidia-smi -i "$SMOKE_GPU_INDEX" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d '[:space:]')"
  echo "$used"
}

wait_for_idle_gpu() {
  local log_file="$1"
  command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found; cannot check GPU ${SMOKE_GPU_INDEX}"
  while :; do
    local used
    used="$(gpu_used_mib)"
    [[ "$used" =~ ^[0-9]+$ ]] || die "Unexpected nvidia-smi memory.used for GPU ${SMOKE_GPU_INDEX}: '${used}'"
    local poll_msg
    poll_msg="$(date -Iseconds) [GPU_POLL] gpu=${SMOKE_GPU_INDEX} memory.used_mib=${used} idle_if_used_mib<$((GPU_MEM_IDLE_MAX_MB + 1))"
    echo "$poll_msg" >>"$log_file"
    echo "$poll_msg"
    if [[ "$used" -le "$GPU_MEM_IDLE_MAX_MB" ]]; then
      return 0
    fi
    sleep "$GPU_POLL_SEC"
  done
}

# --- Experiment runners (METHOD x MODE x BACKBONE x SPARSITY) ---
# shellcheck source=smoke_experiment_lib.sh
source "$ROOT/smoke_experiment_lib.sh"

# Re-run only the four Wav2Vec2 magnitude smoke slots that previously failed at offline prune
# (09 unstr sp50, 10 unstr sp90, 13 str sp50, 14 str sp90). Invoked when SMOKE_CMD=mag_w2v_retry_4.
mag_w2v_retry_4_main() {
  preflight
  local smoke_root="$ROOT/smoke_runs_${MAX_STEPS}"
  local console_dir="$smoke_root/_console"
  mkdir -p "$console_dir"
  local CLEAN_RETRY="${CLEAN_RETRY:-1}"
  local -a roots=(
    "$smoke_root/09_mag_unstr_w2v_sp50"
    "$smoke_root/10_mag_unstr_w2v_sp90"
    "$smoke_root/13_mag_str_w2v_sp50"
    "$smoke_root/14_mag_str_w2v_sp90"
  )
  local p d
  for p in "${roots[@]}"; do
    if [[ "$CLEAN_RETRY" == "all" ]]; then
      echo "[RETRY4 CLEAN=all] rm -rf $p/stage2 $p/out"
      rm -rf "$p/stage2" "$p/out"
    elif [[ "$CLEAN_RETRY" == "1" ]]; then
      echo "[RETRY4 CLEAN=1] $p -> remove stage2 + out/checkpoint-*/pruned"
      rm -rf "$p/stage2"
      shopt -s nullglob
      for d in "$p"/out/checkpoint-*/pruned; do
        if [[ -d "$d" ]]; then
          echo "[RETRY4 CLEAN] rm -rf $d"
          rm -rf "$d"
        fi
      done
      shopt -u nullglob
    else
      echo "[RETRY4 CLEAN=$CLEAN_RETRY] skip directory cleanup (set CLEAN_RETRY=1 or all)"
    fi
  done

  local SMOKE_GPU_WAIT_LOG="$console_dir/gpu_wait_retry_mag_w2v_4.log"
  : >"$SMOKE_GPU_WAIT_LOG"
  wait_for_idle_gpu "$SMOKE_GPU_WAIT_LOG"
  export CUDA_VISIBLE_DEVICES="$SMOKE_GPU_INDEX"
  local _u
  _u="$(gpu_used_mib)"
  echo "[RETRY4 GPU_READY] gpu=${SMOKE_GPU_INDEX} memory.used_mib=${_u} cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
  echo "[RETRY4] DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-} PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-}"

  local retry4_fail=0
  retry4_one() {
    local name="$1"
    shift
    local log_file="$console_dir/retry4_${name}.log"
    if [[ "${WAIT_GPU_EACH_TASK:-0}" == "1" ]]; then
      wait_for_idle_gpu "$SMOKE_GPU_WAIT_LOG"
    fi
    echo "[RETRY4 START] task=${name} log=${log_file}"
    if "$@" >"$log_file" 2>&1; then
      echo "[RETRY4 OK] task=${name}"
    else
      local rc=$?
      local tail_line
      tail_line="$(awk 'NF{line=$0} END{print line}' "$log_file" | cut -c1-220)"
      echo "[RETRY4 FAIL] task=${name} exit=${rc} summary=${tail_line}"
      retry4_fail=$((retry4_fail + 1))
    fi
  }

  retry4_one "09_mag_unstr_w2v_sp50"  run_smoke_cell mag unstr w2v sp50 "$smoke_root/09_mag_unstr_w2v_sp50"
  retry4_one "10_mag_unstr_w2v_sp90"  run_smoke_cell mag unstr w2v sp90 "$smoke_root/10_mag_unstr_w2v_sp90"
  retry4_one "13_mag_str_w2v_sp50"   run_smoke_cell mag str   w2v sp50 "$smoke_root/13_mag_str_w2v_sp50"
  retry4_one "14_mag_str_w2v_sp90"   run_smoke_cell mag str   w2v sp90 "$smoke_root/14_mag_str_w2v_sp90"

  echo "========== RETRY4 SUMMARY =========="
  echo "fail_count=${retry4_fail} (09,10 unstr w2v + 13,14 str w2v)"
  [[ "$retry4_fail" -eq 0 ]] || exit 1
}

main() {
  preflight
  local smoke_root="$ROOT/smoke_runs_${MAX_STEPS}"
  mkdir -p "$smoke_root"
  local console_dir="$smoke_root/_console"
  mkdir -p "$console_dir"
  echo "Smoke output root: $smoke_root"
  echo "MAX_STEPS=$MAX_STEPS  MAG_PRUNE_STEPS=$MAG_PRUNE_STEPS  SKIP_MAGNITUDE=$SKIP_MAGNITUDE"
  echo "DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-}  PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-}"
  echo "SMOKE_GPU_INDEX=$SMOKE_GPU_INDEX  GPU_MEM_IDLE_MAX_MB=$GPU_MEM_IDLE_MAX_MB  GPU_POLL_SEC=$GPU_POLL_SEC  WAIT_GPU_EACH_TASK=$WAIT_GPU_EACH_TASK"

  SMOKE_GPU_WAIT_LOG="$console_dir/gpu_wait.log"
  : >"$SMOKE_GPU_WAIT_LOG"
  wait_for_idle_gpu "$SMOKE_GPU_WAIT_LOG"
  export CUDA_VISIBLE_DEVICES="$SMOKE_GPU_INDEX"
  local _u
  _u="$(gpu_used_mib)"
  echo "[GPU_READY] gpu=${SMOKE_GPU_INDEX} memory.used_mib=${_u} cuda_visible_devices=${CUDA_VISIBLE_DEVICES} wait_log=${SMOKE_GPU_WAIT_LOG}"
  local ok=0 fail=0 skip=0
  local done=0 total=20
  local -a failed_summaries=()

  summarize_failure() {
    local log_file="$1"
    local last_line
    last_line="$(awk 'NF{line=$0} END{print line}' "$log_file")"
    if [[ -z "$last_line" ]]; then
      echo "empty log"
      return
    fi
    # Keep summary short and stable for console.
    echo "$last_line" | cut -c1-220
  }

  print_failed_history() {
    if [[ ${#failed_summaries[@]} -eq 0 ]]; then
      return
    fi
    echo "[FAILED_SUMMARY_HISTORY_BEGIN]"
    local item
    for item in "${failed_summaries[@]}"; do
      echo "$item"
    done
    echo "[FAILED_SUMMARY_HISTORY_END]"
  }

  run_slot() {
    local name="$1"
    shift
    if [[ "$WAIT_GPU_EACH_TASK" == "1" ]]; then
      wait_for_idle_gpu "$SMOKE_GPU_WAIT_LOG"
    fi
    local idx=$((done + 1))
    local log_file="$console_dir/${idx}_${name}.log"
    echo "[START] task=${name} progress=${idx}/${total} log=${log_file}"
    "$@" >"$log_file" 2>&1
    local rc=$?
    done=$((done + 1))
    if [[ $rc -eq 0 ]]; then
      ok=$((ok + 1))
      echo "[OK] task=${name} progress=${done}/${total}"
    elif [[ $rc -eq 42 ]]; then
      skip=$((skip + 1))
      echo "[SKIP] task=${name} progress=${done}/${total}"
    else
      local short_summary
      short_summary="$(summarize_failure "$log_file")"
      failed_summaries+=("[FAIL] task=${name} progress=${done}/${total} exit=${rc} summary=${short_summary}")
      fail=$((fail + 1))
      echo "[FAIL] task=${name} progress=${done}/${total} exit=${rc}"
      print_failed_history
    fi
  }

  # --- SASPG 8: run_smoke_cell METHOD MODE BACKBONE SPARSITY OUT_DIR ---
  run_slot "01_saspg_unstr_w2v_sp50"  run_smoke_cell saspg unstr w2v sp50 "$smoke_root/01_saspg_unstr_w2v_sp50"
  run_slot "02_saspg_unstr_w2v_sp90"  run_smoke_cell saspg unstr w2v sp90 "$smoke_root/02_saspg_unstr_w2v_sp90"
  run_slot "03_saspg_unstr_hubert_sp50" run_smoke_cell saspg unstr hubert sp50 "$smoke_root/03_saspg_unstr_hubert_sp50"
  run_slot "04_saspg_unstr_hubert_sp90" run_smoke_cell saspg unstr hubert sp90 "$smoke_root/04_saspg_unstr_hubert_sp90"
  run_slot "05_saspg_str_w2v_sp50"   run_smoke_cell saspg str   w2v sp50 "$smoke_root/05_saspg_str_w2v_sp50"
  run_slot "06_saspg_str_w2v_sp90"   run_smoke_cell saspg str   w2v sp90 "$smoke_root/06_saspg_str_w2v_sp90"
  run_slot "07_saspg_str_hubert_sp50" run_smoke_cell saspg str   hubert sp50 "$smoke_root/07_saspg_str_hubert_sp50"
  run_slot "08_saspg_str_hubert_sp90" run_smoke_cell saspg str   hubert sp90 "$smoke_root/08_saspg_str_hubert_sp90"

  # --- Magnitude 8 ---
  if [[ "$SKIP_MAGNITUDE" == "1" ]]; then
    for i in 09 10 11 12 13 14 15 16; do
      echo "SKIP (SKIP_MAGNITUDE=1): ${i}_mag_*"
      skip=$((skip + 1))
    done
  else
    run_slot "09_mag_unstr_w2v_sp50"  run_smoke_cell mag unstr w2v sp50 "$smoke_root/09_mag_unstr_w2v_sp50"
    run_slot "10_mag_unstr_w2v_sp90"  run_smoke_cell mag unstr w2v sp90 "$smoke_root/10_mag_unstr_w2v_sp90"
    run_slot "11_mag_unstr_hubert_sp50" run_smoke_cell mag unstr hubert sp50 "$smoke_root/11_mag_unstr_hubert_sp50"
    run_slot "12_mag_unstr_hubert_sp90" run_smoke_cell mag unstr hubert sp90 "$smoke_root/12_mag_unstr_hubert_sp90"
    run_slot "13_mag_str_w2v_sp50"   run_smoke_cell mag str   w2v sp50 "$smoke_root/13_mag_str_w2v_sp50"
    run_slot "14_mag_str_w2v_sp90"   run_smoke_cell mag str   w2v sp90 "$smoke_root/14_mag_str_w2v_sp90"
    run_slot "15_mag_str_hubert_sp50" run_smoke_cell mag str   hubert sp50 "$smoke_root/15_mag_str_hubert_sp50"
    run_slot "16_mag_str_hubert_sp90" run_smoke_cell mag str   hubert sp90 "$smoke_root/16_mag_str_hubert_sp90"
  fi

  # --- NASP 4 (str only) ---
  run_slot "17_nasp_w2v_sp50"    run_smoke_cell nasp str w2v sp50 "$smoke_root/17_nasp_w2v_sp50"
  run_slot "18_nasp_w2v_sp90"    run_smoke_cell nasp str w2v sp90 "$smoke_root/18_nasp_w2v_sp90"
  run_slot "19_nasp_hubert_sp50" run_smoke_cell nasp str hubert sp50 "$smoke_root/19_nasp_hubert_sp50"
  run_slot "20_nasp_hubert_sp90" run_smoke_cell nasp str hubert sp90 "$smoke_root/20_nasp_hubert_sp90"

  echo "========== SUMMARY =========="
  echo "ok=$ok  fail=$fail  skipped=$skip  done=$done total=$total"
  print_failed_history

  # Drop heavy smoke weights; keep SMOKE_CKPT_PLACEHOLDER.json for agent audit (see ../strip_smoke_checkpoints.sh).
  if [[ "${STRIP_SMOKE_CKPTS:-1}" == "1" && "$fail" -eq 0 ]]; then
    local strip_script="$ROOT/../strip_smoke_checkpoints.sh"
    if [[ -x "$strip_script" ]]; then
      echo "[STRIP] Removing smoke checkpoint weights under $ROOT (STRIP_SMOKE_CKPTS=1)..."
      bash "$strip_script" || echo "[STRIP] warning: strip_smoke_checkpoints.sh failed (non-fatal)" >&2
    fi
  fi

  # Non-zero exit if any hard failure (skips do not fail the script)
  [[ "$fail" -eq 0 ]] || exit 1
}

case "${SMOKE_CMD:-main}" in
  mag_w2v_retry_4) mag_w2v_retry_4_main "$@" ;;
  *) main "$@" ;;
esac
