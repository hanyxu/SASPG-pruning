#!/usr/bin/env bash
# Wait until physical GPU SMOKE_GPU_INDEX (default 2 on bdda7) has low VRAM usage.
# Source after paths.env:  source "$(dirname "$0")/gpu_wait_smoke.sh"
set -euo pipefail

: "${SMOKE_GPU_INDEX:?set SMOKE_GPU_INDEX (e.g. 2 for bdda7 card 2)}"

GPU_MEM_IDLE_MAX_MB="${GPU_MEM_IDLE_MAX_MB:-99}"
GPU_POLL_SEC="${GPU_POLL_SEC:-30}"

gpu_used_mib() {
  local used
  used="$(nvidia-smi -i "$SMOKE_GPU_INDEX" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d '[:space:]')"
  echo "$used"
}

gpu_wait_log_default() {
  local root="${WORK_ROOT:-.}"
  echo "${root}/logs/superb/gpu_wait.log"
}

# Usage: wait_for_idle_gpu [log_file]
wait_for_idle_gpu() {
  local log_file="${1:-$(gpu_wait_log_default)}"
  mkdir -p "$(dirname "$log_file")"
  command -v nvidia-smi >/dev/null 2>&1 || {
    echo "ERROR: nvidia-smi not found; cannot check GPU ${SMOKE_GPU_INDEX}" >&2
    exit 1
  }
  echo "[GPU_WAIT] host=$(hostname) gpu=${SMOKE_GPU_INDEX} idle_threshold_mib<=$((GPU_MEM_IDLE_MAX_MB + 1)) poll_sec=${GPU_POLL_SEC} log=${log_file}"
  while :; do
    local used poll_msg
    used="$(gpu_used_mib)"
    [[ "$used" =~ ^[0-9]+$ ]] || {
      echo "ERROR: bad nvidia-smi memory.used for GPU ${SMOKE_GPU_INDEX}: '${used}'" >&2
      exit 1
    }
    poll_msg="$(date -Iseconds) [GPU_POLL] gpu=${SMOKE_GPU_INDEX} memory.used_mib=${used} (idle if <=${GPU_MEM_IDLE_MAX_MB})"
    echo "$poll_msg" | tee -a "$log_file"
    if [[ "$used" -le "$GPU_MEM_IDLE_MAX_MB" ]]; then
      echo "[GPU_READY] gpu=${SMOKE_GPU_INDEX} memory.used_mib=${used}"
      return 0
    fi
    sleep "$GPU_POLL_SEC"
  done
}
