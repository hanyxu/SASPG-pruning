#!/usr/bin/env bash
# Wait for ASR rotate (02) to finish, run SUPERB rotate (03), append debug trace to SMOKE_DEBUG_LOG.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=paths.env
source "${ROOT}/paths.env"

DEBUG_LOG="${SMOKE_DEBUG_LOG}"
mkdir -p "$(dirname "${DEBUG_LOG}")"

_ts() { date -Iseconds; }
_dbg() { echo "[$(_ts)] $*" | tee -a "${DEBUG_LOG}"; }

_dbg "=== smoke orchestrator start host=$(hostname) ==="
_dbg "WORK_ROOT=${WORK_ROOT}"
_dbg "SMOKE_GPU_INDEX=${SMOKE_GPU_INDEX} SUPERB_USE_ASR_CONDA=${SUPERB_USE_ASR_CONDA:-1}"

# shellcheck source=lib/asr_conda_activate.sh
source "${ROOT}/lib/asr_conda_activate.sh"
smoke_conda_activate || exit 1
export CUDA_VISIBLE_DEVICES="${SMOKE_GPU_INDEX}"
_dbg "shared conda env=${CONDA_ENV_NAME} (ASR+SUPERB)"
_dbg "torch $(python -c 'import torch; print(torch.__version__, "cuda_build", torch.version.cuda, "ok", torch.cuda.is_available())' 2>&1 || echo FAIL)"
_dbg "lightning $(python -c 'import pytorch_lightning as pl; from lightning_lite.utilities.rank_zero import _get_rank; print("pl", pl.__version__)' 2>&1 || echo FAIL)"

_wait_asr() {
  local _done="${WORK_ROOT}/logs/asr/.rotate_complete"
  local _pat='validation_smoke_8exp/user_sim/02_run_asr_rotate_4'
  _dbg "waiting for ASR rotate (marker ${_done} or local pgrep ${_pat}) ..."
  local _n=0
  while [[ ! -f "${_done}" ]]; do
    if pgrep -af "${_pat}" >/dev/null 2>&1; then
      : # local process still running
    elif (( _n > 2 )); then
      _dbg "no local ASR process and no marker yet (ASR may run on another node via NFS)"
    fi
    _n=$((_n + 1))
    if (( _n % 5 == 1 )); then
      _asr_log="${WORK_ROOT}/logs/asr"
      _dbg "ASR wait poll #${_n}; logs:"
      ls -lt "${_asr_log}"/*.log 2>/dev/null | head -4 | tee -a "${DEBUG_LOG}" || true
      _latest="$(ls -t "${_asr_log}"/*.log 2>/dev/null | head -1 || true)"
      if [[ -n "${_latest}" ]]; then
        tail -3 "${_latest}" 2>/dev/null | sed 's/^/  /' | tee -a "${DEBUG_LOG}" || true
      fi
    fi
    sleep 60
  done
  _dbg "ASR rotate complete: $(cat "${_done}" 2>/dev/null || true)"
}

_collect_asr_summary() {
  _dbg "--- ASR summary ---"
  local _d="${WORK_ROOT}/logs/asr"
  local _ok=0 _fail=0
  for _tag in 01_wav2vec2_base_unstr 02_wav2vec2_base_str 03_hubert_large_unstr 04_hubert_large_str; do
    local _lf="${_d}/${_tag}.log"
    if [[ ! -f "${_lf}" ]]; then
      _dbg "  [MISSING] ${_tag}.log"
      _fail=$((_fail + 1))
      continue
    fi
    if grep -qE 'Training complete|train_runtime|max_steps.*50/50|100%.*50/50' "${_lf}" 2>/dev/null \
      || grep -q '\[OK\]' "${_lf}" 2>/dev/null; then
      _dbg "  [OK?] ${_tag} ($(wc -l < "${_lf}") lines) tail:"
      tail -2 "${_lf}" | sed 's/^/    /' | tee -a "${DEBUG_LOG}"
      _ok=$((_ok + 1))
    elif grep -qiE 'error|traceback|CUDA out of memory|FAIL' "${_lf}" 2>/dev/null; then
      _dbg "  [FAIL] ${_tag} — see ${_lf}"
      tail -15 "${_lf}" | sed 's/^/    /' | tee -a "${DEBUG_LOG}"
      _fail=$((_fail + 1))
    else
      _dbg "  [?] ${_tag} ($(wc -l < "${_lf}") lines, check manually)"
      tail -5 "${_lf}" | sed 's/^/    /' | tee -a "${DEBUG_LOG}"
    fi
  done
  _dbg "ASR logs scanned ok~=${_ok} fail~=${_fail}"
}

_wait_asr
_collect_asr_summary

_dbg "========== SUPERB rotate 4 =========="
set +e
bash "${ROOT}/03_run_superb_rotate_4.sh" 2>&1 | tee -a "${DEBUG_LOG}"
_superb_rc=${PIPESTATUS[0]}
set -e

_dbg "--- SUPERB summary (exit=${_superb_rc}) ---"
_sd="${SUPERB_SASPG_WORK_ROOT}/logs/superb"
for _e in hubert_base_100_unstr_saspg hubert_base_100_str_saspg hubert_large_100_unstr_saspg hubert_large_100_str_saspg; do
  _lf="${_sd}/${_e}.log"
  if [[ ! -f "${_lf}" ]]; then
    _dbg "  [MISSING] ${_e}.log"
    continue
  fi
  if grep -qE 'SUPERB smoke OK|done training|train_inner' "${_lf}" 2>/dev/null \
    || grep -q 'max_updates=' "${_lf}" 2>/dev/null; then
    _dbg "  [ran] ${_e} ($(wc -l < "${_lf}") lines)"
  fi
  if grep -qiE 'traceback|error:|failed|No such file' "${_lf}" 2>/dev/null; then
    _dbg "  [errors in] ${_e}:"
    grep -iE 'traceback|error:|failed|No such file' "${_lf}" | tail -5 | sed 's/^/    /' | tee -a "${DEBUG_LOG}" || true
  fi
  tail -8 "${_lf}" 2>/dev/null | sed 's/^/    /' | tee -a "${DEBUG_LOG}" || true
done

_dbg "=== orchestrator done SUPERB_rc=${_superb_rc} debug_log=${DEBUG_LOG} ==="
exit "${_superb_rc}"
