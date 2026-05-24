#!/usr/bin/env bash
# Pick test_saspg (cu121) vs test_saspg_cu118 (cu118) from driver version; activate conda.
# Source after paths.env:  source "${ROOT}/lib/asr_conda_activate.sh" && asr_conda_activate

asr_nvidia_driver_version() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return 1
  fi
  nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | tr -d ' '
}

# Prints conda env name to stdout.
asr_resolve_conda_env_name() {
  if [[ -n "${ASR_CONDA_ENV_FORCE:-}" ]]; then
    echo "${ASR_CONDA_ENV_FORCE}"
    return 0
  fi

  local cu121="${CONDA_ENV_NAME_CU121:-test_saspg}"
  local cu118="${CONDA_ENV_NAME_CU118:-test_saspg_cu118}"
  local variant="${ASR_CUDA_VARIANT:-auto}"

  case "$variant" in
    cu121|12) echo "$cu121"; return 0 ;;
    cu118|11) echo "$cu118"; return 0 ;;
    auto) ;;
    *)
      echo "ERROR: unknown ASR_CUDA_VARIANT=${variant} (use auto, cu121, cu118)" >&2
      return 1
      ;;
  esac

  local ver
  ver="$(asr_nvidia_driver_version || true)"
  if [[ -z "$ver" ]]; then
    echo "WARN: nvidia-smi unavailable; defaulting to ${cu121}" >&2
    echo "$cu121"
    return 0
  fi

  local major="${ver%%.*}"
  if (( major >= 12 )); then
    echo "$cu121"
  elif (( major >= 11 )); then
    echo "$cu118"
  else
    echo "WARN: driver ${ver} < 11; trying ${cu118}" >&2
    echo "$cu118"
  fi
}

asr_conda_activate() {
  if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda not found on PATH" >&2
    return 1
  fi
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"

  local chosen
  chosen="$(asr_resolve_conda_env_name)" || return 1
  export CONDA_ENV_NAME="$chosen"

  if ! conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV_NAME}"; then
    local ver
    ver="$(asr_nvidia_driver_version || echo unknown)"
    echo "ERROR: conda env '${CONDA_ENV_NAME}' not found (driver=${ver}, host=$(hostname -s))." >&2
    if [[ "${CONDA_ENV_NAME}" == *cu118* ]] || [[ "${CONDA_ENV_NAME}" == "${CONDA_ENV_NAME_CU118:-test_saspg_cu118}" ]]; then
      echo "  Create cu118 env on this node: bash user_sim/00_create_conda_env_cu118.sh" >&2
    else
      echo "  Create cu121 env on this node: bash user_sim/00_create_conda_env.sh" >&2
    fi
    return 1
  fi

  conda activate "${CONDA_ENV_NAME}"
  export PYTHON="${CONDA_PREFIX}/bin/python"
  echo "[ASR] conda env=${CONDA_ENV_NAME} host=$(hostname -s) driver=$(asr_nvidia_driver_version || echo n/a)"
}

# Shared conda for ASR + SUPERB smoke (default). Override: SUPERB_USE_ASR_CONDA=0 SUPERB_CONDA_ENV=dphubert
smoke_conda_activate() {
  if [[ "${SUPERB_USE_ASR_CONDA:-1}" == "1" ]]; then
    asr_conda_activate || return 1
    export SUPERB_CONDA_ENV="${CONDA_ENV_NAME}"
    echo "[smoke] SUPERB uses ASR conda env=${CONDA_ENV_NAME}"
    return 0
  fi

  if [[ -z "${SUPERB_CONDA_ENV:-}" ]]; then
    echo "ERROR: SUPERB_USE_ASR_CONDA=0 but SUPERB_CONDA_ENV is unset" >&2
    return 1
  fi
  if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda not found on PATH" >&2
    return 1
  fi
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${SUPERB_CONDA_ENV}"
  export CONDA_ENV_NAME="${SUPERB_CONDA_ENV}"
  export PYTHON="${CONDA_PREFIX}/bin/python"
  echo "[smoke] SUPERB legacy conda env=${SUPERB_CONDA_ENV} host=$(hostname -s)"
}
