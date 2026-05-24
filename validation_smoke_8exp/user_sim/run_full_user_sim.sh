#!/usr/bin/env bash
# End-to-end new-user simulation: conda -> prepare -> ASR×4 -> SUPERB×4
# One conda env (test_saspg / test_saspg_cu118) for both ASR and SUPERB.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
chmod +x "${ROOT}"/*.sh "${ROOT}"/lib/*.sh "${ROOT}"/SUPERB/*.sh "${ROOT}"/../ASR/*.sh 2>/dev/null || true

[[ -f "${ROOT}/paths.env" ]] || {
  echo "ERROR: copy paths.env.example -> paths.env and edit paths" >&2
  exit 1
}

bash "${ROOT}/00_create_conda_env_auto.sh"
bash "${ROOT}/lib/verify_smoke_env.sh"
bash "${ROOT}/01_prepare_data_and_models.sh"
bash "${ROOT}/02_run_asr_rotate_4.sh"
# Optional MAG (4 = unstr+str × w2v+hubert) / NASP str-only (2):
# bash "${ROOT}/05_run_asr_mag_rotate_4.sh"
# bash "${ROOT}/06_run_asr_nasp_str_rotate_2.sh"
bash "${ROOT}/03_run_superb_rotate_4.sh"

echo "[DONE] full user simulation finished."
