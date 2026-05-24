#!/usr/bin/env bash
# SUPERB smoke #5: hubert-base, SASPG unstructured, train100 (100h)
set -euo pipefail
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")/../user_sim/SUPERB" && pwd)/05_hubert_base_unstr_saspg_100h.sh" "$@"
