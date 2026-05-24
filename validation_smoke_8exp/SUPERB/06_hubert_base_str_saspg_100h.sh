#!/usr/bin/env bash
# SUPERB smoke #6: hubert-base, SASPG structured, train100 (100h)
set -euo pipefail
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")/../user_sim/SUPERB" && pwd)/06_hubert_base_str_saspg_100h.sh" "$@"
