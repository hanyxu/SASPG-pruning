#!/usr/bin/env bash
# SUPERB smoke #8: hubert-large, SASPG structured, train100 (100h)
set -euo pipefail
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")/../user_sim/SUPERB" && pwd)/08_hubert_large_str_saspg_100h.sh" "$@"
