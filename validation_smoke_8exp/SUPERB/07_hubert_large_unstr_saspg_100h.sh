#!/usr/bin/env bash
# SUPERB smoke #7: hubert-large, SASPG unstructured, train100 (100h)
set -euo pipefail
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")/../user_sim/SUPERB" && pwd)/07_hubert_large_unstr_saspg_100h.sh" "$@"
