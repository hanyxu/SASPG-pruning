#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [[ ! -d "$ROOT/upstream_str" || ! -d "$ROOT/upstream_unstr" ]]; then
  echo "[INFO] Missing upstream trees under $ROOT, preparing now..."
  bash "$ROOT/prepare_local_dependencies.sh"
fi

# Unified pipeline (see MIGRATION.md): python3 -m core pipeline ...
exec python3 -m core pipeline "$@"
