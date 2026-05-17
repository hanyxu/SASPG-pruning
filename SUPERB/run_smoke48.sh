#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [[ ! -d "$ROOT/upstream_str" || ! -d "$ROOT/upstream_unstr" ]]; then
  bash "$ROOT/prepare_local_dependencies.sh"
fi

exec python3 -m core smoke48 "$@"
