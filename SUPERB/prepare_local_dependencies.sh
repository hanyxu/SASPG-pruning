#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_ROOT="${SASPG_SOURCE_ROOT:-$(cd "$ROOT/../.." && pwd)}"
EXCLUDES="$ROOT/configs/rsync_upstream_excludes.txt"

if [[ ! -f "$EXCLUDES" ]]; then
  echo "ERROR: missing exclude file: $EXCLUDES" >&2
  exit 1
fi

copy_repo() {
  local src_name="$1"
  local dst_name="$2"
  local src="$SOURCE_ROOT/$src_name"
  local dst="$ROOT/$dst_name"
  if [[ ! -d "$src" ]]; then
    echo "ERROR: missing source directory: $src" >&2
    exit 1
  fi
  mkdir -p "$dst"
  rsync -a --delete --exclude-from="$EXCLUDES" "$src/" "$dst/"
  echo "[OK] $src_name -> $dst_name"
}

copy_repo "DPHuBERT_pretrain" "upstream_str"
copy_repo "DPHuBERT_pretrain_unstr" "upstream_unstr"

echo "Merged dependencies are ready under: $ROOT (upstream only; s3prl not bundled)"
