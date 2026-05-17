#!/usr/bin/env bash
# Fail if any config.out under smoke_runs still passes --value-* (NASP-only flags).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SMOKE_ROOT="${1:-$ROOT/smoke_runs_500}"

fail=0
while IFS= read -r -d '' f; do
  slot="$(basename "$(dirname "$f")")"
  case "$slot" in
    17_*|18_*|19_*|20_*) continue ;;  # NASP str slots may use --value-*
  esac
  if grep -qE -- '--value-(1|075|05|025|0125|01|0075) [^0]' "$f" 2>/dev/null; then
    echo "NON-NASP config still has --value-*: $f"
    grep -E -- '--value-' "$f" || true
    fail=1
  fi
done < <(find "$SMOKE_ROOT" -name 'config.out' -print0 2>/dev/null)

if [[ "$fail" -eq 0 ]]; then
  echo "OK: no --value-* in non-NASP config.out under $SMOKE_ROOT"
else
  exit 1
fi
