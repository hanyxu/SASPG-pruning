#!/usr/bin/env bash
# Publish the full SASPG-pruning monorepo (SUPERB + ASR), not ASR-only.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
exec bash "$ROOT/scripts/publish_to_github.sh" "$@"
