#!/usr/bin/env bash
# Download Hugging Face checkpoints into hf_models/ (same layout as the release bundle).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p hf_models

if ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "Install: pip install -U huggingface_hub" >&2
  exit 1
fi

fetch() {
  local repo_id="$1"
  local dest="$2"
  echo "==> $repo_id -> hf_models/$dest"
  huggingface-cli download "$repo_id" \
    --local-dir "hf_models/$dest" \
    --local-dir-use-symlinks False
}

fetch facebook/wav2vec2-base              wav2vec2-base
fetch facebook/wav2vec2-base-960h         wav2vec2-base-100h
fetch facebook/hubert-large-ll60k         hubert-large-ll60k

echo "Done. Optional extras:"
echo "  fetch facebook/hubert-base-ls960 hubert-base-ls960"
echo "  fetch microsoft/wavlm-base-plus wavlm-base-plus"
