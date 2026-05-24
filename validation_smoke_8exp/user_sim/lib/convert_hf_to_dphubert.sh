#!/usr/bin/env bash
# Convert SUPERB upstream SSL HuBERT (hubert-base-ls960, hubert-large-ll60k) -> DPHuBERT .hf.pth.
# ASR uses separate fine-tuned checkpoints (ASR_W2V_MODEL_PATH, ASR_HUBERT_MODEL_PATH); not converted here.
# Full SUPERB grid may also reference SUPERB_HF_W2V_SSL / SUPERB_HF_WAVLM under HF_MODELS_SRC.
set -euo pipefail

convert_one() {
  local hf_dir="$1"
  local out_pth="$2"
  local model_kind="$3"  # hubert_base | hubert_large

  if [[ -f "$out_pth" ]]; then
    echo "[skip] exists: $out_pth"
    return 0
  fi

  echo "[convert] ${model_kind} <- ${hf_dir} -> ${out_pth}"
  python3 - <<PY
import torch
from pathlib import Path
from transformers import HubertConfig, HubertModel
from torchaudio.models.wav2vec2.utils import import_huggingface_model

hf_dir = Path("${hf_dir}")
out_pth = Path("${out_pth}")
kind = "${model_kind}"

config = HubertConfig.from_pretrained(str(hf_dir))
original = HubertModel(config)
bin_path = hf_dir / "pytorch_model.bin"
if bin_path.is_file():
    state = torch.load(bin_path, map_location="cpu", weights_only=True)
    original.load_state_dict(state)
else:
    original = HubertModel.from_pretrained(str(hf_dir), local_files_only=True)
imported = import_huggingface_model(original)

if kind == "hubert_base":
    config = dict(
        extractor_mode="group_norm",
        extractor_conv_layer_config=[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512, 2, 2)] * 2,
        extractor_conv_bias=False,
        encoder_embed_dim=768,
        encoder_projection_dropout=0.1,
        encoder_pos_conv_kernel=128,
        encoder_pos_conv_groups=16,
        encoder_num_layers=12,
        encoder_use_attention=[True] * 12,
        encoder_use_feed_forward=[True] * 12,
        encoder_num_heads=[12] * 12,
        encoder_head_dim=64,
        encoder_attention_dropout=0.1,
        encoder_ff_interm_features=[3072] * 12,
        encoder_ff_interm_dropout=0.0,
        encoder_dropout=0.1,
        encoder_layer_norm_first=False,
        encoder_layer_drop=0.05,
        aux_num_out=None,
        normalize_waveform=False,
        extractor_prune_conv_channels=False,
        encoder_prune_attention_heads=False,
        encoder_prune_attention_layer=False,
        encoder_prune_feed_forward_intermediate=False,
        encoder_prune_feed_forward_layer=False,
    )
else:
    config = dict(
        extractor_mode="layer_norm",
        extractor_conv_layer_config=[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512, 2, 2)] * 2,
        extractor_conv_bias=True,
        encoder_embed_dim=1024,
        encoder_projection_dropout=0.1,
        encoder_pos_conv_kernel=128,
        encoder_pos_conv_groups=16,
        encoder_num_layers=24,
        encoder_use_attention=[True] * 24,
        encoder_use_feed_forward=[True] * 24,
        encoder_num_heads=[16] * 24,
        encoder_head_dim=64,
        encoder_attention_dropout=0.1,
        encoder_ff_interm_features=[4096] * 24,
        encoder_ff_interm_dropout=0.0,
        encoder_dropout=0.1,
        encoder_layer_norm_first=True,
        encoder_layer_drop=0.0,
        aux_num_out=None,
        normalize_waveform=False,
        extractor_prune_conv_channels=False,
        encoder_prune_attention_heads=False,
        encoder_prune_attention_layer=False,
        encoder_prune_feed_forward_intermediate=False,
        encoder_prune_feed_forward_layer=False,
    )

out_pth.parent.mkdir(parents=True, exist_ok=True)
torch.save({"state_dict": imported.state_dict(), "config": config}, out_pth)
print(f"[OK] wrote {out_pth} ({out_pth.stat().st_size // (1024*1024)} MiB)")
PY
}

: "${HF_HUBERT_BASE:?set HF_HUBERT_BASE}"
: "${HF_HUBERT_LARGE:?set HF_HUBERT_LARGE}"

convert_one "${HF_HUBERT_BASE}" \
  "${DPHUBERT_PRETRAINED_DIR}/hubert-base-ls960.hf.pth" hubert_base

convert_one "${HF_HUBERT_LARGE}" \
  "${DPHUBERT_PRETRAINED_DIR}/hubert-large-ll60k.hf.pth" hubert_large
