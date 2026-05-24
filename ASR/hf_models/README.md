# Bundled Hugging Face checkpoints (smoke)

Two **ASR LibriSpeech-100h fine-tuning starting points** and upstream HuBERT-large SSL layout.
Weights use **Git LFS** — after clone: `git lfs install && git lfs pull`.

| Directory | Use |
|-----------|-----|
| `wav2vec2-base-100h/` | ASR smoke — Wav2Vec2 100h CTC baseline |
| `hubert-large-librispeech-100h/` | ASR smoke — HuBERT-large 100h CTC baseline (author FT ckpt) |
| `hubert-large-ll60k/` | Upstream SSL HuBERT-large (HF layout; teacher also in `SUPERB/pretrained/hubert-large-ll60k.hf.pth`) |

Set in `validation_smoke_8exp/user_sim/paths.env`:

```bash
export HF_MODELS_SRC="${SASPG_REPO}/ASR/hf_models"
export ASR_W2V_MODEL_PATH="${HF_MODELS_SRC}/wav2vec2-base-100h"
export ASR_HUBERT_MODEL_PATH="${HF_MODELS_SRC}/hubert-large-librispeech-100h"
```

LibriSpeech **audio** is not in the repo; **CSV metadata** is under `ASR/data/librispeech/csv_metadata/`.
