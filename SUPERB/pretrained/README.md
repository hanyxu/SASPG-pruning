# SUPERB upstream teacher checkpoints (smoke / 100h)

Four **DPHuBERT-format** teacher weights (`.hf.pth`) for LibriSpeech 100h upstream SASPG smoke.
Stored with **Git LFS** (`git lfs pull` after clone).

| File | Role |
|------|------|
| `hubert-base-ls960.hf.pth` | Teacher for hubert-base str/unstr smoke |
| `hubert-large-ll60k.hf.pth` | Teacher for hubert-large str/unstr smoke |
| `wav2vec2-base.hf.pth` | Wav2Vec2 SSL teacher (matrix / future smoke) |
| `wavlm-base-plus.hf.pth` | WavLM SSL teacher (matrix / future smoke) |

**Not included here (by design):** smoke-run outputs (`exp_minmax_*`, distilled/pruned ckpts under `validation_smoke_8exp/user_sim/work/`).

Point `DPHuBERT_PRETRAINED_SRC` or `WORK_ROOT/superb/pretrained` symlinks to this directory — see [`docs/SMOKE.md`](../docs/SMOKE.md).
