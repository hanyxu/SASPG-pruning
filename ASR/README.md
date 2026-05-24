# SASPG-pruning (LibriSpeech ASR)

Open-source release for **SASPG / magnitude / NASP** pruning on LibriSpeech ASR with Wav2Vec2 and HuBERT CTC models.

## Smoke test (authors)

Reproduce the **four** LibriSpeech-100h SASPG smoke runs from a clean clone:

1. `git lfs pull` (bundled `hf_models/` baselines).
2. Follow **[`docs/SMOKE.md`](docs/SMOKE.md)** — `validation_smoke_8exp/user_sim` → `02_run_asr_rotate_4.sh`.

Optional: MAG ×4 (`05_run_asr_mag_rotate_4.sh`), NASP str ×2 (`06_run_asr_nasp_str_rotate_2.sh`).

## Features

- **Methods**: SASPG (unstructured & structured), magnitude pruning (prune-first + finetune), NASP (structured Gumbel ladder)
- **Backbones**: Wav2Vec2, HuBERT (see `utils/hf_models.py`)
- **Data**: CSV manifests in-repo; LibriSpeech **audio** must be obtained separately ([`DATA.md`](DATA.md))

## Manual training (advanced)

```bash
cd ASR
export PYTHONPATH="$(pwd)"

python main_prune.py train-pruned \
  --cuda --model-name wav2vec2 --type 100 \
  --output-dir ./runs/demo --max-steps 10 \
  --reg-type saspg_unstr --max-prune-ratio 0.5 --min-prune-ratio 0.48
```

## Layout

| Path | Role |
|------|------|
| `main_prune.py` | Training / pruning CLI |
| `mag_asr_prune_export.py` | MAG prune-first export |
| `models/`, `pruning/` | Prunable model & pruner modules |
| `utils/` | HF loading, LibriSpeech dataset, Click options |
| `data/librispeech/csv_metadata/` | Train/dev/test manifests |
| `hf_models/` | Bundled HF checkpoints (Git LFS) |
| `docs/SMOKE.md` | Author smoke guide |

## Acknowledgments

SUPERB-style upstream ideas and teacher formats build on **[DPHuBERT](https://github.com/pyf98/DPHuBERT)**.

## License

MIT — see [LICENSE](LICENSE). Third-party model weights follow their Hugging Face / Fairseq licenses.
