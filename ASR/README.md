# SASPG-pruning (LibriSpeech ASR)

Open-source release for **SASPG / magnitude / NASP** pruning on LibriSpeech ASR with Wav2Vec2 and HuBERT CTC models.

## Features

- **Methods**: SASPG (unstructured & structured), magnitude pruning, NASP (structured Gumbel ladder)
- **Backbones**: Wav2Vec2, HuBERT (see `utils/hf_models.py`)
- **Smoke grid**: 20 slots — method × unstr/str × backbone × sparsity — see [`smoke_experiment_matrix.md`](smoke_experiment_matrix.md)
- **Data**: CSV manifests in-repo; LibriSpeech **audio** must be obtained separately ([`DATA.md`](DATA.md))

## Quick start

```bash
cd SSLprune_ASR_release   # or clone repo root if published flat

# 1) Pretrained weights (not in git — ~3 GB)
bash scripts/fetch_hf_models.sh

# 2) LibriSpeech audio (license: openslr.org/12)
ln -sfn /path/to/LibriSpeech data/librispeech/audio

# 3) Optional env
cp env.example env.sh && source env.sh

export PYTHONPATH="$(pwd)"

python main_prune.py train-pruned \
  --cuda --model-name wav2vec2 --type 100 \
  --output-dir ./runs/demo --max-steps 10 \
  --reg-type saspg_unstr --max-prune-ratio 0.5 --min-prune-ratio 0.48
```

Full smoke grid (500 steps per slot):

```bash
./smoke_20exp_500steps.sh
```

## Layout

| Path | Role |
|------|------|
| `main_prune.py` | Training / pruning CLI |
| `prune_ASR_*_mag.py` | Magnitude structural export |
| `models/`, `pruning/` | Prunable model & pruner modules |
| `utils/` | HF loading, LibriSpeech dataset, Click options |
| `data/librispeech/csv_metadata/` | Train/dev/test manifests |
| `hf_models/` | Local HF checkpoints (via `scripts/fetch_hf_models.sh`) |
| `scripts/` | Data conversion, GitHub publish helper |

## Publish to GitHub

Maintainers: set `GITHUB_TOKEN` (repo scope) or run `gh auth login`, then:

```bash
bash scripts/publish_to_github.sh
```

Default remote: `https://github.com/hanyxu/SASPG-pruning.git`

## License

MIT — see [LICENSE](LICENSE). Third-party model weights follow their Hugging Face / Fairseq licenses.
