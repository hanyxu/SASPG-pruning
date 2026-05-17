# LibriSpeech data for SSLprune ASR

## What ships in this repo

| Path | Contents |
|------|----------|
| `data/librispeech/csv_metadata/*.csv` | Manifests (`file`, `text`). `file` is **relative** to the audio root, e.g. `dev-clean/2277/.../xxx.flac`. |
| `hf_models/` | Pretrained Wav2Vec2 / HuBERT checkpoints (`bash scripts/fetch_hf_models.sh`). |
| `cache/processed_audio/` | **Not** in git — created on first train (preprocessed features). |

Audio (**`.flac` files**) is **not** redistributed. You must obtain LibriSpeech under license and point the code at it.

## 1. Download LibriSpeech

1. Register / accept the license at [openslr.org/12](https://www.openslr.org/12.html).
2. Download and unpack the splits you need, e.g.:
   - `train-clean-100.tar.gz` → `--type 100`
   - full 960h (clean-100 + clean-360 + other-500) → `--type 960`
   - `dev-clean`, `dev-other`, `test-clean`, `test-other` for eval/test

Standard layout after unpack:

```text
LibriSpeech/
  train-clean-100/
  train-clean-360/    # 960h only
  train-other-500/      # 960h only
  dev-clean/
  dev-other/
  test-clean/
  test-other/
```

## 2. Point audio at this repo

Default audio root: `data/librispeech/audio/` (must contain `dev-clean/`, `test-clean/`, etc. **directly underneath**).

**Option A — symlink (recommended if you already have LibriSpeech elsewhere):**

```bash
cd SSLprune_ASR_release
rm -rf data/librispeech/audio
ln -sfn /path/to/LibriSpeech data/librispeech/audio
# audio/dev-clean/... must exist directly under audio/ (not audio/LibriSpeech/dev-clean)
```

**Option B — copy or unpack** flac trees into `data/librispeech/audio/`.

**Option C — custom root:**

```bash
export LIBRISPEECH_AUDIO_ROOT=/data/LibriSpeech
```

## 3. CSV manifests

Manifests live in `data/librispeech/csv_metadata/`:

- `train-clean-100.csv` / `train-960.csv` — training (`--type 100` or `960`)
- `dev-clean.csv`, `dev-other.csv`, `test-clean.csv`, `test-other.csv` — eval/test

To regenerate from another metadata tree:

```bash
python scripts/convert_librispeech_csv.py \
  --src-dir /path/to/old/csv_metadata \
  --dst-dir data/librispeech/csv_metadata
```

Override manifest directory only:

```bash
export SSLPRUNE_LIBRISPEECH_CSV_ROOT=/path/to/csv_metadata
```

## 4. Preprocessed cache (Arrow)

On first `train-pruned`, the code maps audio → `input_values` / `labels` and saves under:

```text
cache/processed_audio/processed_datasets_100.arrow   # --type 100
cache/processed_audio/processed_datasets_960.arrow   # --type 960
```

Override:

```bash
export SSLPRUNE_DATASET_CACHE=/scratch/my_asr_cache
```

Optional: persist length-filtered train split:

```bash
export SAVE_FILTERED_TRAIN_CACHE=1
```

**Note:** If you used an older cluster cache named `processed_datasets.arrow`, rename or symlink it to `processed_datasets_960.arrow`, or delete it and rebuild.

## 5. Quick train / smoke

```bash
export PYTHONPATH="$(pwd)"
# ensure data/librispeech/audio -> your LibriSpeech tree

python main_prune.py train-pruned \
  --cuda --model-name wav2vec2 --type 100 \
  --output-dir ./runs/demo --max-steps 10 \
  --reg-type saspg_unstr --max-prune-ratio 0.5 --min-prune-ratio 0.48
```

Smoke grid: `./smoke_20exp_500steps.sh` (same data layout).

## Backup before this layout

Snapshot before open-source data refactor:

`../SSLprune_ASR_release_backup_20260517_162518/` — see `BACKUP_STATUS.md` inside.
