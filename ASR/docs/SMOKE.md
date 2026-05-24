# ASR smoke test (LibriSpeech 100h) — author guide

This is the **recommended entry** for reproducing SASPG ASR smoke on a fresh machine.
You run **four ASR jobs** (Wav2Vec2-base + HuBERT-large × structured/unstructured SASPG on LibriSpeech `train-clean-100`).

Repository: [hanyxu/SASPG-pruning](https://github.com/hanyxu/SASPG-pruning)

SUPERB upstream smoke (optional, same conda env): [`SUPERB/docs/SMOKE.md`](../../SUPERB/docs/SMOKE.md).

## What you need locally (not in git)

| Item | Notes |
|------|--------|
| **LibriSpeech** | `train-clean-100`, `dev-*`, `test-*` on disk |
| **GPU node** | `conda`, `nvidia-smi` |
| **Disk** | Writable dir for logs/ckpts (`ASR_SMOKE_OUT_ROOT`; default under `user_sim/work/`) |

**Already in git:** LibriSpeech **CSV** under `ASR/data/librispeech/csv_metadata/`, **2 ASR baselines** under `ASR/hf_models/` (Git LFS), smoke scripts under `validation_smoke_8exp/`.

**Not in git:** LibriSpeech **audio**, smoke **checkpoints** (`ASR_SMOKE_OUT_ROOT`, `ASR/smoke_100h_8exp/`, `user_sim/work/`).

## 0. Clone with LFS

```bash
git clone https://github.com/hanyxu/SASPG-pruning.git
cd SASPG-pruning
git lfs install
git lfs pull
```

Verify bundled ASR weights:

```bash
ls ASR/hf_models/wav2vec2-base-100h ASR/hf_models/hubert-large-librispeech-100h
```

## 1. Configure paths

```bash
cd validation_smoke_8exp/user_sim
cp paths.env.example paths.env
```

Edit at minimum:

| Variable | Purpose |
|----------|---------|
| `SASPG_REPO` | Absolute path to this clone |
| `LIBRISPEECH_ROOT` | LibriSpeech root (audio) |
| `ASR_SMOKE_OUT_ROOT` | Writable smoke outputs (use a filesystem with quota) |
| `SMOKE_LOG_ROOT` | Optional; defaults next to `WORK_ROOT` |

Bundled defaults use in-repo `ASR/hf_models/` for Wav2Vec2 and HuBERT-large **100h fine-tuned** baseline. Do **not** point `ASR_HUBERT_MODEL_PATH` at upstream `hubert-large-ll60k` (that is for SUPERB SSL only).

## 2. One conda env (ASR + optional SUPERB)

On the **GPU node** where you train:

```bash
bash 00_create_conda_env_auto.sh    # test_saspg (driver 12.x) or test_saspg_cu118 (11.x)
bash lib/verify_smoke_env.sh
```

Pinned deps: [`requirements-smoke.txt`](../../validation_smoke_8exp/user_sim/requirements-smoke.txt) (Transformers; **no** separate fairseq/`dphubert` env for ASR smoke).

## 3. Prepare data symlink

```bash
bash 01_prepare_data_and_models.sh
```

Symlinks LibriSpeech into `ASR/data/librispeech/audio` and refreshes HF paths under `WORK_ROOT` when needed.

## 4. Run ASR SASPG smoke ×4 (required)

```bash
export SMOKE_GPU_INDEX=0    # or set in paths.env
bash 02_run_asr_rotate_4.sh
```

| # | Experiment | Launcher |
|---|------------|----------|
| 1 | Wav2Vec2-base unstr | `validation_smoke_8exp/ASR/01_wav2vec2_base_unstr_saspg_100h.sh` |
| 2 | Wav2Vec2-base str | `validation_smoke_8exp/ASR/02_wav2vec2_base_str_saspg_100h.sh` |
| 3 | HuBERT-large unstr | `validation_smoke_8exp/ASR/03_hubert_large_unstr_saspg_100h.sh` |
| 4 | HuBERT-large str | `validation_smoke_8exp/ASR/04_hubert_large_str_saspg_100h.sh` |

Logs: `${WORK_ROOT}/logs/asr/*.log` (or `${SMOKE_LOG_ROOT}/asr/` if set).

Success: log ends with `[ASR smoke OK] <slot_name>` and `main_prune.py train-pruned` completes `MAX_STEPS` (default **50** in `paths.env`).

Outputs (gitignored): `${ASR_SMOKE_OUT_ROOT}/${MAX_STEPS}/<slot>/` — checkpoints, `out/pruned/` for structured runs, etc.

## 5. Optional: MAG smoke ×4

Prune-first magnitude export + single CTC finetune (does not change SASPG paths):

```bash
bash 05_run_asr_mag_rotate_4.sh
```

Logs: `${SMOKE_LOG_ROOT:-${WORK_ROOT}/logs}/asr_mag/*.log`

## 6. Optional: NASP structured smoke ×2

Seven-tier Gumbel ladder (structured only):

```bash
bash 06_run_asr_nasp_str_rotate_2.sh
```

## Environment variables (summary)

| Variable | Default purpose |
|----------|-----------------|
| `MAX_STEPS` | Training steps per smoke job (default 50) |
| `SMOKE_SPARSITY_UNSTR` / `SMOKE_SPARSITY_STR` | `sp50` / `sp90` tags → `--max-prune-ratio` |
| `SMOKE_TINY_BATCH` | Smaller batch for fast smoke (default 1) |
| `SMOKE_NUM_TRAIN_SAMPLES` | Cap train samples (default 64) |
| `ASR_CUDA_VARIANT` | `auto` picks cu121 vs cu118 from driver |
| `SMOKE_ASR_HOST` | Empty = no hostname lock; set to pin a cluster node |
| `WAIT_GPU_EACH_TASK` | Serial GPU jobs when sharing one GPU with SUPERB |

## Troubleshooting

| Issue | Action |
|-------|--------|
| `git lfs pull` missing weights | `.bin` files are pointer stubs; ASR will fail at load |
| `torch.cuda.is_available() False` | Reinstall CUDA torch for your driver; or use `00_create_conda_env_cu118.sh` on driver 11.x |
| Disk quota on repo FS | Point `ASR_SMOKE_OUT_ROOT` and `SMOKE_LOG_ROOT` to another mount |
| HuBERT path wrong | Use `hubert-large-librispeech-100h` (bundled), not `hubert-large-ll60k` |
| Only 2/4 ASR jobs in rotate script | Ensure `02_run_asr_rotate_4.sh` runs all four `_run` lines (not commented) |

Full pipeline (conda + prepare + ASR×4 + SUPERB×4): `bash validation_smoke_8exp/user_sim/run_full_user_sim.sh`.

More detail: [`validation_smoke_8exp/user_sim/README.md`](../../validation_smoke_8exp/user_sim/README.md).
