# SUPERB smoke test (LibriSpeech 100h) — author guide

This guide covers **SUPERB upstream** smoke on a fresh machine.
You run **four SUPERB jobs** (HuBERT-base/large × structured/unstructured).

**Authors:** start with **ASR smoke ×4** — [`ASR/docs/SMOKE.md`](../../ASR/docs/SMOKE.md) (`02_run_asr_rotate_4.sh`). ASR and SUPERB share the same conda env.

Repository: [hanyxu/SASPG-pruning](https://github.com/hanyxu/SASPG-pruning)

## What you need locally (not in git)

| Item | Notes |
|------|--------|
| **LibriSpeech** | `train-clean-100`, `dev-*`, `test-*` on disk |
| **GPU node** | `conda`, `nvidia-smi` |
| **Disk** | Writable dir for logs/ckpts (`SUPERB_SASPG_WORK_ROOT`, default under `user_sim/work/`) |

**Already in git:** LibriSpeech **CSV** under `ASR/data/librispeech/csv_metadata/`, **6 bundled checkpoints** (4× `SUPERB/pretrained/*.hf.pth` + 2× ASR under `ASR/hf_models/`), smoke scripts under `validation_smoke_8exp/`.

**Not in git:** LibriSpeech audio, smoke **output** ckpts (`exp_minmax_*`, `user_sim/work/`).

## 0. Clone with LFS

```bash
git clone https://github.com/hanyxu/SASPG-pruning.git
cd SASPG-pruning
git lfs install
git lfs pull
```

## 1. One conda env (ASR + SUPERB)

On the **GPU node** where you train:

```bash
cd validation_smoke_8exp/user_sim
cp paths.env.example paths.env
# Edit LIBRISPEECH_ROOT and SASPG_REPO (absolute path to repo root)

bash 00_create_conda_env_auto.sh    # test_saspg (driver 12.x) or test_saspg_cu118 (11.x)
bash lib/verify_smoke_env.sh
```

Pinned Python deps: [`requirements-smoke.txt`](../../validation_smoke_8exp/user_sim/requirements-smoke.txt) (Transformers + PyTorch Lightning; **no** separate `dphubert` env).

## 2. Prepare teachers & TSV

```bash
bash 01_prepare_data_and_models.sh
```

This symlinks LibriSpeech into `ASR/data/librispeech/audio`, builds SUPERB TSV under `work/superb/tsv`, and uses bundled `SUPERB/pretrained/*.hf.pth` (no manual convert required if files are present).

## 3. Run SUPERB smoke ×4

```bash
export SMOKE_GPU_INDEX=0    # or set in paths.env
bash 03_run_superb_rotate_4.sh
```

| # | Experiment | Launcher |
|---|------------|----------|
| 1 | HuBERT-base unstr | `user_sim/SUPERB/05_hubert_base_unstr_saspg_100h.sh` |
| 2 | HuBERT-base str | `user_sim/SUPERB/06_hubert_base_str_saspg_100h.sh` |
| 3 | HuBERT-large unstr | `user_sim/SUPERB/07_hubert_large_unstr_saspg_100h.sh` |
| 4 | HuBERT-large str | `user_sim/SUPERB/08_hubert_large_str_saspg_100h.sh` |

Logs: `${SUPERB_SASPG_WORK_ROOT}/logs/superb/*.log` (default `user_sim/work` is gitignored).

Success: log contains `[SUPERB smoke OK]` and distill/prune/final_distill reach smoke step limits (`SUPERB_SMOKE_MAX` / `SUPERB_SMOKE_FINAL_MAX` in `paths.env`).

## 4. Optional: MAG smoke ×4

```bash
bash 04_run_superb_mag_rotate_4.sh
```

## Environment variables (summary)

| Variable | Default purpose |
|----------|-----------------|
| `SUPERB_USE_ASR_CONDA=1` | SUPERB uses same env as ASR (`test_saspg`) |
| `SUPERB_SASPG_WORK_ROOT` | Writable SUPERB outputs (keep off small-quota FS) |
| `DPHuBERT_PRETRAINED_SRC` | `${SASPG_REPO}/SUPERB/pretrained` after prepare |
| `WAIT_GPU_EACH_TASK=1` | Serial GPU jobs — no ASR/SUPERB clash |

## Troubleshooting

- **`git lfs pull` missing weights** — `.hf.pth` files will be pointer stubs; training will fail immediately.
- **Disk quota** — point `SUPERB_SASPG_WORK_ROOT` to another filesystem in `paths.env`.
- **distill exit 1 but ckpt exists** — smoke launcher tolerates Lightning exit after `max_steps` (see `lib/superb_smoke_from_launcher.sh`).

Full pipeline (conda + prepare + ASR×4 + SUPERB×4): `bash validation_smoke_8exp/user_sim/run_full_user_sim.sh`.
