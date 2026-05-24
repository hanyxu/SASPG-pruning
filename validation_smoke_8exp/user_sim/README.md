# 8-experiment smoke (new user, from zero)

> **Authors:** run **ASR smoke ×4** first — full steps in [`../../ASR/docs/SMOKE.md`](../../ASR/docs/SMOKE.md).  
> SUPERB smoke ×4 is optional on the same env: [`../../SUPERB/docs/SMOKE.md`](../../SUPERB/docs/SMOKE.md).

End-to-end validation of **4 ASR + 4 SUPERB** SASPG smoke jobs on LibriSpeech 100h.

**One conda environment** serves both tracks:

| Driver | Env name | Create script (auto picks) |
|--------|----------|----------------------------|
| NVIDIA 12.x | `test_saspg` | `00_create_conda_env.sh` |
| NVIDIA 11.x | `test_saspg_cu118` | `00_create_conda_env_cu118.sh` |

ASR uses Hugging Face (`transformers`, `datasets`). SUPERB smoke uses **PyTorch Lightning** in the same env (`SUPERB_USE_ASR_CONDA=1`; no separate `dphubert` env required).

## Prerequisites

- Linux GPU node with `conda`, `nvidia-smi`, enough disk for LibriSpeech + HF weights
- Clone [SASPG-pruning](https://github.com/hanyxu/SASPG-pruning) and `git lfs pull`
- LibriSpeech (`train-clean-100`, `dev-*`, `test-*`)
- Bundled ASR baselines under `ASR/hf_models/` after LFS (Wav2Vec2 100h + HuBERT-large 100h FT)

## 1. Configure paths

```bash
cd validation_smoke_8exp/user_sim
cp paths.env.example paths.env
# Edit: SASPG_REPO, LIBRISPEECH_ROOT, ASR_SMOKE_OUT_ROOT (writable), optional SMOKE_LOG_ROOT
```

Point `ASR_SMOKE_OUT_ROOT` / `SUPERB_*_WORK_ROOT` to a filesystem with quota if the clone lives on a small project volume.

## 2. Create unified conda env (on the GPU node)

```bash
bash 00_create_conda_env_auto.sh    # picks test_saspg vs test_saspg_cu118 from driver
bash lib/verify_smoke_env.sh        # torch CUDA + transformers + lightning
```

Recreate from scratch: `RECREATE_CONDA=1 bash 00_create_conda_env_auto.sh`

**Existing env — refresh pip only** (no conda recreate):

```bash
conda activate test_saspg   # or test_saspg_cu118
source lib/smoke_pip_install.sh
smoke_pip_install "$(pwd)/requirements-smoke.txt"
```

Pinned list: [`requirements-smoke.txt`](requirements-smoke.txt)

## 3. Prepare data & teachers

```bash
bash 01_prepare_data_and_models.sh
```

Symlinks LibriSpeech into `ASR/data/librispeech/audio`, builds SUPERB TSV, uses bundled `SUPERB/pretrained/*.hf.pth` and `ASR/hf_models/`.

## 4. Run smoke grid

| Step | Script | What |
|------|--------|------|
| **ASR SASPG ×4** | `02_run_asr_rotate_4.sh` | **Required** — wav2vec2-base + hubert-large × str/unstr |
| SUPERB SASPG ×4 | `03_run_superb_rotate_4.sh` | optional |
| SUPERB MAG ×4 | `04_run_superb_mag_rotate_4.sh` | optional |
| ASR MAG ×4 | `05_run_asr_mag_rotate_4.sh` | optional |
| ASR NASP str ×2 | `06_run_asr_nasp_str_rotate_2.sh` | optional |

**ASR-only quick path:**

```bash
bash 01_prepare_data_and_models.sh
bash 02_run_asr_rotate_4.sh
```

**Full pipeline (ASR + SUPERB):**

```bash
bash run_full_user_sim.sh
```

Set `SMOKE_GPU_INDEX`, `MAX_STEPS`, `SMOKE_SPARSITY_UNSTR` / `SMOKE_SPARSITY_STR` in `paths.env` or on the CLI.

## How ASR and SUPERB share one env without conflict

1. **Same conda** — `smoke_conda_activate()` in `lib/asr_conda_activate.sh` (default `SUPERB_USE_ASR_CONDA=1`).
2. **Separate write dirs** — `ASR_SMOKE_OUT_ROOT`, `SUPERB_SASPG_WORK_ROOT`, `SUPERB_MAG_WORK_ROOT` (and `WORK_ROOT` for symlinks/TSV).
3. **Serial GPU** — `WAIT_GPU_EACH_TASK=1` waits until `SMOKE_GPU_INDEX` is idle before each job.

## Troubleshooting

| Issue | Action |
|-------|--------|
| `conda env not found` | Run `00_create_conda_env_auto.sh` on **this** node (glibc must match) |
| `torch.cuda.is_available() False` after pip | Reinstall CUDA torch (see [`ASR/docs/SMOKE.md`](../../ASR/docs/SMOKE.md)) |
| SUPERB distill exit 1 but ckpt exists | See `lib/superb_smoke_from_launcher.sh` |
| Disk quota on `WORK_ROOT` | Move `ASR_SMOKE_OUT_ROOT` / `SUPERB_*_WORK_ROOT` in `paths.env` |
| Missing `[ASR smoke OK]` in log | Check tail of `${WORK_ROOT}/logs/asr/*.log` for Python traceback |

## File map

```
user_sim/
  paths.env.example     # template
  paths.env             # your machine (gitignored)
  requirements-smoke.txt
  00_create_conda_env_auto.sh
  01_prepare_data_and_models.sh
  02_run_asr_rotate_4.sh      # ASR SASPG ×4 (authors)
  03_run_superb_rotate_4.sh
  05_run_asr_mag_rotate_4.sh
  06_run_asr_nasp_str_rotate_2.sh
  run_full_user_sim.sh
  lib/
    asr_conda_activate.sh
    verify_smoke_env.sh
    superb_smoke_from_launcher.sh
../ASR/
  01_wav2vec2_base_unstr_saspg_100h.sh
  ...
```
