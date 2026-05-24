# 8-experiment smoke (new user, from zero)

> **Authors:** you should run **SUPERB smoke ×4** — full steps in [`../../SUPERB/docs/SMOKE.md`](../../SUPERB/docs/SMOKE.md).

End-to-end validation of **4 ASR + 4 SUPERB** SASPG smoke jobs on LibriSpeech 100h.

**One conda environment** serves both tracks:

| Driver | Env name | Create script (auto picks) |
|--------|----------|----------------------------|
| NVIDIA 12.x | `test_saspg` | `00_create_conda_env.sh` |
| NVIDIA 11.x | `test_saspg_cu118` | `00_create_conda_env_cu118.sh` |

ASR uses Hugging Face (`transformers`, `datasets`). SUPERB upstream uses **PyTorch Lightning** (no separate `dphubert` / fairseq env for smoke).

## Prerequisites

- Linux GPU node with `conda`, `nvidia-smi`, enough disk for LibriSpeech + HF weights
- Clone [SASPG-pruning](https://github.com/your-org/SASPG-pruning) (this repo)
- LibriSpeech (`train-clean-100`, `dev-*`, `test-*`)
- HF bundle: at least `wav2vec2-base-100h`, upstream SSL teachers (`hubert-base-ls960`, `hubert-large-ll60k`, …)
- HuBERT-large **100h ASR fine-tuned** checkpoint (for ASR smoke; path differs from upstream `hubert-large-ll60k`)

## 1. Configure paths

```bash
cd validation_smoke_8exp/user_sim
cp paths.env.example paths.env
# Edit: HF_MODELS_SRC, LIBRISPEECH_ROOT, ASR_HUBERT_MODEL_PATH, SASPG_REPO, writable roots
```

Writable outputs default under `WORK_ROOT` (see `paths.env`). Point `ASR_SMOKE_OUT_ROOT` / `SUPERB_*_WORK_ROOT` to a filesystem with quota if needed.

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
# If torch lost CUDA after pip:
# pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

Pinned list: [`requirements-smoke.txt`](requirements-smoke.txt)

## 3. Prepare data & teachers

```bash
bash 01_prepare_data_and_models.sh
```

Symlinks LibriSpeech + HF into `ASR/`, builds SUPERB TSV, converts/copies `.hf.pth` teachers under `WORK_ROOT/superb/pretrained`.

## 4. Run smoke grid

| Step | Script | What |
|------|--------|------|
| ASR ×4 | `02_run_asr_rotate_4.sh` | wav2vec2-base + hubert-large × str/unstr |
| SUPERB SASPG ×4 | `03_run_superb_rotate_4.sh` | hubert-base/large × str/unstr |
| SUPERB MAG ×4 | `04_run_superb_mag_rotate_4.sh` | optional |
| ASR MAG ×4 | `05_run_asr_mag_rotate_4.sh` | optional |

**Full pipeline:**

```bash
bash run_full_user_sim.sh
```

Set `SMOKE_GPU_INDEX` and `MAX_STEPS` in `paths.env` or on the CLI.

## How ASR and SUPERB share one env without conflict

1. **Same conda** — `smoke_conda_activate()` in `lib/asr_conda_activate.sh` (default `SUPERB_USE_ASR_CONDA=1`).
2. **Separate write dirs** — `ASR_SMOKE_OUT_ROOT`, `SUPERB_SASPG_WORK_ROOT`, `SUPERB_MAG_WORK_ROOT` (and `WORK_ROOT` for symlinks/TSV).
3. **Serial GPU** — `WAIT_GPU_EACH_TASK=1` waits until `SMOKE_GPU_INDEX` is idle before each job.

Legacy separate SUPERB env (not recommended for smoke):

```bash
export SUPERB_USE_ASR_CONDA=0 SUPERB_CONDA_ENV=dphubert
```

## Troubleshooting

| Issue | Action |
|-------|--------|
| `conda env not found` | Run `00_create_conda_env_auto.sh` on **this** node (glibc must match) |
| `torch.cuda.is_available() False` after pip | Reinstall CUDA torch (see pip refresh above) |
| SUPERB distill exit 1 but ckpt exists | Fixed in `lib/superb_smoke_from_launcher.sh` (pipefail + ckpt check) |
| Disk quota on `WORK_ROOT` | Move `ASR_SMOKE_OUT_ROOT` / `SUPERB_*_WORK_ROOT` to another mount in `paths.env` |

## File map

```
user_sim/
  paths.env.example     # template
  paths.env             # your machine (not required in git if you use example only)
  requirements-smoke.txt
  00_create_conda_env_auto.sh
  01_prepare_data_and_models.sh
  02_run_asr_rotate_4.sh
  03_run_superb_rotate_4.sh
  run_full_user_sim.sh
  lib/
    asr_conda_activate.sh   # asr_conda_activate + smoke_conda_activate
    smoke_pip_install.sh
    verify_smoke_env.sh
    superb_smoke_from_launcher.sh
```
