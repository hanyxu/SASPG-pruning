# SASPG-pruning

Official release for **SASPG** structured / unstructured pruning on speech models.

| Directory | Contents |
|-----------|----------|
| [`SUPERB/`](SUPERB/) | Upstream DPHuBERT-style SASPG experiment orchestrator (48-config matrix) |
| [`ASR/`](ASR/) | LibriSpeech ASR pruning & training (`main_prune.py`, smoke grid) |
| [`validation_smoke_8exp/`](validation_smoke_8exp/) | End-to-end smoke harness (shared `test_saspg` conda env) |

## Start here: ASR smoke (recommended for authors)

Reproduce **four** LibriSpeech-100h ASR smoke runs (Wav2Vec2-base + HuBERT-large × str/unstr SASPG) from a clean clone:

1. **`git lfs pull`** — fetch bundled checkpoints (`ASR/hf_models/*`, `SUPERB/pretrained/*.hf.pth`).
2. Follow **[`ASR/docs/SMOKE.md`](ASR/docs/SMOKE.md)** (conda → prepare → `02_run_asr_rotate_4.sh`).
3. Configure paths: `cp validation_smoke_8exp/user_sim/paths.env.example validation_smoke_8exp/user_sim/paths.env` and set `LIBRISPEECH_ROOT`, `SASPG_REPO`, and writable `ASR_SMOKE_OUT_ROOT`.

**In the repo:** smoke scripts, LibriSpeech **CSV** metadata, **6 model checkpoints** (4 SUPERB teachers + 2 ASR baselines).  
**Not in the repo:** LibriSpeech audio, smoke **training outputs** (`ASR_SMOKE_OUT_ROOT`, `user_sim/work/`, `ASR/smoke_100h_8exp/`).

### One conda environment (`test_saspg`)

ASR and SUPERB smoke share **`test_saspg`** (CUDA 12.x) or **`test_saspg_cu118`** (CUDA 11.x):

```bash
cd validation_smoke_8exp/user_sim
bash 00_create_conda_env_auto.sh
bash lib/verify_smoke_env.sh
```

See [`validation_smoke_8exp/user_sim/README.md`](validation_smoke_8exp/user_sim/README.md) for the full grid (ASR SASPG×4 required; SUPERB×4, MAG, NASP optional).

## SUPERB smoke (optional)

Upstream LibriSpeech-100h smoke: **[`SUPERB/docs/SMOKE.md`](SUPERB/docs/SMOKE.md)** → `03_run_superb_rotate_4.sh`.

## Other links

- SUPERB matrix: `cd SUPERB && python3 -m core experiments list`
- ASR data layout: [`ASR/DATA.md`](ASR/DATA.md)
- Checkpoints: [`SUPERB/pretrained/README.md`](SUPERB/pretrained/README.md), [`ASR/hf_models/README.md`](ASR/hf_models/README.md)

## Acknowledgments

SUPERB upstream training follows the distillation/pruning pipeline popularized by **[DPHuBERT](https://github.com/pyf98/DPHuBERT)** (joint distillation and structured pruning of SSL speech models). We thank the authors for open-sourcing their code and pretrained teachers.

## Layout note

Sources are maintained locally as `SASPG_superb_release_work` and `SSLprune_ASR_release`; this repository publishes them as `SUPERB` and `ASR`.
