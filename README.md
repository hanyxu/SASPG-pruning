# SASPG-pruning

Official release for **SASPG** structured / unstructured pruning on speech models.

| Directory | Contents |
|-----------|----------|
| [`SUPERB/`](SUPERB/) | Upstream DPHuBERT / SASPG experiment orchestrator (48-config matrix) |
| [`ASR/`](ASR/) | LibriSpeech ASR pruning & training (`main_prune.py`, smoke grid) |
| [`validation_smoke_8exp/`](validation_smoke_8exp/) | End-to-end smoke harness (shared `test_saspg` conda env) |

## Start here: SUPERB smoke (recommended for authors)

Reproduce **four** LibriSpeech-100h upstream smoke runs (HuBERT-base/large × str/unstr) from a clean clone:

1. **`git lfs pull`** — fetch bundled checkpoints (`SUPERB/pretrained/*.hf.pth`, `ASR/hf_models/*`).
2. Follow **[`SUPERB/docs/SMOKE.md`](SUPERB/docs/SMOKE.md)** (conda → prepare → `03_run_superb_rotate_4.sh`).
3. Configure paths: `cp validation_smoke_8exp/user_sim/paths.env.example validation_smoke_8exp/user_sim/paths.env` and set `LIBRISPEECH_ROOT`.

**In the repo:** smoke scripts, LibriSpeech **CSV** metadata, **6 model checkpoints** (4 SUPERB teachers + 2 ASR baselines).  
**Not in the repo:** LibriSpeech audio, smoke **training outputs** (`exp_minmax_*`, `user_sim/work/`).

### One conda environment (`test_saspg`)

ASR and SUPERB smoke share **`test_saspg`** (CUDA 12.x) or **`test_saspg_cu118`** (CUDA 11.x):

```bash
cd validation_smoke_8exp/user_sim
bash 00_create_conda_env_auto.sh
bash lib/verify_smoke_env.sh
```

See [`validation_smoke_8exp/user_sim/README.md`](validation_smoke_8exp/user_sim/README.md) for the full 8-experiment grid (ASR optional).

## Other links

- SUPERB matrix: `cd SUPERB && python3 -m core experiments list`
- ASR data layout: [`ASR/DATA.md`](ASR/DATA.md)
- Checkpoints: [`SUPERB/pretrained/README.md`](SUPERB/pretrained/README.md), [`ASR/hf_models/README.md`](ASR/hf_models/README.md)

## Layout note

Sources are maintained locally as `SASPG_superb_release_work` and `SSLprune_ASR_release`; this repository publishes them as `SUPERB` and `ASR`.
