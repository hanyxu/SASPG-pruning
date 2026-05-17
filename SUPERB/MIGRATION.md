# Migration: thin shell scripts → `python3 -m core`

All previous `python3 scripts/*.py` entry points remain as **compatibility wrappers** but the supported API is the unified CLI.

## New commands (preferred)

| Old | New |
|-----|-----|
| `python3 scripts/exp_manager.py list` | `python3 -m core experiments list` |
| `python3 scripts/exp_manager.py run ...` | `python3 -m core experiments run ...` |
| `python3 scripts/exp_manager.py validate` | `python3 -m core experiments validate` |
| `python3 scripts/stage2_superb_manager.py list` | `python3 -m core downstream list` |
| `python3 scripts/stage2_superb_manager.py run ...` | `python3 -m core downstream run ...` |
| `python3 scripts/stage2_superb_manager.py validate` | `python3 -m core downstream validate` |
| `bash run_two_stage_pipeline.sh ...` | `python3 -m core pipeline ...` (same flags) |
| `python3 scripts/smoke24_runner.py ...` | `python3 -m core smoke24 ...` or `bash run_smoke24.sh` |

The root `bash` helpers (`run_experiment.sh`, `run_superb_stage2.sh`, `run_two_stage_pipeline.sh`, `run_smoke24.sh`, `list_*.sh`, `validate_*.sh`) now `exec` the `core` CLI. **No change to flags** for typical use.

## Environment variables (unchanged)

- `SASPG_WORKSPACE` — workspace for resolving `DPHuBERT*` paths (default: release root).
- `SASPG_S3PRL_ROOT` — override s3prl root for downstream.
- `UPSTREAM_CKPT` / `UPSTREAM_EXP_ID` / `STAGE1_EXP_ID` / `STAGE3_TASK` — still read by `core pipeline` when flags are omitted.

## Recipe model

- **Source of truth:** `configs/experiments.csv` (40 rows) + `recipes/family_defaults.json` (per-family defaults).
- **Enriched fields** (not extra CSV columns): `family`, `upstream_mode` are derived in `core/registry.py` from the `method` column.

## Preparing a GitHub clone

1. `bash ./prepare_local_dependencies.sh` (vendors `DPHuBERT*` from your full workspace, excluding large artifacts per `configs/rsync_upstream_excludes.txt`).
2. `python3 -m core experiments validate` and `python3 -m core downstream validate`.
3. For smoke: `python3 -m core smoke24 --dry-run` then real runs with a real `--upstream-ckpt`.
