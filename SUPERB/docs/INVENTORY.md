# Upstream code layout (duplication & shared core)

## Historical layout

The repository historically shipped three sibling directories:

- `DPHuBERT/` — distillation / pruning utilities and `s3prl/` SUPERB integration.
- `DPHuBERT_pretrain/` — structured / unstructured SASPG-style launchers (many `.sh` scripts).
- `DPHuBERT_pretrain_unstr/` — parallel unstr launchers and tooling.

These trees overlap heavily (e.g. `distill.py`, `lightning.py`, dataset helpers): differences are mostly **launcher scripts** and **sparse/unstructured hyper-parameters**, not fundamental algorithms.

## Unified orchestration (`core/`)

All GitHub-facing orchestration now lives under **`core/`**:

| Module | Role |
|--------|------|
| `core/registry.py` | Loads `configs/experiments.csv`, derives **`family`** (native / saspg / magnitude) and default **`upstream_mode`** |
| `core/experiments.py` | Resolves `bash` launchers per row |
| `core/downstream.py` | Wraps `DPHuBERT/s3prl/s3prl/run_downstream.py` |
| `core/pipeline.py` | Integrated upstream + SUPERB stage3 |
| `core/cli.py` | Single entry: `python3 -m core …` |

Upstream **training still runs the original bash scripts** inside those directories (reproducibility of numbers); `core` centralizes discovery, validation, and downstream wiring.

## Recipes (`recipes/`)

- **`recipes/family_defaults.json`**: per-family default `upstream_mode` and short descriptions.
- The **canonical experiment table** remains **`configs/experiments.csv`**; “recipe” metadata is the combination of CSV row + family defaults (see `recipes/README.md`).

## What is *not* deduplicated in this pass

A full merge of the three Python codebases (single `src/` with all models) is a larger refactor. This release **deduplicates orchestration** and documents shared vs per-launcher content; code-level merge can be a follow-up if you want a single import path for `lightning` / `distill`.
