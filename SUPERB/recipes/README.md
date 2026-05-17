# Recipes

In this release, a **recipe** is:

1. One row of **`configs/experiments.csv`** (`exp_id`, `method`, `source_repo`, `entry_script`, …), plus  
2. **`recipes/family_defaults.json`**, which defines default **`upstream_mode`** and descriptions per **family**.

The **`method`** column maps to **`family`** inside `core/registry.py`:

| method values | family |
|---------------|--------|
| `native` | `native` |
| `str_saspg`, `unstr_saspg` | `saspg` |
| `str_magnitude`, `unstr_magnitude` | `magnitude` |

Smoke matrix **`configs/smoke24.csv`** references either `--exp-id` (saspg/mag rows) or `--upstream-entry` (explicit DPHuBERT launcher paths). Those paths remain relative to the release root after vendoring.

To adjust family defaults, edit **`recipes/family_defaults.json`** (JSON, no extra Python deps).
