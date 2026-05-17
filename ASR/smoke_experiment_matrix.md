# 20-slot smoke experiment matrix

All outputs live under `smoke_runs_${MAX_STEPS}/`. Slot folder names encode the four factors:

| Factor | Values in slot name |
|--------|---------------------|
| **Method** | `saspg`, `mag`, `nasp` |
| **Prune mode** | `unstr` (mask×weight at inference) / `str` (smaller tensors after prune) |
| **Backbone** | `w2v` / `hubert` |
| **Target sparsity** | `sp50` (~50% pruned) / `sp90` (~90% pruned) |

## Semantics (canonical)

| Mode | Weight matrix shape | At inference | What you must ship |
|------|---------------------|--------------|-------------------|
| **unstr** | Unchanged (full size) | `effective_weight = weight * mask` | Full **weight** + **mask** (mask may be external files or learned buffers) |
| **str** | **Smaller** after prune | Use shrunk weights as-is | **Shrunk weight** checkpoint only; **no mask** |

- **unstr cannot** complete “real” pruning as shape reduction alone; sparsity is enforced by multiplying mask into the forward pass.
- **str can** complete real shape reduction (fewer rows/cols/heads/channels in `state_dict`).

## 20 slots

| # | Slot | Method | Mode | Backbone | Sparsity |
|---|------|--------|------|----------|----------|
| 01 | `01_saspg_unstr_w2v_sp50` | SASPG | unstr | wav2vec2 | 50% |
| 02 | `02_saspg_unstr_w2v_sp90` | SASPG | unstr | wav2vec2 | 90% |
| 03 | `03_saspg_unstr_hubert_sp50` | SASPG | unstr | HuBERT | 50% |
| 04 | `04_saspg_unstr_hubert_sp90` | SASPG | unstr | HuBERT | 90% |
| 05 | `05_saspg_str_w2v_sp50` | SASPG | str | wav2vec2 | 50% |
| 06 | `06_saspg_str_w2v_sp90` | SASPG | str | wav2vec2 | 90% |
| 07 | `07_saspg_str_hubert_sp50` | SASPG | str | HuBERT | 50% |
| 08 | `08_saspg_str_hubert_sp90` | SASPG | str | HuBERT | 90% |
| 09 | `09_mag_unstr_w2v_sp50` | Magnitude | unstr | wav2vec2 | 50% |
| 10 | `10_mag_unstr_w2v_sp90` | Magnitude | unstr | wav2vec2 | 90% |
| 11 | `11_mag_unstr_hubert_sp50` | Magnitude | unstr | HuBERT | 50% |
| 12 | `12_mag_unstr_hubert_sp90` | Magnitude | unstr | HuBERT | 90% |
| 13 | `13_mag_str_w2v_sp50` | Magnitude | str | wav2vec2 | 50% |
| 14 | `14_mag_str_w2v_sp90` | Magnitude | str | wav2vec2 | 90% |
| 15 | `15_mag_str_hubert_sp50` | Magnitude | str | HuBERT | 50% |
| 16 | `16_mag_str_hubert_sp90` | Magnitude | str | HuBERT | 90% |
| 17 | `17_nasp_w2v_sp50` | NASP | str | wav2vec2 | 50% |
| 18 | `18_nasp_w2v_sp90` | NASP | str | wav2vec2 | 90% |
| 19 | `19_nasp_hubert_sp50` | NASP | str | HuBERT | 50% |
| 20 | `20_nasp_hubert_sp90` | NASP | str | HuBERT | 90% |

NASP is **str-only** (4 slots). Magnitude and SASPG each cover 2×2×2 = 8 slots.

## Legacy `--reg-type` names (internal only)

Smoke scripts pass **clear aliases** to `main_prune.py` (`saspg_unstr`, `mag_str`, …). Those resolve to old implementation keys:

| Clear alias (`--reg-type`) | Legacy key | Used for |
|----------------------------|------------|----------|
| `saspg_unstr` + w2v | `saspg` | SASPG unstr, learned element mask in pruner |
| `saspg_unstr` + hubert | `saspg_hubert` | same, HuBERT |
| `saspg_str` + w2v | `channelpruning` | SASPG str: per-layer channel gate from target sparsity (no Gumbel `--value-*` ladder) |
| `saspg_str` + hubert | `channelpruninghubert` | same, HuBERT |
| `mag_unstr` + w2v | `mag_mask` | Magnitude unstr, external `mag_mask` × weight |
| `mag_unstr` + hubert | `mag_mask_hubert` | same, HuBERT |
| `mag_str` + w2v | `mag_mask` | Magnitude str (train); str export via `prune_ASR_*_mag.py` |
| `mag_str` + hubert | `mag_mask_hubert` | same |
| `nasp_str` + w2v | `channelpruning` | NASP str: **7-tier** Gumbel ratio ladder (`--value-1` … `--value-0075`) |
| `nasp_str` + hubert | `channelpruninghubert` | same |

**`saspg` is not a method name** — it is a historical label for the SASPG-unstr pruner module. Prefer the aliases above in new scripts and papers.

## Pipeline per (method, mode)

| Method | unstr | str |
|--------|-------|-----|
| **SASPG** | Single-stage train; mask×weight in pruner | Single-stage `--channel-pruning`; **no** Gumbel ratio ladder; post-train structural export (`pruned_attention_heads`, `pruned_ffn_inter`) |
| **Magnitude** | Stage1 mag train → Stage2: load stage1 weights + **fixed `mag_mask×weight` each forward** (no grad on masked weights); no structural `pruned/` | Stage1 → **`prune_ASR_*_mag.py`** (smaller shapes) → Stage2 from `checkpoint-*/pruned/` |
| **NASP** | — | Single-stage `channelpruning` + **7-tier** Gumbel `--value-*` ladder only |

## Structured (str) prune units

All str paths physically shrink only:

1. **Attention heads** per encoder layer (`num_heads` in `pruned_attention_heads`)
2. **FFN intermediate channels** per layer (`ff_interm_features` in `pruned_ffn_inter`)

Not structurally pruned in this release: CNN feature extractor, layer norms, CTC head, embedding tables (weights may be sliced with kept channels but there is no separate prune unit beyond the two above).

Implementation details: `smoke_experiment_lib.sh` + `smoke_20exp_500steps.sh`.
