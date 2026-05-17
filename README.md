# SASPG-pruning

Official release for **SASPG** structured / unstructured pruning on speech models.

| Directory | Contents |
|-----------|----------|
| [`SUPERB/`](SUPERB/) | Upstream DPHuBERT / SASPG experiment orchestrator (48-config matrix) |
| [`ASR/`](ASR/) | LibriSpeech ASR pruning & training (`main_prune.py`, smoke grid) |

## Quick links

- SUPERB: `cd SUPERB && python3 -m core experiments list`
- ASR: see [`ASR/README.md`](ASR/README.md) and [`ASR/DATA.md`](ASR/DATA.md)

## Layout note

Sources are maintained locally as `SASPG_superb_release_work` and `SSLprune_ASR_release`; this repository publishes them as `SUPERB` and `ASR`.
