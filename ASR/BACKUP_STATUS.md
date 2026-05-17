# Pre-refactor backup

A full tree snapshot was taken **before** the open-source data-path refactor:

**`../SSLprune_ASR_release_backup_20260517_162518/`**

See that directory’s `BACKUP_STATUS.md` for size, exclusions, and restore instructions.

Current tree changes (summary):

- `data/librispeech/csv_metadata/` — relative-path manifests in-repo
- `utils/librispeech_data.py` — `LIBRISPEECH_AUDIO_ROOT`, `SSLPRUNE_DATASET_CACHE`
- Removed cluster hardcoded paths from `benchmark_tasks.py`, `main_prune.py`, `hf_models.py`
- `DATA.md` — LibriSpeech download and layout
