"""LibriSpeech CSV manifests and local path helpers for open-source release."""

from __future__ import annotations

import os
from pathlib import Path

_RELEASE_ROOT = Path(__file__).resolve().parents[1]

CSV_DIRNAME = "csv_metadata"
AUDIO_DIRNAME = "audio"


def release_root() -> Path:
    return Path(os.environ.get("SSLPRUNE_RELEASE_ROOT", _RELEASE_ROOT))


def csv_metadata_dir() -> Path:
    override = os.environ.get("SSLPRUNE_LIBRISPEECH_CSV_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return release_root() / "data" / "librispeech" / CSV_DIRNAME


def audio_root() -> Path:
    override = os.environ.get("LIBRISPEECH_AUDIO_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return release_root() / "data" / "librispeech" / AUDIO_DIRNAME


def dataset_cache_dir() -> Path:
    override = os.environ.get("SSLPRUNE_DATASET_CACHE")
    if override:
        return Path(override).expanduser().resolve()
    return release_root() / "cache" / "processed_audio"


def csv_path(name: str) -> str:
    path = csv_metadata_dir() / name
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing LibriSpeech manifest: {path}. "
            "See DATA.md for download and directory layout."
        )
    return str(path)


def train_csv_name(data_type: str) -> str:
    if data_type == "960":
        return "train-960.csv"
    if data_type == "100":
        return "train-clean-100.csv"
    raise ValueError(f"Unsupported data.type {data_type!r}; use '100' or '960'.")


def resolve_audio_path(file_field: str) -> str:
    """Turn a manifest path (relative or legacy absolute) into an on-disk flac path."""
    if os.path.isabs(file_field) and os.path.isfile(file_field):
        return file_field
    marker = "/LibriSpeech/"
    if marker in file_field:
        file_field = file_field.split(marker, 1)[1]
    file_field = file_field.lstrip("/")
    return str(audio_root() / file_field)
