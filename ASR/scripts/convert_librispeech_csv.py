#!/usr/bin/env python3
"""Rewrite LibriSpeech CSV manifests to paths relative to data/librispeech/audio/."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

NAMES = (
    "train-960.csv",
    "train-clean-100.csv",
    "dev-clean.csv",
    "dev-other.csv",
    "test-clean.csv",
    "test-other.csv",
)


def to_relative_flac(path: str) -> str:
    path = path.strip()
    for marker in ("/LibriSpeech/", "\\LibriSpeech\\"):
        if marker in path:
            return path.split(marker, 1)[1].replace("\\", "/")
    p = Path(path)
    for part in ("dev-clean", "dev-other", "test-clean", "test-other", "train-clean-100", "train-clean-360", "train-other-500"):
        if part in p.parts:
            idx = p.parts.index(part)
            return "/".join(p.parts[idx:])
    raise ValueError(f"Cannot infer relative LibriSpeech path from: {path[:120]}...")


def convert_file(src: Path, dst: Path) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with src.open(newline="", encoding="utf-8") as fin, dst.open("w", newline="", encoding="utf-8") as fout:
        reader = csv.DictReader(fin)
        if "file" not in reader.fieldnames or "text" not in reader.fieldnames:
            raise ValueError(f"{src}: expected columns file,text got {reader.fieldnames}")
        writer = csv.DictWriter(fout, fieldnames=["file", "text"])
        writer.writeheader()
        for row in reader:
            writer.writerow({"file": to_relative_flac(row["file"]), "text": row["text"]})
            n += 1
    return n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src-dir",
        type=Path,
        default=Path(__file__).resolve().parents[3]
        / "superb_datasets/ls960/LibriSpeech/csv_metadata",
        help="Source csv_metadata directory (cluster or superb_datasets copy).",
    )
    parser.add_argument(
        "--dst-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "librispeech" / "csv_metadata",
    )
    args = parser.parse_args()
    src_dir = args.src_dir.resolve()
    dst_dir = args.dst_dir.resolve()
    for name in NAMES:
        src = src_dir / name
        if not src.is_file():
            print(f"skip missing {src}")
            continue
        n = convert_file(src, dst_dir / name)
        print(f"wrote {dst_dir / name} ({n} rows)")
    print(f"Done. Audio root should be: {dst_dir.parent / 'audio'}")


if __name__ == "__main__":
    main()
