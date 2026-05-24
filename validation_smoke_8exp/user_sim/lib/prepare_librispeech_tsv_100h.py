#!/usr/bin/env python3
"""Create train100.tsv + valid.tsv only (skip full 960h scan). Matches upstream prepare_data.py format."""

from __future__ import annotations

import argparse
from pathlib import Path

import torchaudio
from tqdm import tqdm


def write_subset(root: Path, out: Path, patterns: list[str], name: str) -> int:
    files: list[Path] = []
    for pat in patterns:
        files.extend(sorted(root.glob(pat)))
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w") as f:
        print(root, file=f)
        for fname in tqdm(files, desc=name):
            rel = fname.relative_to(root)
            frames = torchaudio.info(fname).num_frames
            print(f"{rel}\t{frames}", file=f)
            n += 1
    return n


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    root = args.data.resolve()
    out = args.out.resolve()
    assert root.is_dir(), root

    n100 = write_subset(
        root,
        out / "train100.tsv",
        ["train-clean-100/**/*.flac"],
        "train100",
    )
    nval = write_subset(
        root,
        out / "valid.tsv",
        ["dev-clean/**/*.flac", "dev-other/**/*.flac"],
        "valid",
    )
    # train960 placeholder (not used for 100h smoke; tiny stub if missing)
    stub = out / "train960.tsv"
    if not stub.exists():
        with stub.open("w") as f:
            print(root, file=f)
    print(f"[OK] train100={n100} valid={nval} -> {out}")


if __name__ == "__main__":
    main()
