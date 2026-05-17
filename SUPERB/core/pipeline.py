"""Upstream-only launcher (stage 1+2 bash recipes from configs/experiments.csv)."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

from core.experiments import run_row, select_rows
from core.paths import release_root
from core.registry import load_experiment_rows

ROOT = release_root()


def normalize_upstream_mode(mode: str, exp_id: str | None, upstream_entry: str | None) -> str:
    if mode != "auto":
        return mode
    if exp_id and "_magnitude" in exp_id:
        return "prune_then_ft"
    if upstream_entry and "mag" in upstream_entry.lower():
        return "prune_then_ft"
    return "distill_prune_ft"


def cmd_pipeline_run(args: argparse.Namespace) -> int:
    exp_id = getattr(args, "exp_id", None) or os.environ.get("UPSTREAM_EXP_ID") or os.environ.get(
        "STAGE1_EXP_ID", ""
    )
    exp_id = exp_id.strip() if exp_id else None
    upstream_entry = (getattr(args, "upstream_entry", None) or os.environ.get("UPSTREAM_ENTRY") or "").strip()

    mode = normalize_upstream_mode(args.upstream_mode, exp_id, upstream_entry or None)

    print("[INFO] Upstream exp id    :", exp_id or "not_set")
    print("[INFO] Upstream entry     :", upstream_entry or "not_set")
    print("[INFO] Upstream mode      :", mode)
    print("[INFO] Run upstream first :", args.run_upstream_first)
    print("[INFO] Dry run            :", int(args.dry_run))

    if not args.run_upstream_first:
        print(
            "ERROR: this release runs upstream training only; pass --run-upstream-first.",
            file=sys.stderr,
        )
        return 1

    if upstream_entry and exp_id:
        print("ERROR: use either --exp-id or --upstream-entry, not both.", file=sys.stderr)
        return 1
    if upstream_entry:
        upath = ROOT / upstream_entry
        if not upath.is_file():
            print(f"ERROR: upstream entry not found: {upath}", file=sys.stderr)
            return 1
        udir = upath.parent.resolve()
        if args.dry_run:
            print(f"[RUN ] upstream-entry -> bash -euo pipefail {upath.name} (cwd {udir})")
            return 0
        return subprocess.call(["bash", "-euo", "pipefail", str(upath)], cwd=str(udir))
    if exp_id:
        rows = select_rows(load_experiment_rows(), exp_id=exp_id)
        if not rows:
            print(f"ERROR: unknown exp-id: {exp_id}", file=sys.stderr)
            return 1
        return run_row(rows[0], dry_run=args.dry_run)

    print("ERROR: --run-upstream-first requires --exp-id or --upstream-entry.", file=sys.stderr)
    return 1


def build_pipeline_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Upstream SSL compression training (bash launchers only)")
    p.add_argument("--exp-id", default="", help="Experiment id in configs/experiments.csv")
    p.add_argument(
        "--upstream-entry",
        default="",
        help="Launcher path under release root, e.g. upstream_str/run_....sh",
    )
    p.add_argument("--run-upstream-first", action="store_true")
    p.add_argument(
        "--upstream-mode",
        default="auto",
        choices=["auto", "distill_prune_ft", "prune_then_ft"],
        help="Informational label for logs; launchers encode the real recipe.",
    )
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_pipeline_run)
    return p


def main_pipeline(argv: list[str] | None) -> int:
    parser = build_pipeline_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main_pipeline(sys.argv[1:]))
