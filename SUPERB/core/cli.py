"""Unified CLI: experiments | pipeline | smoke48."""

from __future__ import annotations

import argparse
import sys

from core.experiments import build_experiments_parser, rows_ready_matrix
from core.paths import experiments_csv_path
from core.pipeline import build_pipeline_parser, cmd_pipeline_run

SMOKE_CSV = experiments_csv_path()


def cmd_smoke48(args: argparse.Namespace) -> int:
    if not SMOKE_CSV.exists():
        print(f"ERROR: missing {SMOKE_CSV}", file=sys.stderr)
        return 2

    rows = rows_ready_matrix(
        data_hours=args.hours or None,
        model=args.model or None,
    )

    if args.limit:
        rows = rows[: args.limit]

    if len(rows) != 48 and not args.limit and not args.hours and not args.model:
        print(
            f"WARNING: expected 48 ready rows in experiments.csv, found {len(rows)} "
            "(use --limit / filters intentionally).",
            file=sys.stderr,
        )

    rc = 0
    for i, row in enumerate(rows, 1):
        exp_id = (row.get("exp_id") or "").strip()
        if not exp_id:
            print(f"[FAIL] row {i}: missing exp_id", file=sys.stderr)
            rc = 1
            continue

        ns = argparse.Namespace(
            exp_id=exp_id,
            upstream_entry="",
            run_upstream_first=True,
            upstream_mode="auto",
            dry_run=args.dry_run,
        )

        label = exp_id
        print(f"\n=== [{i}/{len(rows)}] {label} ===")
        cur = cmd_pipeline_run(ns)
        if cur != 0 and rc == 0:
            rc = cur
    return rc


def build_smoke48_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Sequential smoke: all ready rows in experiments.csv")
    p.add_argument("--exp-prefix", default="", help="Unused; reserved for log tagging.")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--hours", type=int, default=0, choices=[0, 100, 960])
    p.add_argument("--model", default="")
    p.set_defaults(func=cmd_smoke48)
    return p


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        print(
            "Usage: python -m core.cli <experiments|pipeline|smoke48> ...",
            file=sys.stderr,
        )
        return 2

    head, rest = argv[0], argv[1:]

    if head == "experiments":
        parser = build_experiments_parser()
        args = parser.parse_args(rest)
        return args.func(args)

    if head == "pipeline":
        parser = build_pipeline_parser()
        args = parser.parse_args(rest)
        return args.func(args)

    if head == "smoke48":
        parser = build_smoke48_parser()
        args = parser.parse_args(rest)
        return args.func(args)

    if head in ("smoke24", "downstream"):
        print(
            f"Command {head!r} was removed; use smoke48 (upstream only) or experiments run.",
            file=sys.stderr,
        )
        return 2

    print(f"Unknown command: {head}", file=sys.stderr)
    print("Try: experiments | pipeline | smoke48", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
