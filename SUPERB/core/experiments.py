"""48-experiment matrix: list / run / validate upstream bash launchers."""

from __future__ import annotations

import argparse
import subprocess
import sys

from core.registry import load_experiment_rows, resolve_upstream_script


def select_rows(
    rows: list[dict[str, str]],
    exp_id: str | None = None,
    model: str | None = None,
    data_hours: int | None = None,
    method: str | None = None,
    ready_only: bool = False,
) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for row in rows:
        if exp_id and row["exp_id"] != exp_id:
            continue
        if model and row["model"] != model:
            continue
        if data_hours and row["data_hours"] != str(data_hours):
            continue
        if method and row["method"] != method:
            continue
        if ready_only and (row.get("status") or "").strip() != "ready":
            continue
        out.append(row)
    return out


def rows_matching_filters(
    args: argparse.Namespace,
    *,
    ready_only: bool = False,
) -> list[dict[str, str]]:
    """Shared list/run/smoke matrix selection."""
    return select_rows(
        load_experiment_rows(),
        exp_id=getattr(args, "exp_id", None),
        model=getattr(args, "model", None) or None,
        data_hours=getattr(args, "data_hours", None) or None,
        method=getattr(args, "method", None) or None,
        ready_only=ready_only,
    )


def rows_ready_matrix(*, data_hours: int | None = None, model: str | None = None) -> list[dict[str, str]]:
    """Ready rows from experiments.csv (enriched like list/run)."""
    return select_rows(
        load_experiment_rows(),
        ready_only=True,
        data_hours=data_hours or None,
        model=model or None,
    )


def render_table(rows: list[dict[str, str]]) -> None:
    headers = [
        "exp_id",
        "model",
        "data_hours",
        "method",
        "family",
        "upstream_mode",
        "status",
        "source_repo",
        "entry_script",
    ]
    widths = {h: len(h) for h in headers}
    for row in rows:
        for h in headers:
            widths[h] = max(widths[h], len(row.get(h, "")))

    def fmt(r: dict[str, str]) -> str:
        return " | ".join(r.get(h, "").ljust(widths[h]) for h in headers)

    print(fmt({h: h for h in headers}))
    print("-+-".join("-" * widths[h] for h in headers))
    for row in rows:
        print(fmt(row))


def run_row(row: dict[str, str], dry_run: bool = False) -> int:
    script = resolve_upstream_script(row)
    if script is None:
        print(f"[SKIP] {row['exp_id']}: TODO entry")
        return 2
    if not script.exists():
        print(f"[SKIP] {row['exp_id']}: missing script -> {script}")
        return 2

    cmd = ["bash", "-euo", "pipefail", str(script)]
    print(f"[RUN ] {row['exp_id']} -> {' '.join(cmd)}")
    if dry_run:
        return 0
    return subprocess.call(cmd, cwd=script.parent)


def cmd_list(args: argparse.Namespace) -> int:
    rows = rows_matching_filters(args, ready_only=args.ready_only)
    if not rows:
        print("No experiments matched.")
        return 1
    render_table(rows)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    rows = rows_matching_filters(args, ready_only=args.ready_only)
    if not rows:
        print("No experiments matched.")
        return 1
    if args.exp_id is None and not args.all:
        print("Use --all for batch run or specify --exp-id.")
        return 1

    rc = 0
    for row in rows:
        row_rc = run_row(row, dry_run=args.dry_run)
        if row_rc != 0 and rc == 0:
            rc = row_rc
    return rc


def cmd_validate(_args: argparse.Namespace) -> int:
    rows = load_experiment_rows()
    missing: list[tuple[str, str]] = []
    for row in rows:
        script = resolve_upstream_script(row)
        if script is None:
            missing.append((row["exp_id"], "TODO"))
        elif not script.exists():
            missing.append((row["exp_id"], str(script)))
    if not missing:
        print("All experiment entries resolved.")
        return 0
    print("Unresolved experiment entries:")
    for exp_id, reason in missing:
        print(f"- {exp_id}: {reason}")
    return 2


def build_experiments_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="48-experiment matrix (recipe-enriched)")
    sub = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--exp-id")
    common.add_argument("--model")
    common.add_argument("--data-hours", type=int, choices=[100, 960])
    common.add_argument("--method")
    common.add_argument("--ready-only", action="store_true")

    p_list = sub.add_parser("list", parents=[common])
    p_list.set_defaults(func=cmd_list)

    p_run = sub.add_parser("run", parents=[common])
    p_run.add_argument("--all", action="store_true")
    p_run.add_argument("--dry-run", action="store_true")
    p_run.set_defaults(func=cmd_run)

    p_validate = sub.add_parser("validate")
    p_validate.set_defaults(func=cmd_validate)
    return p


def main_experiments(argv: list[str] | None = None) -> int:
    parser = build_experiments_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main_experiments())
