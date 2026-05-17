#!/usr/bin/env python3
"""Remove MaskedLinear and unstr_pruning=True branches; keep structural (else) paths only."""

from __future__ import annotations

import re
import sys
from pathlib import Path

KEEP_SUFFIX = ("str_modeling_wav2vec2_minmax_magnitude.py", "str_modeling_hubert_minmax_magnitude.py")

_UNSTR_IF = re.compile(r"if (not )?self\.unstr_pruning\b")
_CONFIG_UNSTR_IF = re.compile(r"if self\.config\.unstr_pruning\b")


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip())


def _dedent_block(lines: list[str], amount: int = 4) -> list[str]:
    out: list[str] = []
    for ln in lines:
        if ln.strip() == "":
            out.append(ln)
            continue
        n = _indent(ln)
        out.append(ln[amount:] if n >= amount else ln.lstrip("\n") if ln.strip() else ln)
    return out


def _read_block(lines: list[str], start: int, base_indent: int) -> tuple[list[str], int]:
    """Body of `if` at lines[start]; returns (body_lines, index_after_block)."""
    body: list[str] = []
    i = start + 1
    while i < len(lines):
        ln = lines[i]
        if ln.strip() == "":
            body.append(ln)
            i += 1
            continue
        cur = _indent(ln)
        stripped = ln.lstrip()
        if cur <= base_indent and stripped and not stripped.startswith("#"):
            if stripped.startswith(("elif ", "else:")):
                return body, i
            break
        body.append(ln)
        i += 1
    return body, i


def _read_else_block(lines: list[str], else_idx: int, base_indent: int) -> tuple[list[str], int]:
    i = else_idx + 1
    body: list[str] = []
    while i < len(lines):
        ln = lines[i]
        if ln.strip() == "":
            body.append(ln)
            i += 1
            continue
        cur = _indent(ln)
        if cur <= base_indent and ln.strip() and not ln.lstrip().startswith("#"):
            break
        body.append(ln)
        i += 1
    return body, i


def collapse_unstr_conditionals(text: str) -> str:
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        m = _UNSTR_IF.match(line.lstrip())
        if m:
            negated = m.group(1) is not None
            base = _indent(line)
            if_body, j = _read_block(lines, i, base)
            if j < len(lines) and lines[j].lstrip().startswith("else:"):
                else_body, j = _read_else_block(lines, j, base)
                kept = else_body if not negated else if_body
                out.extend(_dedent_block(kept))
            else:
                if negated:
                    out.extend(_dedent_block(if_body))
            i = j
            continue
        out.append(line)
        i += 1
    return "".join(out)


def strip_config_unstr_blocks(text: str) -> str:
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if _CONFIG_UNSTR_IF.match(line.lstrip()):
            base = _indent(line)
            _, j = _read_block(lines, i, base)
            if j < len(lines) and lines[j].lstrip().startswith("else:"):
                else_body, j = _read_else_block(lines, j, base)
                out.extend(_dedent_block(else_body))
            i = j
            continue
        out.append(line)
        i += 1
    return "".join(out)


def strip_masked_linear_class(text: str) -> str:
    return re.sub(
        r"\nclass MaskedLinear\(nn\.Module\):.*?(?=\n# Copied from|\nclass )",
        "\n",
        text,
        count=1,
        flags=re.DOTALL,
    )


def strip_unstr_params(text: str) -> str:
    text = re.sub(r",?\s*unstr_pruning: bool = False", "", text)
    text = re.sub(r",?\s*unstr_pruning=False", "", text)
    text = re.sub(r",?\s*unstr_pruning=config\.unstr_pruning", "", text)
    text = re.sub(r"\n\s*self\.unstr_pruning = unstr_pruning\n", "\n", text)
    text = text.replace("unstr:{self.config.unstr_pruning}", "str_export")
    text = re.sub(r"\n\s*config\.unstr_pruning\s*=\s*False\n", "\n", text)
    return text


def process(path: Path) -> None:
    original = path.read_text()
    text = strip_masked_linear_class(original)
    text = strip_unstr_params(text)
    text = collapse_unstr_conditionals(text)
    text = strip_config_unstr_blocks(text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    path.write_text(text)
    print(f"stripped: {path}")


def main() -> int:
    root = Path(__file__).resolve().parents[1] / "models_my"
    for name in KEEP_SUFFIX:
        p = root / name
        if not p.is_file():
            print(f"missing: {p}", file=sys.stderr)
            return 1
        process(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
