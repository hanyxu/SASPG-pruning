#!/usr/bin/env bash
# Run one SUPERB 100h MAG (magnitude) smoke: prune_mag + final_distill_mag (no SASPG distill).
# Usage: superb_smoke_mag_from_launcher.sh <exp_id>
set -euo pipefail

_exp_id="${1:?exp_id}"

_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_usim_root="$(cd "${_script_dir}/.." && pwd)"
# shellcheck source=../paths.env
source "${_usim_root}/paths.env"

# shellcheck source=asr_conda_activate.sh
source "${_script_dir}/asr_conda_activate.sh"
smoke_conda_activate || exit 1

SUPERB_ROOT="${SASPG_REPO}/SUPERB"
export SASPG_ROOT="${SASPG_ROOT:-$(cd "${SASPG_REPO}/../.." && pwd)}"

_mag_pretrained="${SUPERB_MAG_WORK_ROOT}/superb/pretrained"
mkdir -p "${_mag_pretrained}" "${SUPERB_MAG_WORK_ROOT}/logs/superb_mag"
_src_pretrained="${WORK_ROOT}/superb/pretrained"
for _f in hubert-base-ls960.hf.pth hubert-large-ll60k.hf.pth; do
  if [[ -e "${_src_pretrained}/${_f}" && ! -e "${_mag_pretrained}/${_f}" ]]; then
    ln -sf "${_src_pretrained}/${_f}" "${_mag_pretrained}/${_f}"
  fi
done

export DPHuBERT_PRETRAINED_DIR="${_mag_pretrained}"
export DPHuBERT_PRETRAINED_ROOT="${DPHuBERT_PRETRAINED_DIR}"
export DPHuBERT_TSV_DIR="${WORK_ROOT}/superb/tsv"
export CUDA_VISIBLE_DEVICES="${SMOKE_GPU_INDEX}"
export PYTHONUNBUFFERED=1

_resolve_launcher() {
  python3 - <<PY
import csv
from pathlib import Path
exp_id = "${_exp_id}"
csv_path = Path("${SUPERB_ROOT}") / "configs" / "experiments.csv"
with csv_path.open() as f:
    for row in csv.DictReader(f):
        if row["exp_id"] == exp_id:
            print(row["source_repo"].strip())
            print(row["entry_script"].strip())
            break
    else:
        raise SystemExit(f"unknown exp_id: {exp_id}")
PY
}

mapfile -t _meta < <(_resolve_launcher)
_family="${_meta[0]}"
_entry="${_meta[1]}"
_src="${SUPERB_ROOT}/${_family}/${_entry}"
[[ -f "${_src}" ]] || { echo "ERROR: missing launcher ${_src}" >&2; exit 1; }

_upstream_dir="${SUPERB_ROOT}/${_family}"
export SUPERB_MAG_EXP_ROOT="${SUPERB_MAG_WORK_ROOT}/exp/${_family}"
mkdir -p "${SUPERB_MAG_EXP_ROOT}"
_dst="${_upstream_dir}/.smoke_${_exp_id}.sh"

# Short final_distill steps only; path fixes. Do NOT use global 's/^#     /' (breaks commented distill block).
sed \
  -e "s/^final_max=[0-9]*/final_max=${SUPERB_SMOKE_FINAL_MAX}/" \
  -e "s/^final_warmup=[0-9]*/final_warmup=${SUPERB_SMOKE_WARMUP}/" \
  -e '/^# save final model/,/^.*exit 1;/s/^/# /' \
  -e '/^    --config_path \${pruned_ckpt}/,/^.*exit 1;/s/^/# /' \
  -e '/^rm -rf \${root_dir}\/ckpts\/pruned_/s/^/# /' \
  -e 's/--num_workers 12/--num_workers 2/g' \
  -e "s|\${SASPG_ROOT}/DPHuBERT_pretrain_unstr/pretrained|\${DPHuBERT_PRETRAINED_ROOT}|g" \
  "${_src}" > "${_dst}"

python3 - "${_dst}" <<'PY'
from pathlib import Path
import os
import re
import sys

p = Path(sys.argv[1])
lines = p.read_text().splitlines(keepends=True)


def _ensure_comment(line: str) -> str:
    if not line.strip() or line.lstrip().startswith("#"):
        return line
    m = re.match(r"^(\s*)", line)
    indent = m.group(1) if m else ""
    body = line[len(indent) :]
    nl = "\n" if line.endswith("\n") else ""
    return f"{indent}# {body}{nl}"


def _strip_comment(line: str) -> str:
    if not line.strip():
        return line
    m = re.match(r"^(\s*)#\s?(.*)$", line.rstrip("\n"))
    if not m:
        return line
    nl = "\n" if line.endswith("\n") else ""
    return f"{m.group(1)}{m.group(2)}{nl}"


def _is_distill_line(line: str) -> bool:
    return bool(re.search(r"distill_1_01\.py", line))


def _is_active_cmd_start(line: str, verbs: tuple) -> bool:
    s = line.lstrip()
    if s.startswith("#"):
        s = s[1:].lstrip()
    return any(re.match(rf"python\s+{v}", s) for v in verbs)


def _continuation_line(line: str) -> bool:
    s = line.lstrip().lstrip("#").lstrip()
    return s.startswith("--") or s.endswith("\\") or "| tee" in s


def patch_pruned_ckpt(text: str) -> str:
    for name in ("hubert-base-ls960", "hubert-large-ll60k"):
        pruned = "pruned_hubert_base.pth" if "base" in name else "pruned_hubert_large.pth"
        old = f"pruned_ckpt=${{root_dir}}/ckpts/{pruned}"
        new = f'pruned_ckpt="${{teacher_ckpt%{name}.hf.pth}}/{pruned}"'
        if old in text:
            text = text.replace(old, new, 1)
    return re.sub(
        r"pruned_ckpt=\$\{root_dir\}/ckpts/(pruned_hubert_[a-z]+\.pth)",
        r'pruned_ckpt="$(dirname "${student_ckpt}")/\1"',
        text,
        count=1,
    )


out: list[str] = []
i = 0
while i < len(lines):
    line = lines[i]

    # Keep SASPG distill block fully commented (MAG skips distill).
    if _is_distill_line(line):
        while i < len(lines):
            ln = lines[i]
            out.append(_ensure_comment(ln) if not ln.lstrip().startswith("#") else ln)
            if "distill.log" in ln and "exit" in ln:
                i += 1
                break
            i += 1
        continue

    # Uncomment prune_*_mag / final_distill*mag and their --arg continuations.
    stripped = line.lstrip()
    if stripped.startswith("#") and _is_active_cmd_start(
        line, ("prune_1_01", "prune_1_01_large", "final_distill", "final_distill_1_01")
    ):
        out.append(_strip_comment(line))
        i += 1
        while i < len(lines) and _continuation_line(lines[i]):
            cont = lines[i]
            if cont.lstrip().startswith("#"):
                out.append(_strip_comment(cont))
            else:
                out.append(cont)
            if "|| exit" in cont or (
                not cont.rstrip().endswith("\\") and cont.strip() and not _continuation_line(cont)
            ):
                if "|| exit" in cont:
                    i += 1
                    break
            i += 1
            if i < len(lines) and lines[i - 1].strip() and "|| exit" in lines[i - 1]:
                break
        continue

    out.append(line)
    i += 1

text = patch_pruned_ckpt("".join(out))
exp_root = os.environ.get("SUPERB_MAG_EXP_ROOT", "").rstrip("/")
if exp_root:

    def _prefix_root_dir(m):
        path = m.group(1).strip()
        if path.startswith("/") or path.startswith("${"):
            return m.group(0)
        return f"root_dir={exp_root}/{path}"

    text = re.sub(r"^root_dir=([^\n]+)$", _prefix_root_dir, text, flags=re.M)

lines = text.splitlines(keepends=True)
fixed: list[str] = []
for line in lines:
    s = line.strip()
    # Upstream MAG launchers may end with a stray path line (no leading #).
    if s and not s.startswith("#") and re.match(r"^exp_minmax_", s):
        line = _ensure_comment(line)
    if s and not s.startswith("#") and s.endswith(".pth") and "python" not in s and "=" not in s:
        line = _ensure_comment(line)
    fixed.append(line)
p.write_text("".join(fixed))
PY
chmod +x "${_dst}"

_log="${SUPERB_MAG_WORK_ROOT}/logs/superb_mag/${_exp_id}.log"
mkdir -p "$(dirname "${_log}")"
echo "[SUPERB MAG smoke] exp_id=${_exp_id} work=${SUPERB_MAG_WORK_ROOT} launcher=${_dst} log=${_log}"

cd "${_upstream_dir}"
# shellcheck disable=SC1090
bash -euo pipefail "${_dst}" 2>&1 | tee "${_log}"

echo "[SUPERB MAG smoke OK] ${_exp_id}"
