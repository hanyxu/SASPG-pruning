#!/usr/bin/env bash
# Run one SUPERB 100h SASPG smoke job: generate short-step launcher from CSV entry, then execute.
# Usage: superb_smoke_from_launcher.sh <exp_id>
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
ASR_ROOT="${SASPG_REPO}/ASR"
export SASPG_ROOT="${SASPG_ROOT:-$(cd "${SASPG_REPO}/../.." && pwd)}"

_saspg_pretrained="${SUPERB_SASPG_WORK_ROOT}/superb/pretrained"
mkdir -p "${_saspg_pretrained}" "${SUPERB_SASPG_WORK_ROOT}/logs/superb"
_src_pretrained="${WORK_ROOT}/superb/pretrained"
for _f in hubert-base-ls960.hf.pth hubert-large-ll60k.hf.pth; do
  if [[ -e "${_src_pretrained}/${_f}" && ! -e "${_saspg_pretrained}/${_f}" ]]; then
    ln -sf "${_src_pretrained}/${_f}" "${_saspg_pretrained}/${_f}"
  fi
done

export DPHuBERT_PRETRAINED_DIR="${_saspg_pretrained}"
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
            fam = row["source_repo"].strip()
            script = row["entry_script"].strip()
            print(fam)
            print(script)
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
export SUPERB_SASPG_EXP_ROOT="${SUPERB_SASPG_WORK_ROOT}/exp/${_family}"
mkdir -p "${SUPERB_SASPG_EXP_ROOT}"
_dst="${_upstream_dir}/.smoke_${_exp_id}.sh"

# Short steps + uncomment full distill/prune/final_distill blocks; skip save_final_ckpt.
sed \
  -e "s/^max=[0-9]*/max=${SUPERB_SMOKE_MAX}/" \
  -e "s/^warmup=[0-9]*/warmup=${SUPERB_SMOKE_WARMUP}/" \
  -e "s/^final_max=[0-9]*/final_max=${SUPERB_SMOKE_FINAL_MAX}/" \
  -e "s/^final_warmup=[0-9]*/final_warmup=${SUPERB_SMOKE_WARMUP}/" \
  -e "s/^sparsity_warmup=[0-9]*/sparsity_warmup=${SUPERB_SMOKE_SP_WARMUP}/" \
  -e 's/^#     /    /' \
  -e 's/^# python distill/python distill/' \
  -e 's/^# python prune/python prune/' \
  -e 's/^# python final_distill/python final_distill/' \
  -e '/^# save final model/,/exit 1;/s/^/# /' \
  -e '/^    --config_path \${pruned_ckpt}/,/^.*exit 1;/s/^/# /' \
  -e 's/--num_workers 12/--num_workers 2/g' \
  -e 's/tsv_dir=${DPHuBERT_TSV_DIR:-data\/librispeech}_bdda9/tsv_dir=${DPHuBERT_TSV_DIR:-data\/librispeech}/' \
  -e "s|\${SASPG_ROOT}/DPHuBERT_pretrain_unstr/pretrained|\${DPHuBERT_PRETRAINED_ROOT}|g" \
  "${_src}" > "${_dst}"
python3 - "${_dst}" <<'PY'
from pathlib import Path
import os
import re
import sys

p = Path(sys.argv[1])
t = p.read_text()

exp_root = os.environ.get("SUPERB_SASPG_EXP_ROOT", "").rstrip("/")
if exp_root:

    def _prefix_root_dir(m):
        path = m.group(1).strip()
        if path.startswith("/") or path.startswith("${"):
            return m.group(0)
        return f"root_dir={exp_root}/{path}"

    t = re.sub(r"^root_dir=([^\n]+)$", _prefix_root_dir, t, flags=re.M)

old = "--distilled_ckpt ${root_dir}/ckpts/*.ckpt"
new = '--distilled_ckpt "$(ls -t "${root_dir}"/ckpts/*.ckpt 2>/dev/null | head -1)"'
if old not in t:
    raise SystemExit("patch failed: prune distilled_ckpt line not found")
t = t.replace(old, new, 1)

# Lightning may exit non-zero after max_steps under pipefail; smoke continues if ckpt exists.
_DISTILL_TEE = (
    "2>&1 | tee ${root_dir}/distill.log;\n"
    "_distill_rc=${PIPESTATUS[0]};\n"
    'if [[ ${_distill_rc} -ne 0 ]] && ! compgen -G "${root_dir}/ckpts/*.ckpt" > /dev/null; then\n'
    '  echo "[SUPERB smoke] distill failed: rc=${_distill_rc}, no ckpt in ${root_dir}/ckpts" >&2;\n'
    "  exit 1;\n"
    "fi;\n"
    'if [[ ${_distill_rc} -ne 0 ]]; then\n'
    '  echo "[SUPERB smoke] distill rc=${_distill_rc} but ckpt found; continuing" >&2;\n'
    "fi;\n"
)
if "2>&1 | tee ${root_dir}/distill.log || exit 1;" not in t:
    raise SystemExit("patch failed: distill tee line not found")
t = t.replace("2>&1 | tee ${root_dir}/distill.log || exit 1;", _DISTILL_TEE, 1)

t = re.sub(
    r"2>&1 \| tee (\$\{final_exp_dir\}/[^\s]+) \|\| exit 1;",
    r"2>&1 | tee \1;\n"
    r"_final_rc=${PIPESTATUS[0]};\n"
    r"if [[ ${_final_rc} -ne 0 ]] && ! grep -q 'max_steps=' \1 2>/dev/null; then exit 1; fi;",
    t,
)

t = re.sub(r"^# mkdir -p \$\{root_dir\}\s*$", "mkdir -p ${root_dir}", t, flags=re.M)
t = re.sub(r"^# mkdir -p \$\{final_exp_dir\}\s*$", "mkdir -p ${final_exp_dir}", t, flags=re.M)


def _ensure_comment(line: str) -> str:
    if not line.strip() or line.lstrip().startswith("#"):
        return line
    m = re.match(r"^(\s*)", line)
    indent = m.group(1) if m else ""
    body = line[len(indent) :]
    nl = "\n" if line.endswith("\n") else ""
    return f"{indent}# {body}{nl}"


fixed = []
for line in t.splitlines(keepends=True):
    s = line.strip()
    if s and not s.startswith("#") and re.match(r"^exp_minmax_", s):
        line = _ensure_comment(line)
    if s and not s.startswith("#") and s.endswith(".pth") and "python" not in s and "=" not in s:
        line = _ensure_comment(line)
    fixed.append(line)
p.write_text("".join(fixed))
PY
chmod +x "${_dst}"

_log="${SUPERB_SASPG_WORK_ROOT}/logs/superb/${_exp_id}.log"
mkdir -p "$(dirname "${_log}")"
echo "[SUPERB smoke] exp_id=${_exp_id} work=${SUPERB_SASPG_WORK_ROOT} launcher=${_dst} log=${_log}"

cd "${_upstream_dir}"
# shellcheck disable=SC1090
bash -euo pipefail "${_dst}" 2>&1 | tee "${_log}"

echo "[SUPERB smoke OK] ${_exp_id}"
