#!/usr/bin/env bash
set -euo pipefail

# Run all ready upstream jobs from configs/experiments.csv (48 SSL compression experiments)
# on a single GPU (default: physical GPU 0), one after another.
#
# Data: upstream launchers source scripts/source_dphubert_env.sh, which sets DPHuBERT_TSV_DIR to
# ${SSLprune_root}/DPHuBERT_pretrain_unstr/data/librispeech when present.
#
# Usage:
#   bash ./run_smoke960_dual_gpu.sh
#   bash ./run_smoke960_dual_gpu.sh --pretrained-dir /path/to/pretrained
# Optional filters (passed through to the embedded runner):
#   SMOKE_HOURS=960 SMOKE_MODEL=hubert_base bash ./run_smoke960_dual_gpu.sh
# Optional: use another physical id (still one GPU, sequential):
#   SMOKE960_PHYSICAL_GPU=1 bash ./run_smoke960_dual_gpu.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${ROOT_DIR}/smoke960_logs"
mkdir -p "${LOG_DIR}"

PRETRAINED_DIR_ARG=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --pretrained-dir)
      if [[ $# -lt 2 ]]; then
        echo "ERROR: --pretrained-dir requires a path argument." >&2
        exit 1
      fi
      PRETRAINED_DIR_ARG="$2"
      shift 2
      ;;
    *)
      echo "ERROR: Unknown argument: $1" >&2
      echo "Usage: bash ./run_smoke960_dual_gpu.sh [--pretrained-dir /path/to/pretrained]" >&2
      exit 1
      ;;
  esac
done

if [[ -n "${PRETRAINED_DIR_ARG}" ]]; then
  if [[ ! -d "${PRETRAINED_DIR_ARG}" ]]; then
    echo "ERROR: pretrained dir not found: ${PRETRAINED_DIR_ARG}" >&2
    exit 1
  fi
  export DPHuBERT_PRETRAINED_DIR="${PRETRAINED_DIR_ARG}"
  echo "Using DPHuBERT_PRETRAINED_DIR=${DPHuBERT_PRETRAINED_DIR}"
fi

cd "${ROOT_DIR}"

export SMOKE_HOURS="${SMOKE_HOURS:-}"
export SMOKE_MODEL="${SMOKE_MODEL:-}"
export SMOKE_LIMIT="${SMOKE_LIMIT:-}"

python3 - <<'PY'
import csv
import json
import os
import subprocess
import time
from pathlib import Path

root = Path.cwd()
log_dir = root / "smoke960_logs"
log_dir.mkdir(exist_ok=True)
pretrained_dir = os.environ.get("DPHuBERT_PRETRAINED_DIR", "").strip()

if pretrained_dir:
    pretrained_path = Path(pretrained_dir)
    required_hf = [
        "hubert-base-ls960.hf.pth",
        "hubert-large-ll60k.hf.pth",
        "wav2vec2-base.hf.pth",
        "wavlm-base-plus.hf.pth",
    ]
    missing = [name for name in required_hf if not (pretrained_path / name).is_file()]
    if missing:
        raise SystemExit(
            "Missing required hf checkpoints in DPHuBERT_PRETRAINED_DIR:\n"
            + "\n".join(f"- {name}" for name in missing)
            + f"\nDirectory: {pretrained_path}"
        )
    print(f"[CHECK] DPHuBERT_PRETRAINED_DIR={pretrained_path}")
    print(f"[CHECK] Found {len(required_hf)} required hf checkpoints.")

csv_path = root / "configs" / "experiments.csv"
if not csv_path.is_file():
    raise SystemExit(f"Missing {csv_path}")

with csv_path.open("r", encoding="utf-8") as f:
    rows = [r for r in csv.DictReader(f) if (r.get("status") or "").strip() == "ready"]

hours_filter = os.environ.get("SMOKE_HOURS", "").strip()
if hours_filter:
    rows = [r for r in rows if (r.get("data_hours") or "").strip() == hours_filter]

model_filter = os.environ.get("SMOKE_MODEL", "").strip()
if model_filter:
    rows = [r for r in rows if (r.get("model") or "").strip() == model_filter]

limit_s = os.environ.get("SMOKE_LIMIT", "").strip()
if limit_s.isdigit() and int(limit_s) > 0:
    rows = rows[: int(limit_s)]

if not hours_filter and not model_filter and not limit_s and len(rows) != 48:
    print(
        f"[WARN] expected 48 ready experiments, got {len(rows)} "
        "(set SMOKE_HOURS / SMOKE_MODEL / SMOKE_LIMIT to filter).",
        flush=True,
    )

all_tasks = []
for r in rows:
    eid = (r.get("exp_id") or "").strip()
    if not eid:
        continue
    all_tasks.append(
        {
            "tag": eid,
            "cmd": [
                "python3",
                "-m",
                "core",
                "pipeline",
                "--run-upstream-first",
                "--exp-id",
                eid,
            ],
        }
    )

if not all_tasks:
    raise SystemExit("No ready experiments to run after filters.")

# Single physical GPU index on the node (always exposed as cuda:0 inside the job).
PHYSICAL_GPU = int(os.environ.get("SMOKE960_PHYSICAL_GPU", "0"))


def run_queue_sequential(gpu_id, queue):
    results = []
    for i, task in enumerate(queue, 1):
        tag = task["tag"]
        cmd = task["cmd"]
        logf = log_dir / f"gpu{gpu_id}_{i:04d}_{tag}.log"

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env["LOCAL_RANK"] = "0"
        env["RANK"] = "0"
        env["WORLD_SIZE"] = "1"
        env["NODE_RANK"] = "0"
        for k in [
            "SLURM_LOCALID",
            "SLURM_PROCID",
            "SLURM_NODEID",
            "SLURM_NTASKS",
            "SLURM_NTASKS_PER_NODE",
            "SLURM_STEP_GPUS",
            "SLURM_JOB_GPUS",
            "GPU_DEVICE_ORDINAL",
            "OMPI_COMM_WORLD_LOCAL_RANK",
            "OMPI_COMM_WORLD_RANK",
            "OMPI_COMM_WORLD_SIZE",
            "PMI_RANK",
            "PMI_SIZE",
        ]:
            env.pop(k, None)

        t0 = time.time()
        print(f"[GPU{gpu_id}] [{i}/{len(queue)}] START {tag}", flush=True)
        with open(logf, "w", encoding="utf-8") as f:
            p = subprocess.run(cmd, cwd=root, env=env, stdout=f, stderr=subprocess.STDOUT)
        dt = int(time.time() - t0)
        print(f"[GPU{gpu_id}] [{i}/{len(queue)}] END   {tag} rc={p.returncode} t={dt}s", flush=True)

        results.append(
            {
                "gpu": gpu_id,
                "index_in_gpu_queue": i,
                "tag": tag,
                "rc": p.returncode,
                "elapsed_sec": dt,
                "log": str(logf),
                "cmd": cmd,
            }
        )
    return results


print(
    f"[RUN] Single-GPU mode: CUDA_VISIBLE_DEVICES={PHYSICAL_GPU} for all {len(all_tasks)} jobs (sequential).",
    flush=True,
)
all_results = run_queue_sequential(PHYSICAL_GPU, all_tasks)
summary_path = log_dir / "summary.json"
summary_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")

ok = sum(1 for r in all_results if r["rc"] == 0)
fail = len(all_results) - ok
print(f"[DONE] total={len(all_results)} ok={ok} fail={fail}")
print(f"[DONE] summary={summary_path}")
PY
