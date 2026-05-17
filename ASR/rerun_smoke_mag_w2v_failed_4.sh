#!/usr/bin/env bash
# Re-run Wav2Vec2 magnitude slots 09,10 (unstr) and 13,14 (str).
# Uses smoke_experiment_lib.sh: unstr = mask×weight only (no structural pruned/); str = prune_ASR export.
#
# Same environment variables as smoke_20exp_500steps.sh (MAX_STEPS, MAG_PRUNE_STEPS,
# SMOKE_GPU_INDEX, PYTHON, WAIT_GPU_EACH_TASK, etc.). Smoke defaults: DATALOADER_NUM_WORKERS=0,
# PYTHONUNBUFFERED=1 (set in smoke_20exp_500steps.sh).
#
# CLEAN_RETRY (default 1): remove each slot's stage2/ and every out/checkpoint-*/pruned/
#   before re-running. Use CLEAN_RETRY=all to also rm -rf each slot's out/ (full cold restart).
# CLEAN_RETRY=0: do not delete anything (resume / debug only).
#
# If you see "Disk quota exceeded" or incomplete train_filtered_* cache errors, free quota
# then remove the partial cache (auto-removed on next run if code is updated):
#   rm -rf /project_bdda8/bdda/hnxu/prune/SSLprune/TASLP/cache_dir/train_filtered_960_19p1
#
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SMOKE_CMD=mag_w2v_retry_4
exec bash "$ROOT/smoke_20exp_500steps.sh" "$@"
