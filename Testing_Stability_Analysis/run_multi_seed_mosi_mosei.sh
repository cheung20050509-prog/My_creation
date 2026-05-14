#!/usr/bin/env bash
# Multi-seed training stability: same argv as fixed_experiment launchers, only --seed
# and per-seed --checkpoint-dir change.
#
# Scheduling: by default all jobs share **GPU 0** (CUDA_VISIBLE_DEVICES=0) with at most
# **MAX_PARALLEL=2** trainings running at once (parallel within the cap, queued serially
# beyond that — 并行+串行).
#
# Usage (from anywhere):
#   bash Testing_Stability_Analysis/run_multi_seed_mosi_mosei.sh
#
# Env:
#   SEEDS="42 128 256 1024 2024"   default matches multi_seed_verify.sh
#   RUN_MOSI=1 RUN_MOSEI=1         set to 0 to skip one dataset
#   OUT=path                       run root (default: Testing_Stability_Analysis/runs/multi_seed_<UTC>)
#   PYTHON=...                     interpreter (default: ITHP5090 conda python)
#   GPU=0                          default single visible device index (CUDA_VISIBLE_DEVICES)
#   MAX_PARALLEL=2                 max concurrent train processes on that GPU
#
# After runs, summarize dev-selected test MAE from each train.log:
#   python Testing_Stability_Analysis/collect_best_dev_selected_mae.py "$OUT"
#
# Or manually:
#   rg -n "Best Results" "$OUT"/**/train.log
#   rg -n "  MAE:" "$OUT"/mosi_seed128/train.log | head
set -euo pipefail
set -m

MY_CREATION="$(cd "$(dirname "$0")/.." && pwd)"
cd "$MY_CREATION"

PYTHON="${PYTHON:-/root/autodl-tmp/anaconda3/envs/ITHP5090/bin/python}"
if ! "$PYTHON" -c "import torch" 2>/dev/null; then
  PYTHON="${PYTHON:-python3}"
fi

SEEDS="${SEEDS:-42 128 256 1024 2024}"
RUN_MOSI="${RUN_MOSI:-1}"
RUN_MOSEI="${RUN_MOSEI:-1}"
GPU="${GPU:-0}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"
if [[ "$MAX_PARALLEL" -lt 1 ]]; then
  echo "ERROR: MAX_PARALLEL must be >= 1, got: $MAX_PARALLEL" >&2
  exit 1
fi
STAMP="$(date -u +%Y%m%d_%H%M%S)"
OUT="${OUT:-$MY_CREATION/Testing_Stability_Analysis/runs/multi_seed_${STAMP}}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "$OUT"
echo "[multi_seed] OUT=$OUT"
echo "[multi_seed] SEEDS=$SEEDS"
echo "[multi_seed] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES  MAX_PARALLEL=$MAX_PARALLEL"

_throttle() {
  local max="${1:?}"
  while true; do
    local n
    n="$(jobs -rp 2>/dev/null | wc -l | tr -d ' ')"
    [[ "${n:-0}" -lt "$max" ]] && return 0
    sleep 2
  done
}

_launch_mosi() {
  local seed="$1"
  local d="$OUT/mosi_seed${seed}"
  mkdir -p "$d/checkpoints"
  echo "======== MOSI seed=$seed -> $d/train.log (bg) ========"
  (
    set -euo pipefail
    "$PYTHON" -u fixed_experiment/train_fixed_mosi_trial234.py \
      --seed "$seed" \
      --checkpoint-dir "$d/checkpoints" \
      2>&1 | tee "$d/train.log"
  ) &
}

_launch_mosei() {
  local seed="$1"
  local d="$OUT/mosei_seed${seed}"
  mkdir -p "$d/checkpoints"
  echo "======== MOSEI seed=$seed -> $d/train.log (bg) ========"
  (
    set -euo pipefail
    "$PYTHON" -u fixed_experiment/train_fixed_mosei_phase1_trial70.py \
      --seed "$seed" \
      --checkpoint-dir "$d/checkpoints" \
      2>&1 | tee "$d/train.log"
  ) &
}

for seed in $SEEDS; do
  if [[ "$RUN_MOSI" == "1" ]]; then
    _throttle "$MAX_PARALLEL"
    _launch_mosi "$seed"
  fi
done

for seed in $SEEDS; do
  if [[ "$RUN_MOSEI" == "1" ]]; then
    _throttle "$MAX_PARALLEL"
    _launch_mosei "$seed"
  fi
done

echo "[multi_seed] waiting for all jobs (still up to $MAX_PARALLEL were running on GPU)..."
wait

echo "[multi_seed] done. Collect summaries with:"
echo "  python Testing_Stability_Analysis/collect_best_dev_selected_mae.py \"$OUT\""
