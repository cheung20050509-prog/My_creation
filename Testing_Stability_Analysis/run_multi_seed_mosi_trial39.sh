#!/usr/bin/env bash
# Multi-seed MOSI stability — frozen more/mosi Optuna trial 39 (paper PRISM row, MAE ~0.606).
# Hyperparameters match ablation_study/runs/mosi_more_trial39/ gold repro.
#
# Usage (from anywhere):
#   bash Testing_Stability_Analysis/run_multi_seed_mosi_trial39.sh
#
# Env:
#   SEEDS="42 128 256 1024 2024"
#   OUT=.../runs/mosi_t39_<UTC>     default under Testing_Stability_Analysis/runs/
#   PYTHON=...                      default ITHP5090
#   GPU=0  MAX_PARALLEL=2
#
# After completion:
#   python Testing_Stability_Analysis/collect_stability_metrics.py "$OUT"
set -euo pipefail
set -m

MY_CREATION="$(cd "$(dirname "$0")/.." && pwd)"
cd "$MY_CREATION"

PYTHON="${PYTHON:-/root/autodl-tmp/anaconda3/envs/ITHP5090/bin/python}"
if ! "$PYTHON" -c "import torch" 2>/dev/null; then
  PYTHON="${PYTHON:-python3}"
fi

SEEDS="${SEEDS:-42 128 256 1024 2024}"
GPU="${GPU:-0}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"
STAMP="$(date -u +%Y%m%d_%H%M%S)"
OUT="${OUT:-$MY_CREATION/Testing_Stability_Analysis/runs/mosi_t39_${STAMP}}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "$OUT"
echo "[mosi_t39] OUT=$OUT"
echo "[mosi_t39] SEEDS=$SEEDS"
echo "[mosi_t39] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES  MAX_PARALLEL=$MAX_PARALLEL"

_throttle() {
  local max="${1:?}"
  while true; do
    local n
    n="$(jobs -rp 2>/dev/null | wc -l | tr -d ' ')"
    [[ "${n:-0}" -lt "$max" ]] && return 0
    sleep 2
  done
}

_launch() {
  local seed="$1"
  local d="$OUT/mosi_seed${seed}"
  mkdir -p "$d/checkpoints"
  echo "======== MOSI trial39 seed=$seed -> $d/train.log (bg) ========"
  (
    set -euo pipefail
    "$PYTHON" -u ablation_study/train_fixed_mosi_more_trial39.py \
      --ablation none \
      --seed "$seed" \
      --checkpoint-dir "$d/checkpoints" \
      2>&1 | tee "$d/train.log"
  ) &
}

for seed in $SEEDS; do
  _throttle "$MAX_PARALLEL"
  _launch "$seed"
done

echo "[mosi_t39] waiting for all jobs..."
wait

echo "[mosi_t39] done. Summarize:"
echo "  $PYTHON Testing_Stability_Analysis/collect_stability_metrics.py \"$OUT\""
