#!/usr/bin/env bash
# Fixed single-run UR-FUNNY v2 classification — Optuna trial 162 (paper 75.15% test Acc).
set -euo pipefail
MY_CREATION="$(cd "$(dirname "$0")/.." && pwd)"
cd "$MY_CREATION"

PYTHON="${PYTHON:-/root/autodl-tmp/anaconda3/envs/ITHP5090/bin/python}"
if ! "$PYTHON" -c "import torch" 2>/dev/null; then
  PYTHON="${PYTHON:-python3}"
fi

OUT="${MY_CREATION}/fixed_experiment/runs/ur_funny_trial162"
mkdir -p "${OUT}/checkpoints"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "train log: ${OUT}/train.log"
echo "checkpoints: ${OUT}/checkpoints"
"$PYTHON" -u fixed_experiment/train_fixed_ur_funny_trial162.py \
  --checkpoint-dir "${OUT}/checkpoints" \
  2>&1 | tee "${OUT}/train.log"
