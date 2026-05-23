#!/usr/bin/env bash
# Fixed single-run SIMSv2 training — Optuna infogate_simsv2_tier3_fresh_mmsa trial 53 (paper PRISM row).
set -euo pipefail
MY_CREATION="$(cd "$(dirname "$0")/.." && pwd)"
cd "$MY_CREATION"

PYTHON="${PYTHON:-/root/autodl-tmp/anaconda3/envs/ITHP5090/bin/python}"
if ! "$PYTHON" -c "import torch" 2>/dev/null; then
  PYTHON="${PYTHON:-python3}"
fi

OUT="${MY_CREATION}/fixed_experiment/runs/simsv2_tier3_fresh_trial53"
mkdir -p "${OUT}/checkpoints"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "Paper targets: Acc5=55.6 Acc3=73.4 Acc2=80.1 F1=80.1 MAE=0.291 Corr=0.705"
echo "Optuna source: simsv2_tier3_fresh trial 53"
echo "train log: ${OUT}/train.log"
echo "checkpoints: ${OUT}/checkpoints"

"$PYTHON" -u fixed_experiment/train_fixed_simsv2_tier3_fresh_trial53.py \
  --checkpoint-dir "${OUT}/checkpoints" \
  2>&1 | tee "${OUT}/train.log"
