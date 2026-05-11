#!/usr/bin/env bash
# Fixed single-run MOSI training — Optuna infogate_mosi_phase4_mosi_4090d trial 220 (ablation_study).
set -euo pipefail
MY_CREATION="$(cd "$(dirname "$0")/.." && pwd)"
cd "$MY_CREATION"

PYTHON="${PYTHON:-/root/autodl-tmp/anaconda3/envs/ITHP5090/bin/python}"
if ! "$PYTHON" -c "import torch" 2>/dev/null; then
  PYTHON="${PYTHON:-python3}"
fi

OUT="${MY_CREATION}/ablation_study/runs/mosi_trial220"
mkdir -p "${OUT}/checkpoints"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "train log: ${OUT}/train.log"
echo "checkpoints: ${OUT}/checkpoints"
"$PYTHON" -u ablation_study/train_fixed_mosi_trial220.py \
  --ablation none \
  --checkpoint-dir "${OUT}/checkpoints" \
  2>&1 | tee "${OUT}/train.log"
