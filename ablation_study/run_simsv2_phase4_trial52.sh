#!/usr/bin/env bash
# Fixed single-run SIMSv2 training — Optuna 4090D_restart phase4 trial 52 (ablation_study, --ablation none).
set -euo pipefail
MY_CREATION="$(cd "$(dirname "$0")/.." && pwd)"
cd "$MY_CREATION"

PYTHON="${PYTHON:-/root/autodl-tmp/anaconda3/envs/ITHP5090/bin/python}"
if ! "$PYTHON" -c "import torch" 2>/dev/null; then
  PYTHON="${PYTHON:-python3}"
fi

OUT="${MY_CREATION}/ablation_study/runs/simsv2_phase4_trial52"
mkdir -p "${OUT}/checkpoints"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "train log: ${OUT}/train.log"
echo "checkpoints: ${OUT}/checkpoints"
"$PYTHON" -u ablation_study/train_fixed_simsv2_phase4_trial52.py \
  --ablation none \
  --checkpoint-dir "${OUT}/checkpoints" \
  2>&1 | tee "${OUT}/train.log"
