#!/usr/bin/env bash
# MOSI trial234 baseline (--ablation none) on ONE GPU only, under nohup.
# Use this to match fixed_experiment/runs/mosi_trial234/train.log (no second training on the same card).
set -euo pipefail
MY_CREATION="$(cd "$(dirname "$0")/.." && pwd)"
cd "$MY_CREATION"

PYTHON="${PYTHON:-/root/autodl-tmp/anaconda3/envs/ITHP5090/bin/python}"
if ! "$PYTHON" -c "import torch" 2>/dev/null; then
  PYTHON="${PYTHON:-python3}"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

OUT="${OUT:-${MY_CREATION}/ablation_study/runs/mosi_trial234_none_solo}"
mkdir -p "${OUT}/checkpoints"
LOG="${OUT}/train.log"
: >"${LOG}"

echo "======== $(date -Is) none solo (frozen-align) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} ========"
echo "OUT=${OUT}"
echo "LOG=${LOG}"

nohup "$PYTHON" -u ablation_study/train_fixed_mosi_trial234.py \
  --ablation none \
  --checkpoint-dir "${OUT}/checkpoints" \
  >>"${LOG}" 2>&1 &
pid=$!
echo "Started PID=${pid}"
echo "${pid}" >"${OUT}/nohup.pid"
echo "$(date -Is) PID=${pid}" >>"${OUT}/nohup.meta"
