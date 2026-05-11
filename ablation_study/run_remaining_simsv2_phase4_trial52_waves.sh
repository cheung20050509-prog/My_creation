#!/usr/bin/env bash
# Finish remaining SIMSv2 phase4 trial 52 PRISM modes after no_infogate / no_mselector.
# Two physical GPUs (indices 0,1): treat "GPU1" as device 1 and "GPU2" as device 0.
# Wave 1 (parallel): no_ib @ GPU1, no_conf_gating @ GPU0
# Wave 2 (parallel): none @ GPU1, no_adaptive_gate @ GPU0
set -euo pipefail
MY_CREATION="$(cd "$(dirname "$0")/.." && pwd)"
cd "$MY_CREATION"

PYTHON="${PYTHON:-/root/autodl-tmp/anaconda3/envs/ITHP5090/bin/python}"
if ! "$PYTHON" -c "import torch" 2>/dev/null; then
  PYTHON="${PYTHON:-python3}"
fi
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

run_job() {
  local gpu="$1"
  local mode="$2"
  local OUT
  if [[ "$mode" == "none" ]]; then
    OUT="${MY_CREATION}/ablation_study/runs/simsv2_phase4_trial52"
  else
    OUT="${MY_CREATION}/ablation_study/runs/simsv2_phase4_trial52_${mode}"
  fi
  mkdir -p "${OUT}/checkpoints"
  echo "======== $(date -Is) START ablation=${mode} CUDA_VISIBLE_DEVICES=${gpu} OUT=${OUT} ========"
  CUDA_VISIBLE_DEVICES="${gpu}" "$PYTHON" -u ablation_study/train_fixed_simsv2_phase4_trial52.py \
    --ablation "$mode" \
    --checkpoint-dir "${OUT}/checkpoints" \
    >>"${OUT}/train.log" 2>&1
  echo "======== $(date -Is) DONE ablation=${mode} CUDA_VISIBLE_DEVICES=${gpu} ========"
}

echo "======== $(date -Is) SIMSv2 remaining waves: GPU1=dev1 GPU2=dev0 ========"

run_job 1 no_ib &
p1=$!
run_job 0 no_conf_gating &
p2=$!
wait "${p1}" "${p2}"

run_job 1 none &
p3=$!
run_job 0 no_adaptive_gate &
p4=$!
wait "${p3}" "${p4}"

echo "======== $(date -Is) all remaining SIMSv2 phase4 trial 52 modes finished ========"
