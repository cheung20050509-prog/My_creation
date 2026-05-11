#!/usr/bin/env bash
# Run MOSEI phase1 trial70 and SIMSv2 phase4 trial52 in parallel (one GPU each).
# Requires two GPUs (default: MOSEI=0, SIMSv2=1). Override with MOSEI_GPU / SIMSV2_GPU.
set -euo pipefail
MY_CREATION="$(cd "$(dirname "$0")/.." && pwd)"
cd "$MY_CREATION"

PYTHON="${PYTHON:-/root/autodl-tmp/anaconda3/envs/ITHP5090/bin/python}"
MOSEI_GPU="${MOSEI_GPU:-0}"
SIMSV2_GPU="${SIMSV2_GPU:-1}"

mkdir -p fixed_experiment/runs

echo "MOSEI  -> CUDA_VISIBLE_DEVICES=${MOSEI_GPU}"
nohup env CUDA_VISIBLE_DEVICES="${MOSEI_GPU}" PYTHON="${PYTHON}" bash fixed_experiment/run_mosei_phase1_trial70.sh \
  > fixed_experiment/runs/mosei_phase1_trial70_parallel_meta.log 2>&1 &
echo "  PID=$!  log: fixed_experiment/runs/mosei_phase1_trial70/train.log"

echo "SIMSv2 -> CUDA_VISIBLE_DEVICES=${SIMSV2_GPU}"
nohup env CUDA_VISIBLE_DEVICES="${SIMSV2_GPU}" PYTHON="${PYTHON}" bash fixed_experiment/run_simsv2_phase4_trial52.sh \
  > fixed_experiment/runs/simsv2_phase4_trial52_parallel_meta.log 2>&1 &
echo "  PID=$!  log: fixed_experiment/runs/simsv2_phase4_trial52/train.log"

echo "Both launched; tail -f the train.log paths above."
