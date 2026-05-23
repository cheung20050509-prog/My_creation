#!/usr/bin/env bash
# Detached SIMSv2 tier3_fresh trial 53 reproduction (survives SSH logout).
# Train output: fixed_experiment/runs/simsv2_tier3_fresh_trial53/train.log
# This wrapper log:  .../nohup_meta.log
set -euo pipefail
MY_CREATION="$(cd "$(dirname "$0")/.." && pwd)"
cd "$MY_CREATION"

PYTHON="${PYTHON:-/root/autodl-tmp/anaconda3/envs/ITHP5090/bin/python}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
OUT="${MY_CREATION}/fixed_experiment/runs/simsv2_tier3_fresh_trial53"
META="${OUT}/nohup_meta.log"
PIDFILE="${OUT}/train.pid"

mkdir -p "${OUT}/checkpoints"

if [[ -f "${PIDFILE}" ]]; then
  old_pid="$(cat "${PIDFILE}")"
  if kill -0 "${old_pid}" 2>/dev/null; then
    echo "Already running (PID ${old_pid})."
    echo "  train log: ${OUT}/train.log"
    echo "  meta log:  ${META}"
    exit 0
  fi
fi

echo "Launching nohup on GPU ${GPU} ..."
nohup env CUDA_VISIBLE_DEVICES="${GPU}" PYTHON="${PYTHON}" \
  bash fixed_experiment/run_simsv2_tier3_fresh_trial53.sh \
  > "${META}" 2>&1 &
echo $! > "${PIDFILE}"
echo "PID=$(cat "${PIDFILE}")"
echo "  train log: ${OUT}/train.log"
echo "  meta log:  ${META}"
echo "Monitor: tail -f ${OUT}/train.log"
