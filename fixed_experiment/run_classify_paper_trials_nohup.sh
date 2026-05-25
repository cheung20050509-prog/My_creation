#!/usr/bin/env bash
# Detached UR-FUNNY trial 162 -> MUStARD trial 134 (serial, survives SSH logout).
set -euo pipefail
MY_CREATION="$(cd "$(dirname "$0")/.." && pwd)"
cd "$MY_CREATION"

PYTHON="${PYTHON:-/root/autodl-tmp/anaconda3/envs/ITHP5090/bin/python}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
META="${MY_CREATION}/fixed_experiment/runs/classify_paper_serial_nohup.log"
PIDFILE="${MY_CREATION}/fixed_experiment/runs/classify_paper_serial.pid"

mkdir -p "${MY_CREATION}/fixed_experiment/runs"

if [[ -f "${PIDFILE}" ]]; then
  old_pid="$(cat "${PIDFILE}")"
  if kill -0 "${old_pid}" 2>/dev/null; then
    echo "Already running (PID ${old_pid})."
    echo "  meta log: ${META}"
    exit 0
  fi
fi

echo "Launching serial classify repro on GPU ${GPU} ..."
nohup env CUDA_VISIBLE_DEVICES="${GPU}" UR_FUNNY_GPU="${GPU}" MUSTARD_GPU="${GPU}" PYTHON="${PYTHON}" \
  bash fixed_experiment/run_classify_paper_trials_serial.sh \
  > "${META}" 2>&1 &
echo $! > "${PIDFILE}"
echo "PID=$(cat "${PIDFILE}")"
echo "  meta log: ${META}"
echo "  UR log:   fixed_experiment/runs/ur_funny_trial162/train.log"
echo "  MU log:   fixed_experiment/runs/mustard_s2_local_trial134/train.log"
