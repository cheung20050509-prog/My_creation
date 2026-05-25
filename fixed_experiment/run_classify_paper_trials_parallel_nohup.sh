#!/usr/bin/env bash
# Detached parallel reproduction: UR-FUNNY trial 162 + MUStARD s2_local trial 134.
# Default: UR-FUNNY on GPU 0, MUStARD on GPU 1. On a single-GPU host, set both to 0.
set -euo pipefail
MY_CREATION="$(cd "$(dirname "$0")/.." && pwd)"
cd "$MY_CREATION"

PYTHON="${PYTHON:-/root/autodl-tmp/anaconda3/envs/ITHP5090/bin/python}"
UR_GPU="${UR_FUNNY_GPU:-0}"
MU_GPU="${MUSTARD_GPU:-1}"

UR_OUT="${MY_CREATION}/fixed_experiment/runs/ur_funny_trial162"
MU_OUT="${MY_CREATION}/fixed_experiment/runs/mustard_s2_local_trial134"
UR_META="${UR_OUT}/nohup_meta.log"
MU_META="${MU_OUT}/nohup_meta.log"
UR_PID="${UR_OUT}/train.pid"
MU_PID="${MU_OUT}/train.pid"

mkdir -p "${UR_OUT}/checkpoints" "${MU_OUT}/checkpoints"

_running() {
  local pidfile="$1"
  [[ -f "${pidfile}" ]] || return 1
  local pid
  pid="$(cat "${pidfile}")"
  kill -0 "${pid}" 2>/dev/null
}

if _running "${UR_PID}"; then
  echo "UR-FUNNY trial 162 already running (PID $(cat "${UR_PID}"))."
else
  echo "Launch UR-FUNNY trial 162 on GPU ${UR_GPU} ..."
  nohup env CUDA_VISIBLE_DEVICES="${UR_GPU}" PYTHON="${PYTHON}" \
    bash fixed_experiment/run_ur_funny_trial162.sh \
    > "${UR_META}" 2>&1 &
  echo $! > "${UR_PID}"
  echo "  wrapper PID=$(cat "${UR_PID}")"
fi

if _running "${MU_PID}"; then
  echo "MUStARD trial 134 already running (PID $(cat "${MU_PID}"))."
else
  echo "Launch MUStARD trial 134 on GPU ${MU_GPU} ..."
  nohup env CUDA_VISIBLE_DEVICES="${MU_GPU}" PYTHON="${PYTHON}" \
    bash fixed_experiment/run_mustard_s2_local_trial134.sh \
    > "${MU_META}" 2>&1 &
  echo $! > "${MU_PID}"
  echo "  wrapper PID=$(cat "${MU_PID}")"
fi

echo ""
echo "Logs:"
echo "  UR-FUNNY train: ${UR_OUT}/train.log"
echo "  UR-FUNNY meta:  ${UR_META}"
echo "  MUStARD train:  ${MU_OUT}/train.log"
echo "  MUStARD meta:   ${MU_META}"
echo "Monitor:"
echo "  tail -f ${UR_OUT}/train.log ${MU_OUT}/train.log"
