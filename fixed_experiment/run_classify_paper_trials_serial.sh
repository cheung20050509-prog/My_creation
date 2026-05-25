#!/usr/bin/env bash
# Serial reproduction: UR-FUNNY trial 162 then MUStARD s2_local trial 134.
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"

echo "[1/2] UR-FUNNY trial 162"
CUDA_VISIBLE_DEVICES="${UR_FUNNY_GPU:-0}" bash "${DIR}/run_ur_funny_trial162.sh"

echo "[2/2] MUStARD s2_local trial 134"
CUDA_VISIBLE_DEVICES="${MUSTARD_GPU:-0}" bash "${DIR}/run_mustard_s2_local_trial134.sh"
