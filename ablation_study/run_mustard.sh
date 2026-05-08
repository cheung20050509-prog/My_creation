#!/bin/bash
# MUStARD — 6 architectural ablation experiments
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$DIR/logs"
mkdir -p "$LOG_DIR"
cd "$DIR"

ABLATIONS=("none" "no_infogate" "no_mselector" "no_ib" "no_conf_gating" "no_adaptive_gate")

for abl in "${ABLATIONS[@]}"; do
    echo "=== MUStARD --ablation $abl ==="
    python train_classification.py --dataset mustard --ablation "$abl" 2>&1 | tee "$LOG_DIR/mustard_${abl}.log"
done
echo "=== MUStARD all ablations done ==="
