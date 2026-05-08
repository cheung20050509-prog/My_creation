#!/bin/bash
# MOSEI — 6 architectural ablation experiments
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$DIR/logs"
mkdir -p "$LOG_DIR"
cd "$DIR"

ABLATIONS=("none" "no_infogate" "no_mselector" "no_ib" "no_conf_gating" "no_adaptive_gate")

for abl in "${ABLATIONS[@]}"; do
    echo "=== MOSEI --ablation $abl ==="
    python train_regression.py --dataset mosei --ablation "$abl" 2>&1 | tee "$LOG_DIR/mosei_${abl}.log"
done
echo "=== MOSEI all ablations done ==="
