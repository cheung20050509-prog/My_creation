#!/bin/bash
# UR-FUNNY — 6 architectural ablation experiments
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$DIR/logs"
mkdir -p "$LOG_DIR"
cd "$DIR"

ABLATIONS=("none" "no_infogate" "no_mselector" "no_ib" "no_conf_gating" "no_adaptive_gate")

for abl in "${ABLATIONS[@]}"; do
    echo "=== UR-FUNNY --ablation $abl ==="
    python train_classification.py --dataset ur_funny --ablation "$abl" 2>&1 | tee "$LOG_DIR/ur_funny_${abl}.log"
done
echo "=== UR-FUNNY all ablations done ==="
