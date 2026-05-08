#!/bin/bash
# Fixed UR-FUNNY experiment — trial 34 best hparams
# Paper: Acc=74.5%, F1=74.4%
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$DIR/logs"
mkdir -p "$LOG_DIR"

cd "$DIR"

python train_classification.py --dataset ur_funny 2>&1 | tee "$LOG_DIR/ur_funny.log"
