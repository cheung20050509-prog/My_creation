#!/bin/bash
# Fixed MOSI experiment — trial 234 best hparams
# Paper: MAE=0.594, Corr=0.857
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$DIR/logs"
mkdir -p "$LOG_DIR"

cd "$DIR"

python train_regression.py --dataset mosi 2>&1 | tee "$LOG_DIR/mosi.log"
