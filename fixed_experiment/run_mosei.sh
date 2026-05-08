#!/bin/bash
# Fixed MOSEI experiment — trial 70 best hparams
# Paper: MAE=0.499, Corr=0.800
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$DIR/logs"
mkdir -p "$LOG_DIR"

cd "$DIR"

python train_regression.py --dataset mosei 2>&1 | tee "$LOG_DIR/mosei.log"
