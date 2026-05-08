#!/bin/bash
# Fixed SIMSv2 experiment — trial 52 best hparams
# Paper: MAE=0.311, Corr=0.686
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$DIR/logs"
mkdir -p "$LOG_DIR"

cd "$DIR"

python train_regression.py --dataset simsv2 2>&1 | tee "$LOG_DIR/simsv2.log"
