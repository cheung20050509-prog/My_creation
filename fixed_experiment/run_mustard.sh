#!/bin/bash
# Fixed MUStARD experiment — trial 26 best hparams
# Paper: Acc=75.0%, F1=74.9%
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$DIR/logs"
mkdir -p "$LOG_DIR"

cd "$DIR"

python train_classification.py --dataset mustard 2>&1 | tee "$LOG_DIR/mustard.log"
