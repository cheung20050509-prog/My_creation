#!/bin/bash
# All ablation experiments (5 datasets x 6 ablations = 30 runs)
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"

echo "============================================"
echo "  PRISM Ablation Study — All Experiments"
echo "============================================"

echo && echo "=== [1/5] MOSI ===" && bash "$DIR/run_mosi.sh"
echo && echo "=== [2/5] MOSEI ===" && bash "$DIR/run_mosei.sh"
echo && echo "=== [3/5] SIMSv2 ===" && bash "$DIR/run_simsv2.sh"
echo && echo "=== [4/5] UR-FUNNY ===" && bash "$DIR/run_ur_funny.sh"
echo && echo "=== [5/5] MUStARD ===" && bash "$DIR/run_mustard.sh"

echo && echo "=== All ablation experiments done ==="
