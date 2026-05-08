#!/bin/bash
# Run all fixed experiments sequentially
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== [1/5] MOSI ===" && bash "$DIR/run_mosi.sh"
echo "=== [2/5] MOSEI ===" && bash "$DIR/run_mosei.sh"
echo "=== [3/5] SIMSv2 ===" && bash "$DIR/run_simsv2.sh"
echo "=== [4/5] UR-FUNNY ===" && bash "$DIR/run_ur_funny.sh"
echo "=== [5/5] MUStARD ===" && bash "$DIR/run_mustard.sh"
echo "=== All done ==="
