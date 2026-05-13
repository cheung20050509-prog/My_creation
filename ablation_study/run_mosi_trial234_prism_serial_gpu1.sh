#!/usr/bin/env bash
# MOSI trial234: serial PRISM ablations on one GPU (default physical GPU 1).
# Order: none → no_ib (w/o VTB) → no_mselector (w/o DPR) → no_infogate (w/o InfoGate).
#
# Usage (from anywhere):
#   bash /path/to/My_creation/ablation_study/run_mosi_trial234_prism_serial_gpu1.sh
#
# Environment:
#   CUDA_VISIBLE_DEVICES  default 1 (physical GPU 1; process sees cuda:0)
#   MODES                 space-separated ablation flags (default below)
#   PYTHON                python with torch (defaults like run_mosi_trial234.sh)
#   DRY_RUN=1             only print train argv per mode, no training
#
set -euo pipefail
set -o pipefail

MY_CREATION="$(cd "$(dirname "$0")/.." && pwd)"
cd "$MY_CREATION"

PYTHON="${PYTHON:-/root/autodl-tmp/anaconda3/envs/ITHP5090/bin/python}"
if ! "$PYTHON" -c "import torch" 2>/dev/null; then
  PYTHON="${PYTHON:-python3}"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# shellcheck disable=SC2206
MODES=( ${MODES:-none no_ib no_mselector no_infogate} )

DRY_FLAG=()
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  DRY_FLAG=(--dry-run)
fi

echo "======== $(date -Is) MOSI trial234 serial PRISM ablations ========"
echo "MY_CREATION=${MY_CREATION}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "MODES=${MODES[*]}"
echo "PYTHON=${PYTHON}"
echo "DRY_RUN=${DRY_RUN:-0}"
echo "================================================================"

for m in "${MODES[@]}"; do
  echo "-------- $(date -Is) start ablation=${m} --------"
  if [[ "$m" == "none" ]]; then
    OUT="${MY_CREATION}/ablation_study/runs/mosi_trial234"
  else
    OUT="${MY_CREATION}/ablation_study/runs/mosi_trial234_${m}"
  fi
  mkdir -p "${OUT}/checkpoints"
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    "$PYTHON" -u ablation_study/train_fixed_mosi_trial234.py \
      --ablation "$m" \
      --checkpoint-dir "${OUT}/checkpoints" \
      "${DRY_FLAG[@]}"
  else
    "$PYTHON" -u ablation_study/train_fixed_mosi_trial234.py \
      --ablation "$m" \
      --checkpoint-dir "${OUT}/checkpoints" \
      2>&1 | tee -a "${OUT}/train.log"
  fi
  echo "-------- $(date -Is) done ablation=${m} --------"
done

echo "======== $(date -Is) all modes finished ========"
