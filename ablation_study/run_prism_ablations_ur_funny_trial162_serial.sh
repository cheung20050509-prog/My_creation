#!/usr/bin/env bash
# Serial PRISM ablations for UR-FUNNY v2 trial 162 (paper MHD / classification row).
set -euo pipefail

MY_CREATION="$(cd "$(dirname "$0")/.." && pwd)"
cd "$MY_CREATION"

PYTHON="${PYTHON:-/root/autodl-tmp/anaconda3/envs/ITHP5090/bin/python}"
if ! "$PYTHON" -c "import torch" 2>/dev/null; then
  PYTHON="${PYTHON:-python3}"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TQDM_DISABLE="${TQDM_DISABLE:-1}"

# Paper-facing names:
#   no_ib        = w/o VTB
#   no_conf      = w/o confidence bias/value gating
#   no_mselector = w/o DPR
#   no_infogate  = w/o InfoGate
# shellcheck disable=SC2206
MODES=( ${MODES:-no_ib no_conf no_mselector no_infogate} )

MASTER_LOG="${MASTER_LOG:-${MY_CREATION}/ablation_study/runs/ur_funny_trial162_ablation_serial.log}"
PID_FILE="${PID_FILE:-${MY_CREATION}/ablation_study/runs/ur_funny_trial162_ablation_serial.pid}"
mkdir -p "$(dirname "$MASTER_LOG")"
echo "$$" > "$PID_FILE"

log() {
  echo "[$(date -Iseconds)] $*" | tee -a "$MASTER_LOG"
}

log "UR-FUNNY v2 trial162 serial ablations on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
log "Modes: ${MODES[*]}"

for mode in "${MODES[@]}"; do
  OUT="${MY_CREATION}/ablation_study/runs/ur_funny_trial162_${mode}"
  mkdir -p "${OUT}/checkpoints"

  log "--- start mode=${mode} out=${OUT} ---"
  "$PYTHON" -u ablation_study/train_fixed_ur_funny_trial162.py \
    --ablation "${mode}" \
    --checkpoint-dir "${OUT}/checkpoints" \
    2>&1 | tee "${OUT}/train.log"
  log "--- done mode=${mode} ---"
done

log "All UR-FUNNY v2 trial162 ablations finished."
