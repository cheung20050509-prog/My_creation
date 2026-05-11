#!/bin/bash
# InfoGate Binary Classification Test Script (UR-FUNNY / MUSTARD)
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PYTHON="${PYTHON:-python}"
DATASET=${1:-ur_funny}
CHECKPOINT=${2:-checkpoints/infogate_${DATASET}_best.pt}

# HKT-aligned default max_seq_length: 64 humor / 77 sarcasm.
if [ "${DATASET}" = "mustard" ]; then
    DEFAULT_MAX_LEN=77
else
    DEFAULT_MAX_LEN=64
fi
MAX_LEN=${3:-$DEFAULT_MAX_LEN}

echo "Testing InfoGate binary classifier on ${DATASET}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Max seq length: ${MAX_LEN}"

"$PYTHON" -u test_classify.py \
    --dataset "$DATASET" \
    --checkpoint "$CHECKPOINT" \
    --max_seq_length "$MAX_LEN" \
    --test_batch_size 128 \
    --seed 42
