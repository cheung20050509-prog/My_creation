#!/bin/bash
# InfoGate Test Script
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PYTHON="${PYTHON:-python}"
DATASET=${1:-mosi}
CHECKPOINT=${2:-checkpoints/infogate_${DATASET}_best.pt}

echo "Testing InfoGate on ${DATASET}"
echo "Checkpoint: ${CHECKPOINT}"

"$PYTHON" -u test.py \
    --dataset "$DATASET" \
    --checkpoint "$CHECKPOINT" \
    --unified_dim 256 \
    --ib_hidden_dim 256 \
    --bottleneck_dim 128 \
    --num_heads 4 \
    --num_infogate_layers 3 \
    --beta_ib 32 \
    --gamma_cyc 10 \
    --cra_layers 8 \
    --dropout_prob 0.1 \
    --seed 128
