#!/bin/bash
# InfoGate Training Script
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PYTHON="${PYTHON:-python}"
DATASET=${1:-mosi}
EPOCHS=${2:-80}
STAGE1=${3:-8}
BATCH_SIZE=${4:-16}

mkdir -p checkpoints logs

echo "Training InfoGate on ${DATASET} for ${EPOCHS} epochs (stage1: ${STAGE1})"

nohup "$PYTHON" -u train.py \
    --dataset "$DATASET" \
    --n_epochs "$EPOCHS" \
    --stage1_epochs "$STAGE1" \
    --train_batch_size "$BATCH_SIZE" \
    --gradient_accumulation_step 2 \
    --learning_rate 2e-5 \
    --ig_learning_rate 5e-4 \
    --unified_dim 256 \
    --ib_hidden_dim 256 \
    --bottleneck_dim 128 \
    --num_heads 4 \
    --num_infogate_layers 3 \
    --beta_ib 16 \
    --gamma_cyc 1.0 \
    --alpha_ib 0.005 \
    --alpha_nce 0.05 \
    --cra_layers 8 \
    --dropout_prob 0.25 \
    --weight_decay 0.01 \
    --checkpoint_dir checkpoints \
    --seed 128 \
    > "logs/train_${DATASET}.log" 2>&1 &

echo "PID: $!"
echo "Log: logs/train_${DATASET}.log"
