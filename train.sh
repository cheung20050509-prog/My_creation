#!/bin/bash
# InfoGate Training Script
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PYTHON="${PYTHON:-python}"
DATASET=${1:-mosi}
EPOCHS=${2:-80}
STAGE1=${3:-8}
BATCH_SIZE=${4:-16}
LOG_SUFFIX=${5:-}

mkdir -p checkpoints logs

LOGFILE="logs/train_${DATASET}${LOG_SUFFIX}.log"
echo "Training InfoGate on ${DATASET} for ${EPOCHS} epochs (stage1: ${STAGE1})"
echo "Log: ${LOGFILE}"

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
    --mse_weight 0.5 \
    --cra_layers 8 \
    --dropout_prob 0.25 \
    --weight_decay 0.01 \
    --ema_decay 0.999 \
    --ema_start_epoch 5 \
    --checkpoint_dir checkpoints \
    --seed 42 \
    > "${LOGFILE}" 2>&1 &

echo "PID: $!"
