#!/bin/bash
# InfoGate Binary Classification Training: UR-FUNNY (humor) / MUStARD (sarcasm)
#
# HKT-aligned setup (ALBERT + HCF + sliced A/V dims), mirrors the text-tower
# and feature-slicing conventions from https://github.com/matalvepu/HKT :
#   - text backbone: ALBERT-base-v2 (see ./albert-base-v2)
#   - acoustic: 60 dims (pkl[:, 0:60]), visual: 36 dims (pkl[:, 55:91]), HCF: 4 dims
#   - 4-modality InfoGate with MSelector routing
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PYTHON="${PYTHON:-python}"
MODEL="${MODEL:-./albert-base-v2}"
DATASET=${1:-ur_funny}
MAX_LEN=${2:-64}
EPOCHS=${3:-50}
STAGE1=${4:-8}
BATCH_SIZE=${5:-16}
LOG_SUFFIX=${6:-}

mkdir -p checkpoints logs

LOGFILE="logs/train_${DATASET}${LOG_SUFFIX}.log"
echo "Training InfoGate binary classifier on ${DATASET} for ${EPOCHS} epochs (stage1: ${STAGE1})"
echo "Backbone: ${MODEL}"
echo "Log: ${LOGFILE}"

nohup "$PYTHON" -u train_classify.py \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --max_seq_length "$MAX_LEN" \
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
    --alpha_ib 0.005 \
    --selector_target_temp 0.6 \
    --selector_rib_weight 0.05 \
    --dropout_prob 0.25 \
    --weight_decay 0.01 \
    --ema_decay 0.999 \
    --ema_start_epoch 5 \
    --selection_metric binary_acc \
    --checkpoint_dir checkpoints \
    --seed 42 \
    > "${LOGFILE}" 2>&1 &

echo "PID: $!"
echo "Monitor: tail -f ${LOGFILE}"
