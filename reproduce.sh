#!/bin/bash
# Reproduce the train_mosi_no_ord_final_github.log results
# Parameters extracted from log + README + defaults
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PYTHON="${PYTHON:-python}"
mkdir -p checkpoints logs

LOGFILE="logs/train_mosi_reproduce.log"
echo "Reproducing InfoGate best results on MOSI"
echo "Log: ${LOGFILE}"

nohup "$PYTHON" -u train.py \
    --dataset mosi \
    --n_epochs 80 \
    --stage1_epochs 12 \
    --train_batch_size 16 \
    --dev_batch_size 128 \
    --test_batch_size 128 \
    --gradient_accumulation_step 2 \
    --max_seq_length 50 \
    --learning_rate 1.14e-5 \
    --ig_learning_rate 1.93e-4 \
    --warmup_proportion 0.1 \
    --weight_decay 0.005 \
    --seed 42 \
    --unified_dim 256 \
    --ib_hidden_dim 256 \
    --bottleneck_dim 96 \
    --num_heads 4 \
    --num_infogate_layers 4 \
    --beta_ib 15.6 \
    --gamma_cyc 0.582 \
    --alpha_ib 0.00227 \
    --alpha_nce 0.05 \
    --alpha_sac 0.02 \
    --mse_weight 0.67 \
    --cra_layers 8 \
    --cra_dims "64,32,16" \
    --dropout_prob 0.195 \
    --ema_decay 0.999 \
    --ema_start_epoch 999 \
    --checkpoint_dir checkpoints \
    > "${LOGFILE}" 2>&1 &

echo "PID: $!"
echo "Monitor: tail -f ${LOGFILE}"
