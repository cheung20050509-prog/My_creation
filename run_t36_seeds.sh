#!/bin/bash
# Fixed parameters for Trial #36 but run across multiple seeds

cd "$(dirname "$0")"

CONDA_BASE="${CONDA_BASE:-/root/autodl-tmp/anaconda3}"
CONDA_ENV="${CONDA_ENV:-ITHP5090}"
if [[ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PYTHON="${CONDA_PREFIX}/bin/python"

mkdir -p checkpoints/reproduce logs/reproduce

SEEDS=(42 128 256 1024 )

for seed in "${SEEDS[@]}"; do
    LOGFILE="logs/reproduce/mosi_t36_seed_${seed}.log"
    echo "Running Trial 36 reproduction on MOSI with seed ${seed} -> ${LOGFILE}"
    
    nohup "$PYTHON" -u train.py \
        --dataset mosi \
        --n_epochs 84 \
        --stage1_epochs 10 \
        --train_batch_size 8 \
        --gradient_accumulation_step 8 \
        --dev_batch_size 128 \
        --test_batch_size 128 \
        --learning_rate 3.8e-05 \
        --ig_learning_rate 2.25e-4 \
        --beta_ib 61.367 \
        --num_infogate_layers 5 \
        --bottleneck_dim 96 \
        --mse_weight 0.878 \
        --dropout_prob 0.173 \
        --gamma_cyc 0.277 \
        --alpha_ib 0.00203 \
        --warmup_proportion 0.164 \
        --weight_decay 0.000938 \
        --ema_decay 0.995 \
        --text_residual_weight 0.217 \
        --selection_metric mae \
        --seed "${seed}" \
        --checkpoint_dir "checkpoints/reproduce/t36_seed_${seed}" \
        > "${LOGFILE}" 2>&1 &
        
    echo "PID: $!"
    sleep 2
done
echo "All runs submitted. Monitor: tail -f logs/reproduce/mosi_t36_seed_*.log"
