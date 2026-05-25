#!/usr/bin/env bash
# Replay 4090D_restart gold trials under fixed_experiment/ (self-contained paths).
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
# Default to ITHP5090 (transformers 4.29.x), same as other fixed_experiment / ablation_study launchers.
PYTHON="${PYTHON:-/root/autodl-tmp/anaconda3/envs/ITHP5090/bin/python}"
LOGDIR="${PWD}/repro_gold_runs"
mkdir -p "$LOGDIR"

# MOSI phase4_mosi trial 234 (from phase4_mosi/run/mosi.log)
if [[ "${1:-}" == "mosi" || "${1:-}" == "all" ]]; then
  echo "[mosi] logging to ${LOGDIR}/mosi_trial234_fixedexp.log"
  CUDA_VISIBLE_DEVICES="${MOSI_GPU:-0}" "$PYTHON" -u train.py \
    --dataset mosi \
    --n_epochs 98 \
    --train_batch_size 16 \
    --gradient_accumulation_step 2 \
    --checkpoint_dir "${LOGDIR}/ckpt_mosi234" \
    --selection_metric mae \
    --seed 128 \
    --learning_rate 2.478617e-05 \
    --ig_learning_rate 2.266689e-04 \
    --beta_ib 23.3997 \
    --num_infogate_layers 3 \
    --bottleneck_dim 192 \
    --mse_weight 1.2123 \
    --dropout_prob 0.2500 \
    --alpha_ib 0.003726 \
    --stage1_epochs 7 \
    --warmup_proportion 0.1269 \
    --weight_decay 0.000729 \
    --ema_decay 0.9951979795863604 \
    --selector_target_temp 0.7255 \
    --selector_rib_weight 0.0636 \
    --gumbel_tau_start 1.1135 \
    --gumbel_tau_end 0.1685 \
    --num_heads 8 \
    --unified_dim 128 \
    --ib_hidden_dim 128 \
    --ema_start_epoch 5 \
    --early_stop_patience 15 \
    2>&1 | tee "${LOGDIR}/mosi_trial234_fixedexp.log"
fi

# MOSEI phase1 trial 70 (tier1; remaining knobs = optuna_search_v2 DEFAULTS)
if [[ "${1:-}" == "mosei" || "${1:-}" == "all" ]]; then
  echo "[mosei] logging to ${LOGDIR}/mosei_trial70_fixedexp.log"
  CUDA_VISIBLE_DEVICES="${MOSEI_GPU:-1}" "$PYTHON" -u train.py \
    --dataset mosei \
    --n_epochs 50 \
    --train_batch_size 4 \
    --gradient_accumulation_step 8 \
    --checkpoint_dir "${LOGDIR}/ckpt_mosei70" \
    --selection_metric mae \
    --seed 128 \
    --learning_rate 3.078835e-05 \
    --ig_learning_rate 5.115383e-04 \
    --beta_ib 21.7491 \
    --num_infogate_layers 3 \
    --bottleneck_dim 64 \
    --mse_weight 2.1455 \
    --dropout_prob 0.2681 \
    --alpha_ib 0.01 \
    --stage1_epochs 10 \
    --warmup_proportion 0.1 \
    --weight_decay 0.001 \
    --ema_decay 0.999 \
    --selector_target_temp 0.35 \
    --selector_rib_weight 0.05 \
    --gumbel_tau_start 1.0 \
    --gumbel_tau_end 0.5 \
    --num_heads 4 \
    --unified_dim 256 \
    --ib_hidden_dim 256 \
    --ema_start_epoch 5 \
    2>&1 | tee "${LOGDIR}/mosei_trial70_fixedexp.log"
fi

# UR-FUNNY v2 trial 162 (paper classification Acc 75.15%)
if [[ "${1:-}" == "ur_funny" || "${1:-}" == "classify" || "${1:-}" == "all" ]]; then
  echo "[ur_funny] logging to ${LOGDIR}/ur_funny_trial162_fixedexp.log"
  CUDA_VISIBLE_DEVICES="${UR_FUNNY_GPU:-0}" "$PYTHON" -u fixed_experiment/train_fixed_ur_funny_trial162.py \
    --checkpoint-dir "${LOGDIR}/ckpt_ur_funny162" \
    2>&1 | tee "${LOGDIR}/ur_funny_trial162_fixedexp.log"
fi

# MUStARD s2_local trial 134 (paper classification Acc 79.41%)
if [[ "${1:-}" == "mustard" || "${1:-}" == "classify" || "${1:-}" == "all" ]]; then
  echo "[mustard] logging to ${LOGDIR}/mustard_trial134_fixedexp.log"
  CUDA_VISIBLE_DEVICES="${MUSTARD_GPU:-0}" "$PYTHON" -u fixed_experiment/train_fixed_mustard_s2_local_trial134.py \
    --checkpoint-dir "${LOGDIR}/ckpt_mustard134" \
    2>&1 | tee "${LOGDIR}/mustard_trial134_fixedexp.log"
fi
