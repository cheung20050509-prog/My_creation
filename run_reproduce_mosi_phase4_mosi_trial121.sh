#!/usr/bin/env bash
# Reproduce CMU-MOSI Optuna trial 121 from logs/optuna/4090D_restart/phase4_mosi/
# (study infogate_mosi_phase4_mosi_4090d). Invokes train.py with the same CLI shape as
# optuna_search_v2.py objective(); batch grid index 0 => (16, 2) under tier-3 MOSI candidates.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-/root/autodl-tmp/anaconda3/envs/ITHP5090/bin/python}"
if ! "$PYTHON" -c "import torch" 2>/dev/null; then
  PYTHON="${PYTHON:-python3}"
fi

REPRO_ROOT="${REPRO_ROOT:-${ROOT}/logs/optuna/mosi_reproduce_phase4_mosi_trial121}"
mkdir -p "$REPRO_ROOT/train_logs"
LOG="$REPRO_ROOT/train_logs/mosi_reproduce_trial121.log"
CKPT="$REPRO_ROOT/checkpoints/reproduce_trial121"
mkdir -p "$CKPT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "Logging to $LOG"
echo "Checkpoints under $CKPT"

exec "$PYTHON" -u train.py \
  --dataset mosi \
  --n_epochs 96 \
  --train_batch_size 16 \
  --gradient_accumulation_step 2 \
  --checkpoint_dir "$CKPT" \
  --selection_metric mae \
  --seed 128 \
  --learning_rate 3.10498544327124e-05 \
  --ig_learning_rate 2.327301375461295e-04 \
  --beta_ib 24.08947741595869 \
  --num_infogate_layers 3 \
  --bottleneck_dim 192 \
  --mse_weight 1.4714739295371693 \
  --dropout_prob 0.27884966454620314 \
  --alpha_ib 0.004116435342639878 \
  --stage1_epochs 7 \
  --warmup_proportion 0.11626737374740492 \
  --weight_decay 0.0007104949909263443 \
  --ema_decay 0.9951841989858001 \
  --selector_target_temp 0.7617380555153733 \
  --selector_rib_weight 0.05773578127779837 \
  --gumbel_tau_start 1.3184732422896817 \
  --gumbel_tau_end 0.1831571994055495 \
  --num_heads 8 \
  --unified_dim 128 \
  --ib_hidden_dim 128 \
  --ema_start_epoch 4 \
  --early_stop_patience 15 \
  2>&1 | tee "$LOG"
