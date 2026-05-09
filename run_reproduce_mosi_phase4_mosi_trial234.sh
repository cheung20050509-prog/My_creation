#!/usr/bin/env bash
# Reproduce CMU-MOSI Optuna trial 234 from logs/optuna/4090D_restart/phase4_mosi/
# (study infogate_mosi_phase4_mosi_4090d). Paper-aligned Best Results (Acc-7 50.80%, etc.).
#
# IMPORTANT: Float CLI args below match optuna_search_v2.py objective() formatting
# (.6e / .4f / .6f / str(ema_decay)), NOT the full-precision floats in mosi.log's
# ``Trial 234 finished`` line. Otherwise train.py sees different LRs (see gold log header).
# Sanity check: PYTHON scripts/verify_mosi_trial234_optuna_train_argv.py
#
# batch_config 0 => train_batch_size 16, gradient_accumulation_step 2 (tier-3 MOSI grid).
#
# --- Optional: rerun through Optuna so subprocess argv is built only by objective() ---
# Requires a DB that still contains COMPLETE trial 234 in the source study (full runs).
# Example: new empty DB + new study name, enqueue params from the canonical DB, then 1 trial:
#   ROOT="$(pwd)"
#   SRC_DB="sqlite:///${ROOT}/logs/optuna/4090D_restart/phase4_mosi/db/mosi.db"
#   NEW_DB="sqlite:///${ROOT}/logs/optuna/4090D_restart/phase4_mosi/db/mosi_enqueue_t234.db"
#   nohup env CUDA_VISIBLE_DEVICES=0 "${PYTHON}" -u optuna_search_v2.py \
#     --dataset mosi --gpu 0 --disable_two_stage_mosi --search_tier 3 \
#     --n_trials 1 --n_startup_trials 1 \
#     --n_epochs 110 --early_stop_patience 15 --selection_metric mae \
#     --study_name infogate_mosi_phase4_mosi_enqueue_t234 \
#     --db "${NEW_DB}" \
#     --artefact_root "${ROOT}/logs/optuna/4090D_restart/phase4_mosi/" \
#     --stage_label phase4_mosi \
#     --enqueue_trials_storage "${SRC_DB}" \
#     --enqueue_trials_study infogate_mosi_phase4_mosi_4090d \
#     --enqueue_trials_numbers 234 \
#     >> "${ROOT}/logs/optuna/4090D_restart/phase4_mosi/run/enqueue_trial234.log" 2>&1 &
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-/root/autodl-tmp/anaconda3/envs/ITHP5090/bin/python}"
if ! "$PYTHON" -c "import torch" 2>/dev/null; then
  PYTHON="${PYTHON:-python3}"
fi

REPRO_ROOT="${REPRO_ROOT:-${ROOT}/logs/optuna/mosi_reproduce_phase4_mosi_trial234}"
mkdir -p "$REPRO_ROOT/train_logs"
LOG="$REPRO_ROOT/train_logs/mosi_reproduce_trial234.log"
CKPT="$REPRO_ROOT/checkpoints/reproduce_trial234"
mkdir -p "$CKPT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# Reduces CUDA OOM from fragmentation on long DeBERTa runs (safe no-op if unsupported).
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "Logging to $LOG"
echo "Checkpoints under $CKPT"
echo "Tip: run when GPU is idle. Background:"
echo "  cd \"$ROOT\" && nohup env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True bash \"$0\" >>\"${REPRO_ROOT}/train_logs/nohup.wrapper.log\" 2>&1 &"

exec "$PYTHON" -u train.py \
  --dataset mosi \
  --n_epochs 98 \
  --train_batch_size 16 \
  --gradient_accumulation_step 2 \
  --checkpoint_dir "$CKPT" \
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
  2>&1 | tee "$LOG"
