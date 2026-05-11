#!/usr/bin/env bash
# Multi-seed verification of each dataset's current best-MAE Optuna trial.
# MOSI row: 4090D_restart phase3 s2_local trial 89 (MAE 0.5978).
# Reproduces train.py with the saved best hparams using N different seeds,
# so you can report mean ± std in the paper.
#
# Usage:
#   ./multi_seed_verify.sh mosi
#   ./multi_seed_verify.sh mosei
#   ./multi_seed_verify.sh simsv2
#   ./multi_seed_verify.sh all
#
# Env overrides:
#   SEEDS="42 128 256 1024 2024"      seeds to sweep (default 5)
#   GPU=0                              physical GPU id
set -euo pipefail
cd "$(dirname "$0")"

CONDA_BASE="${CONDA_BASE:-/root/autodl-tmp/anaconda3}"
CONDA_ENV="${CONDA_ENV:-ITHP5090}"
if [[ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
fi
PYTHON="${CONDA_PREFIX:-}/bin/python"
[[ -x "$PYTHON" ]] || PYTHON="${PYTHON:-/root/autodl-tmp/anaconda3/envs/ITHP5090/bin/python}"

SEEDS="${SEEDS:-42 128 256 1024 2024}"
GPU="${GPU:-0}"
ROOT="$PWD"
OUT_DIR="$ROOT/logs/multi_seed_verify_$(date -u +%Y%m%d_%H%M%S)"
mkdir -p "$OUT_DIR"

# ---- best hparams (extracted from saved_hparams/<ds>_best_hparams.json) ----

run_mosi () {
  # 4090D_restart phase3 s2_local trial 89, test MAE 0.5978 (Best Results), Tier 3
  local seed=$1
  local logfile="$OUT_DIR/mosi_seed${seed}.log"
  local ckpt_dir="$OUT_DIR/ckpts/mosi_seed${seed}"
  mkdir -p "$ckpt_dir"
  echo "[mosi seed=$seed] -> $logfile"
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -u train.py \
    --dataset mosi \
    --train_batch_size 16 --gradient_accumulation_step 4 \
    --n_epochs 99 --stage1_epochs 8 \
    --learning_rate 4.0435560151827146e-05 --ig_learning_rate 1.9555720219325748e-04 \
    --beta_ib 7.385208105955388 --alpha_ib 0.004458028629011018 \
    --num_infogate_layers 3 --bottleneck_dim 192 --unified_dim 128 --num_heads 8 --ib_hidden_dim 128 \
    --mse_weight 1.4017893314668146 --dropout_prob 0.2878573110199772 \
    --warmup_proportion 0.11123528535141979 --weight_decay 0.0003603150408561708 \
    --ema_decay 0.995 --ema_start_epoch 8 \
    --selector_target_temp 0.6891975097112971 --selector_rib_weight 0.06135064083078536 \
    --gumbel_tau_start 1.1626945117846452 --gumbel_tau_end 0.2645430459011354 \
    --selection_metric mae \
    --checkpoint_dir "$ckpt_dir" \
    --seed "$seed" \
    > "$logfile" 2>&1
}

run_mosei () {
  # mosei trial 37, MAE 0.4941, Tier 2
  local seed=$1
  local logfile="$OUT_DIR/mosei_seed${seed}.log"
  local ckpt_dir="$OUT_DIR/ckpts/mosei_seed${seed}"
  mkdir -p "$ckpt_dir"
  echo "[mosei seed=$seed] -> $logfile"
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -u train.py \
    --dataset mosei \
    --train_batch_size 4 --gradient_accumulation_step 8 \
    --n_epochs 40 --stage1_epochs 6 \
    --learning_rate 4.112e-5 --ig_learning_rate 1.820e-3 \
    --beta_ib 5.447 --alpha_ib 0.002062 \
    --num_infogate_layers 4 --bottleneck_dim 96 --unified_dim 256 --num_heads 4 --ib_hidden_dim 256 \
    --mse_weight 1.658 --dropout_prob 0.316 \
    --warmup_proportion 0.197 --weight_decay 0.090682 \
    --ema_decay 0.999 --ema_start_epoch 5 \
    --selector_target_temp 0.35 --selector_rib_weight 0.05 \
    --gumbel_tau_start 1.0 --gumbel_tau_end 0.5 \
    --selection_metric mae \
    --checkpoint_dir "$ckpt_dir" \
    --seed "$seed" \
    > "$logfile" 2>&1
}

run_simsv2 () {
  # simsv2 v1 trial 196, MAE 0.3160, Tier 2
  local seed=$1
  local logfile="$OUT_DIR/simsv2_seed${seed}.log"
  local ckpt_dir="$OUT_DIR/ckpts/simsv2_seed${seed}"
  mkdir -p "$ckpt_dir"
  echo "[simsv2 seed=$seed] -> $logfile"
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -u train.py \
    --dataset simsv2 \
    --train_batch_size 64 --gradient_accumulation_step 1 \
    --n_epochs 67 --stage1_epochs 12 \
    --learning_rate 3.922e-5 --ig_learning_rate 2.741e-4 \
    --beta_ib 11.924 --alpha_ib 0.003772 \
    --num_infogate_layers 3 --bottleneck_dim 128 --unified_dim 256 --num_heads 4 --ib_hidden_dim 256 \
    --mse_weight 1.720 --dropout_prob 0.058 \
    --warmup_proportion 0.135 --weight_decay 0.008629 \
    --ema_decay 0.9995 --ema_start_epoch 5 \
    --selector_target_temp 0.35 --selector_rib_weight 0.05 \
    --gumbel_tau_start 1.0 --gumbel_tau_end 0.5 \
    --selection_metric mae \
    --checkpoint_dir "$ckpt_dir" \
    --seed "$seed" \
    > "$logfile" 2>&1
}

DATASETS="${1:-all}"
if [[ "$DATASETS" == "all" ]]; then DATASETS="mosi mosei simsv2"; fi

for ds in $DATASETS; do
  for seed in $SEEDS; do
    case "$ds" in
      mosi)   run_mosi   "$seed" ;;
      mosei)  run_mosei  "$seed" ;;
      simsv2) run_simsv2 "$seed" ;;
      *) echo "unknown dataset $ds" >&2; exit 1 ;;
    esac
  done
done

echo
echo "All runs done. Aggregate summary:"
echo "  $OUT_DIR/"
"$PYTHON" - <<PYEOF
import os, re, glob
out = "$OUT_DIR"
pat = re.compile(r"^Best Results.*?MAE:\s+([\d.]+).*?Corr:\s+([\d.]+).*?F1:\s+([\d.]+)", re.S)
for f in sorted(glob.glob(os.path.join(out, "*_seed*.log"))):
    txt = open(f).read()
    m = pat.search(txt)
    if not m:
        print(f"  {os.path.basename(f)}: PARSE FAIL")
        continue
    mae, corr, f1 = m.groups()
    print(f"  {os.path.basename(f):<35}  MAE={mae}  Corr={corr}  F1={f1}")
PYEOF
