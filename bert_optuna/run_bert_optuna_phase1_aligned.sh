#!/usr/bin/env bash
# BERT-base-uncased Optuna on MOSI + MOSEI, budgets aligned with run_optuna_4090d_restart.sh phase1.
# Tier-1 only (no align_mix_floor in search space). For align_floor Tier-3 use run_bert_new_space_align_floor.sh.
# Default: MOSI on physical GPU1, MOSEI on physical GPU0 — both start together; launcher waits for both.
#
# Usage:
#   nohup ./bert_optuna/run_bert_optuna_phase1_aligned.sh >> bert_optuna/logs/phase1/run/launcher.nohup 2>&1 &
# Env:
#   PYTHON, MOSI_GPU (default 1), MOSEI_GPU (default 0), BERT_MODEL (default: My_creation/bert-base-uncase)
#   RUN_MOSI=0 | RUN_MOSEI=0 to skip one leg.
#   SERIAL=1: run MOSI to completion first, then MOSEI (same card defaults as above unless you override GPUs).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MY_CREATION="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$MY_CREATION"

PYTHON="${PYTHON:-/root/autodl-tmp/anaconda3/envs/ITHP5090/bin/python}"
MOSI_GPU="${MOSI_GPU:-1}"
MOSEI_GPU="${MOSEI_GPU:-0}"
BERT_MODEL="${BERT_MODEL:-$MY_CREATION/bert-base-uncase}"
RUN_MOSI="${RUN_MOSI:-1}"
RUN_MOSEI="${RUN_MOSEI:-1}"
SERIAL="${SERIAL:-0}"

if [[ ! -d "$BERT_MODEL" ]]; then
  echo "ERROR: BERT weights directory not found: $BERT_MODEL" >&2
  echo "  Run:  bash $SCRIPT_DIR/download_bert_base_uncased.sh" >&2
  exit 1
fi
BERT_MODEL="$(cd "$(dirname "$BERT_MODEL")" && pwd)/$(basename "$BERT_MODEL")"

PHASE_ROOT="$SCRIPT_DIR/logs/phase1"
mkdir -p "$PHASE_ROOT/db" "$PHASE_ROOT/run" "$PHASE_ROOT/train_logs" "$PHASE_ROOT/checkpoints"

DBBASE="sqlite:///${PHASE_ROOT}/db"
echo "[$(date -Iseconds)] PHASE_ROOT=$PHASE_ROOT"
echo "[$(date -Iseconds)] BERT_MODEL=$BERT_MODEL"
echo "[$(date -Iseconds)] GPUs: MOSI_GPU=$MOSI_GPU MOSEI_GPU=$MOSEI_GPU SERIAL=$SERIAL"

run_mosi () {
  local log="$PHASE_ROOT/run/mosi_phase1_bert.log"
  {
    echo ""
    echo "######## [$(date -Iseconds)] MOSI BERT phase1 (GPU $MOSI_GPU) ########"
  } >> "$log"
  nohup env CUDA_VISIBLE_DEVICES="$MOSI_GPU" "$PYTHON" -u optuna_search_v2.py \
    --dataset mosi \
    --gpu 0 \
    --search_tier 1 \
    --n_trials 100 \
    --stage1_trials 40 \
    --stage2_trials 60 \
    --stage2_top_k 8 \
    --n_startup_trials 20 \
    --selection_metric mae \
    --no_dataset_overrides \
    --study_name infogate_mosi_phase1_bert \
    --db "${DBBASE}/mosi.db" \
    --pretrained_model "$BERT_MODEL" \
    >>"$log" 2>&1 &
  # Must set PID in this shell (not via $(run_mosi)); otherwise wait cannot reap the child.
  MOSI_PID=$!
}

run_mosei () {
  local log="$PHASE_ROOT/run/mosei_phase1_bert.log"
  {
    echo ""
    echo "######## [$(date -Iseconds)] MOSEI BERT phase1 (GPU $MOSEI_GPU) ########"
  } >> "$log"
  nohup env CUDA_VISIBLE_DEVICES="$MOSEI_GPU" "$PYTHON" -u optuna_search_v2.py \
    --dataset mosei \
    --gpu 0 \
    --search_tier 1 \
    --n_trials 80 \
    --n_startup_trials 55 \
    --selection_metric mae \
    --no_dataset_overrides \
    --study_name infogate_mosei_phase1_bert \
    --db "${DBBASE}/mosei.db" \
    --stage_label phase1_bert \
    --pretrained_model "$BERT_MODEL" \
    >>"$log" 2>&1 &
  MOSEI_PID=$!
}

if [[ "$SERIAL" == "1" ]]; then
  if [[ "$RUN_MOSI" == "1" ]]; then
    run_mosi
    echo "[$(date -Iseconds)] MOSI driver PID=$MOSI_PID  tail -F $PHASE_ROOT/run/mosi_phase1_bert.log"
    wait "$MOSI_PID"
    echo "[$(date -Iseconds)] MOSI finished."
  else
    echo "[$(date -Iseconds)] SKIP MOSI (RUN_MOSI=0)"
  fi
  if [[ "$RUN_MOSEI" == "1" ]]; then
    run_mosei
    echo "[$(date -Iseconds)] MOSEI driver PID=$MOSEI_PID  tail -F $PHASE_ROOT/run/mosei_phase1_bert.log"
    wait "$MOSEI_PID"
    echo "[$(date -Iseconds)] MOSEI finished."
  else
    echo "[$(date -Iseconds)] SKIP MOSEI (RUN_MOSEI=0)"
  fi
else
  WAIT_PIDS=()
  if [[ "$RUN_MOSI" == "1" ]]; then
    run_mosi
    echo "[$(date -Iseconds)] MOSI driver PID=$MOSI_PID  tail -F $PHASE_ROOT/run/mosi_phase1_bert.log"
    WAIT_PIDS+=("$MOSI_PID")
  else
    echo "[$(date -Iseconds)] SKIP MOSI (RUN_MOSI=0)"
  fi
  if [[ "$RUN_MOSEI" == "1" ]]; then
    run_mosei
    echo "[$(date -Iseconds)] MOSEI driver PID=$MOSEI_PID  tail -F $PHASE_ROOT/run/mosei_phase1_bert.log"
    WAIT_PIDS+=("$MOSEI_PID")
  else
    echo "[$(date -Iseconds)] SKIP MOSEI (RUN_MOSEI=0)"
  fi
  if ((${#WAIT_PIDS[@]} > 0)); then
    echo "[$(date -Iseconds)] waiting on PIDs: ${WAIT_PIDS[*]}"
    wait "${WAIT_PIDS[@]}"
  fi
  echo "[$(date -Iseconds)] all search drivers finished."
fi

echo "[$(date -Iseconds)] all done."
