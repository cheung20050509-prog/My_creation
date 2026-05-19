#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# SIMSv2 — fresh Tier-3 single-stage search (align_mix_floor in space).
# Same artefact tree as MOSI/MOSEI under regression/.
#
# Do NOT set ENQUEUE_TOP_SIMSV2 (legacy SIMSv2 trials used old MMSA protocol).
#
# Usage (shares GPU 0 with MOSI/MOSEI drivers — time-multiplexed):
#   bash .../new_space_search/run_simsv2_new_space_align_floor.sh
#
# Fresh restart (wipe study + artefacts):
#   FRESH=1 bash .../run_simsv2_new_space_align_floor.sh
# -----------------------------------------------------------------------------
set -euo pipefail

MY="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEFAULT_PY="/root/autodl-tmp/anaconda3/envs/ITHP5090/bin/python"
PY="${PYTHON:-$DEFAULT_PY}"
GPU="${GPU:-0}"
FRESH="${FRESH:-0}"
N_TRIALS="${N_TRIALS:-100}"
N_STARTUP="${N_STARTUP_TRIALS:-20}"

cd "$MY"

R_BASE="$MY/logs/optuna/4090D_restart/new_space_search/regression"
DB_PATH="$R_BASE/db/infogate_simsv2_new_space_align_floor.db"
STUDY="infogate_simsv2_new_space_align_floor_tpe"
LOG="$R_BASE/run/simsv2_driver.log"

mkdir -p "$R_BASE"/{db,train_logs,checkpoints,run}

if [[ "$FRESH" == "1" ]]; then
  echo "[$(date -Iseconds)] FRESH=1: removing SIMSv2 new_space DB and driver log"
  rm -f "$DB_PATH" "$LOG"
  rm -rf "$R_BASE/checkpoints/optuna_simsv2" 2>/dev/null || true
  rm -f "$R_BASE"/train_logs/simsv2_trial_*.log 2>/dev/null || true
fi

if pgrep -f "optuna_search_v2.py --dataset simsv2.*infogate_simsv2_new_space_align_floor" >/dev/null 2>&1; then
  echo "ERROR: SIMSv2 new_space driver already running. Stop it first or use FRESH=1 after kill." >&2
  exit 1
fi

{
  echo ""
  echo "######## [$(date -Iseconds)] SIMSv2 new_space align_floor (GPU $GPU) ########"
} >>"$LOG"

nohup env CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u optuna_search_v2.py \
  --dataset simsv2 \
  --gpu 0 \
  --search_tier 3 \
  --micro_refine none \
  --n_trials "$N_TRIALS" \
  --n_startup_trials "$N_STARTUP" \
  --selection_metric mae \
  --study_name "$STUDY" \
  --db "sqlite:///${DB_PATH}" \
  --artefact_root "$R_BASE" \
  >>"$LOG" 2>&1 &

PID=$!
echo "[$(date -Iseconds)] SIMSv2 driver PID=$PID"
echo "  log: $LOG"
echo "  db:  $DB_PATH"
echo "  tail -F $LOG"
