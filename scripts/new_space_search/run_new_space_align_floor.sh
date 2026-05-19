#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Single-stage Tier-3 (full joint) search including align_mix_floor (0..0.6).
# Layout:
#   regression/{db,train_logs,checkpoints,run}
#   classification/{db,train_logs,checkpoints,run}
#
# Warm-start (optional): set comma-separated sqlite URIs (see optuna_search_v2
# / optuna_search_classify --enqueue_top_from). Do NOT enqueue legacy SIMSv2
# trials here (data protocol / MMSA path changed).
#
# Usage:
#   export GPU=0
#   export ENQUEUE_TOP_MOSI="sqlite:////path/to/old.db"   # optional
#   bash My_creation/scripts/new_space_search/run_new_space_align_floor.sh
#   (runtime artefacts still under logs/optuna/4090D_restart/new_space_search/)
# -----------------------------------------------------------------------------
set -euo pipefail

MY="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEFAULT_PY="/root/autodl-tmp/anaconda3/envs/ITHP5090/bin/python"
PY="${PYTHON:-$DEFAULT_PY}"
GPU="${GPU:-0}"
ENQUEUE_K="${ENQUEUE_TOP_K:-10}"

cd "$MY"

R_BASE="$MY/logs/optuna/4090D_restart/new_space_search/regression"
C_BASE="$MY/logs/optuna/4090D_restart/new_space_search/classification"
mkdir -p "$R_BASE"/{db,train_logs,checkpoints,run}
mkdir -p "$C_BASE"/{db,train_logs,checkpoints,run}

enqueue_v2() {
  # $1 = comma-separated URIs or empty
  local uris="$1"
  if [[ -n "${uris}" ]]; then
    printf '%s\n' "--enqueue_top_from" "${uris}" "--enqueue_top_k" "${ENQUEUE_K}"
  fi
}

enqueue_clf() {
  local uris="$1"
  if [[ -n "${uris}" ]]; then
    printf '%s\n' "--enqueue_top_from" "${uris}" "--enqueue_top_k" "${ENQUEUE_K}"
  fi
}

# --- Regression (MOSI / MOSEI / SIMSv2) ---
$PY -u optuna_search_v2.py --dataset mosi --gpu "$GPU" --search_tier 3 --micro_refine none \
  --disable_two_stage_mosi --n_trials 80 \
  --db "sqlite:///${R_BASE}/db/infogate_mosi_new_space_align_floor.db" \
  --study_name infogate_mosi_new_space_align_floor_tpe \
  --artefact_root "$R_BASE" \
  $(enqueue_v2 "${ENQUEUE_TOP_MOSI:-}")

$PY -u optuna_search_v2.py --dataset mosei --gpu "$GPU" --search_tier 3 --micro_refine none \
  --n_trials 60 \
  --db "sqlite:///${R_BASE}/db/infogate_mosei_new_space_align_floor.db" \
  --study_name infogate_mosei_new_space_align_floor_tpe \
  --artefact_root "$R_BASE" \
  $(enqueue_v2 "${ENQUEUE_TOP_MOSEI:-}")

# SIMSv2: no warm-start from old SIMSv2 DBs by default (leave ENQUEUE_TOP_SIMSV2 unset).
$PY -u optuna_search_v2.py --dataset simsv2 --gpu "$GPU" --search_tier 3 --micro_refine none \
  --n_trials 100 \
  --db "sqlite:///${R_BASE}/db/infogate_simsv2_new_space_align_floor.db" \
  --study_name infogate_simsv2_new_space_align_floor_tpe \
  --artefact_root "$R_BASE" \
  $(enqueue_v2 "${ENQUEUE_TOP_SIMSV2:-}")

# --- Classification (MUStARD / UR-FUNNY) ---
$PY -u optuna_search_classify.py --dataset mustard --gpu "$GPU" --search_tier 3 \
  --disable_two_stage --n_trials 60 \
  --db "sqlite:///${C_BASE}/db/infogate_mustard_new_space_align_floor.db" \
  --study_name infogate_mustard_new_space_align_floor_tpe \
  --artefact_root "$C_BASE" \
  $(enqueue_clf "${ENQUEUE_TOP_MUSTARD:-}")

$PY -u optuna_search_classify.py --dataset ur_funny --gpu "$GPU" --search_tier 3 \
  --n_trials 60 \
  --db "sqlite:///${C_BASE}/db/infogate_ur_funny_new_space_align_floor.db" \
  --study_name infogate_ur_funny_new_space_align_floor_tpe \
  --artefact_root "$C_BASE" \
  $(enqueue_clf "${ENQUEUE_TOP_UR_FUNNY:-}")

echo "All new_space_search studies finished (or were started sequentially)."
