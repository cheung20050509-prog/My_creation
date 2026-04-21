#!/usr/bin/env bash
# Launch Optuna search for InfoGate binary classification:
#   - MUSTARD on GPU $GPU_MUSTARD (two-stage random -> TPE-local)
#   - UR-FUNNY on GPU $GPU_URFUNNY (single-stage TPE)
# Each run gets its own artefact dir:
#   logs/<RUN_TAG>/{run,db,train_logs,checkpoints}
#
# Usage:
#   ./run_optuna_classify.sh
#   GPU_MUSTARD=1 GPU_URFUNNY=2 ./run_optuna_classify.sh
#   PARALLEL=0 GPU_MUSTARD=0 GPU_URFUNNY=0 ./run_optuna_classify.sh
#   PYTHON=/path/to/python ONLY=mustard ./run_optuna_classify.sh

set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-/home/anaconda/envs/ITHP/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="${PYTHON:-python3}"
fi

GPU_MUSTARD="${GPU_MUSTARD:-1}"
GPU_URFUNNY="${GPU_URFUNNY:-2}"
PARALLEL="${PARALLEL:-1}"
ONLY="${ONLY:-both}"     # both | mustard | ur_funny

# Knobs (override via env)
SEARCH_TIER="${SEARCH_TIER:-2}"
N_STARTUP="${N_STARTUP:-20}"
URFUNNY_TRIALS="${URFUNNY_TRIALS:-80}"
MUSTARD_S1_TRIALS="${MUSTARD_S1_TRIALS:-40}"
MUSTARD_S2_TRIALS="${MUSTARD_S2_TRIALS:-80}"
MUSTARD_S2_TOP_K="${MUSTARD_S2_TOP_K:-8}"
SELECTION_METRIC="${SELECTION_METRIC:-binary_acc}"

RUN_TAG="${RUN_TAG:-optuna_classify_$(date +%Y%m%d_%H%M%S)}"
ROOT="$(pwd)"
RUN_ROOT="$ROOT/logs/$RUN_TAG"
mkdir -p "$RUN_ROOT/run" "$RUN_ROOT/db" "$RUN_ROOT/train_logs" "$RUN_ROOT/checkpoints"
echo "$RUN_TAG" > "$RUN_ROOT/RUN_TAG.txt"

echo "RUN_TAG=$RUN_TAG"
echo "RUN_ROOT=$RUN_ROOT"
echo "PYTHON=$PYTHON"

db_uri () {
  local name="$1"
  echo "sqlite:///${RUN_ROOT}/db/${name}.db"
}

launch_mustard () {
  local gpu="$1"
  local logfile="$RUN_ROOT/run/mustard_gpu${gpu}.log"
  local study="infogate_mustard_${RUN_TAG}"
  local db
  db="$(db_uri mustard)"
  echo "[$(date -Iseconds)] start mustard CUDA_VISIBLE_DEVICES=${gpu} -> ${logfile}" >&2
  nohup env CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u optuna_search_classify.py \
    --dataset mustard \
    --gpu 0 \
    --search_tier "${SEARCH_TIER}" \
    --n_startup_trials "${N_STARTUP}" \
    --stage1_trials "${MUSTARD_S1_TRIALS}" \
    --stage2_trials "${MUSTARD_S2_TRIALS}" \
    --stage2_top_k "${MUSTARD_S2_TOP_K}" \
    --selection_metric "${SELECTION_METRIC}" \
    --study_name "${study}" \
    --db "${db}" \
    > "${logfile}" 2>&1 &
  echo $!
}

launch_ur_funny () {
  local gpu="$1"
  local logfile="$RUN_ROOT/run/ur_funny_gpu${gpu}.log"
  local study="infogate_ur_funny_${RUN_TAG}"
  local db
  db="$(db_uri ur_funny)"
  echo "[$(date -Iseconds)] start ur_funny CUDA_VISIBLE_DEVICES=${gpu} -> ${logfile}" >&2
  nohup env CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u optuna_search_classify.py \
    --dataset ur_funny \
    --gpu 0 \
    --search_tier "${SEARCH_TIER}" \
    --n_startup_trials "${N_STARTUP}" \
    --n_trials "${URFUNNY_TRIALS}" \
    --selection_metric "${SELECTION_METRIC}" \
    --study_name "${study}" \
    --db "${db}" \
    > "${logfile}" 2>&1 &
  echo $!
}

PIDS=()
if [[ "${PARALLEL}" == "1" ]]; then
  if [[ "${ONLY}" == "both" || "${ONLY}" == "mustard" ]]; then
    p=$(launch_mustard "${GPU_MUSTARD}");  PIDS+=("mustard:${p}")
  fi
  if [[ "${ONLY}" == "both" || "${ONLY}" == "ur_funny" ]]; then
    p=$(launch_ur_funny "${GPU_URFUNNY}"); PIDS+=("ur_funny:${p}")
  fi
  echo "PIDs: ${PIDS[*]}"
  echo "Tail: tail -F ${RUN_ROOT}/run/*.log"
else
  if [[ "${ONLY}" == "both" || "${ONLY}" == "mustard" ]]; then
    env CUDA_VISIBLE_DEVICES="${GPU_MUSTARD}" "${PYTHON}" -u optuna_search_classify.py \
      --dataset mustard --gpu 0 \
      --search_tier "${SEARCH_TIER}" \
      --n_startup_trials "${N_STARTUP}" \
      --stage1_trials "${MUSTARD_S1_TRIALS}" \
      --stage2_trials "${MUSTARD_S2_TRIALS}" \
      --stage2_top_k "${MUSTARD_S2_TOP_K}" \
      --selection_metric "${SELECTION_METRIC}" \
      --study_name "infogate_mustard_${RUN_TAG}" \
      --db "$(db_uri mustard)" \
      2>&1 | tee "$RUN_ROOT/run/mustard_sequential.log"
  fi
  if [[ "${ONLY}" == "both" || "${ONLY}" == "ur_funny" ]]; then
    env CUDA_VISIBLE_DEVICES="${GPU_URFUNNY}" "${PYTHON}" -u optuna_search_classify.py \
      --dataset ur_funny --gpu 0 \
      --search_tier "${SEARCH_TIER}" \
      --n_startup_trials "${N_STARTUP}" \
      --n_trials "${URFUNNY_TRIALS}" \
      --selection_metric "${SELECTION_METRIC}" \
      --study_name "infogate_ur_funny_${RUN_TAG}" \
      --db "$(db_uri ur_funny)" \
      2>&1 | tee "$RUN_ROOT/run/ur_funny_sequential.log"
  fi
fi
