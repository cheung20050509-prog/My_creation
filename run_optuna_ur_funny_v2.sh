#!/usr/bin/env bash
# UR-FUNNY v2 — NEW study with tightened categorical space and extended
# continuous bounds derived from the top-15 (of 66 COMPLETE) on the v1 study
# `infogate_ur_funny_optuna_classify_ur_funny_20260422_104708` (best Acc 0.7586).
#
# Overrides live in optuna_search_classify.py :: DATASET_*_OVERRIDES["ur_funny"].
# See run_optuna_classify.sh for the base launcher; this wrapper hard-codes:
#   - ONLY=ur_funny                 (never launch MUSTARD from this script)
#   - RUN_TAG=optuna_classify_ur_funny_v2_<TS>
#   - SEARCH_TIER=2                 (match v1 tier to keep comparisons honest)
#   - URFUNNY_TRIALS=80, N_STARTUP=20 (same budget shape as v1)
#   - GPU_URFUNNY=0                 (UR-FUNNY slot on GPU0)
#
# The driver log is appended so a resume never clobbers the prior run history.
#
# Usage:
#   ./run_optuna_ur_funny_v2.sh
#   URFUNNY_TRIALS=120 ./run_optuna_ur_funny_v2.sh   # bigger budget if desired
#
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-/root/autodl-tmp/anaconda3/envs/ITHP5090/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: PYTHON='$PYTHON' is not executable." >&2; exit 1
fi

SEARCH_TIER="${SEARCH_TIER:-2}"
N_STARTUP="${N_STARTUP:-20}"
URFUNNY_TRIALS="${URFUNNY_TRIALS:-80}"
SELECTION_METRIC="${SELECTION_METRIC:-binary_acc}"
GPU_URFUNNY="${GPU_URFUNNY:-0}"

RUN_TAG="${RUN_TAG:-optuna_classify_ur_funny_v2_$(date +%Y%m%d_%H%M%S)}"
ROOT="$(pwd)"
RUN_ROOT="$ROOT/logs/$RUN_TAG"
mkdir -p "$RUN_ROOT/run" "$RUN_ROOT/db" "$RUN_ROOT/train_logs" "$RUN_ROOT/checkpoints"
echo "$RUN_TAG" > "$RUN_ROOT/RUN_TAG.txt"

STUDY="infogate_ur_funny_${RUN_TAG}"
DB="sqlite:///${RUN_ROOT}/db/ur_funny.db"
LOG="$RUN_ROOT/run/ur_funny_gpu${GPU_URFUNNY}.log"

echo "RUN_TAG=$RUN_TAG"
echo "RUN_ROOT=$RUN_ROOT"
echo "STUDY=$STUDY"
echo "DB=$DB"
echo "GPU=$GPU_URFUNNY"
echo "TIER=$SEARCH_TIER  N_STARTUP=$N_STARTUP  TRIALS=$URFUNNY_TRIALS"

# Append launch marker so resumes remain auditable.
{
  echo ""
  echo "######## [$(date -Iseconds)] ur_funny v2 launch ########"
  echo "  study=$STUDY  gpu=$GPU_URFUNNY  tier=$SEARCH_TIER  n_start=$N_STARTUP  trials=$URFUNNY_TRIALS"
} >> "$LOG"

nohup env CUDA_VISIBLE_DEVICES="${GPU_URFUNNY}" "${PYTHON}" -u optuna_search_classify.py \
  --dataset ur_funny \
  --gpu 0 \
  --search_tier "${SEARCH_TIER}" \
  --n_startup_trials "${N_STARTUP}" \
  --n_trials "${URFUNNY_TRIALS}" \
  --selection_metric "${SELECTION_METRIC}" \
  --study_name "${STUDY}" \
  --db "${DB}" \
  >> "${LOG}" 2>&1 &

PID=$!
echo "PID=$PID"
echo "tail -F $LOG"
