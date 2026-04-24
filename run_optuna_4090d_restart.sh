#!/usr/bin/env bash
# Progressive Optuna restart on the 4090D machine.
#
# Layout (chosen by user):
#   GPU0 = MOSI  (alongside the running UR-FUNNY classify search)
#   GPU1 = MOSEI + SIMSv2
#
# Search policy (chosen by user):
#   * Cold start: --no_dataset_overrides (do NOT inherit prior-GPU bound narrowing)
#                 + no --enqueue_top_from
#   * Progressive: tier 1  -> tier 2 -> tier 3 (3 independent studies per dataset)
#   * MOSI two-stage enabled (random anchors -> TPE-local at the SAME tier)
#   * n_startup tuned per phase (random warmup before TPE kicks in)
#
# Outputs (per phase):
#   logs/optuna/4090D_restart/<phase>/db/{mosi,mosei,simsv2}.db
#   logs/optuna/4090D_restart/<phase>/run/{mosi,mosei,simsv2}.log    # driver stdout
#   logs/optuna/4090D_restart/<phase>/train_logs/                    # train.py per trial
#   logs/optuna/4090D_restart/<phase>/checkpoints/                   # train.py per trial
#
# Usage:
#   ./run_optuna_4090d_restart.sh phase1
#   ./run_optuna_4090d_restart.sh phase2     # only after phase1 has converged
#   ./run_optuna_4090d_restart.sh phase3     # only after phase2 has converged
#
#   ONLY=mosi  ./run_optuna_4090d_restart.sh phase1   # launch one dataset only
#
set -euo pipefail
cd "$(dirname "$0")"

PHASE="${1:?Usage: $0 <phase1|phase2|phase3>}"
ONLY="${ONLY:-all}"   # all | mosi | mosei | simsv2

case "$PHASE" in
  phase1) TIER=1 ;;
  phase2) TIER=2 ;;
  phase3) TIER=3 ;;
  *) echo "ERROR: unknown phase '$PHASE'; expected phase1|phase2|phase3" >&2; exit 1 ;;
esac

# Per-phase trial budgets
case "$PHASE" in
  phase1)  # tier 1: 10 params, lower-D, faster convergence
    MOSI_S1_TRIALS=40;  MOSI_S2_TRIALS=60;  MOSI_S2_TOP_K=8;  MOSI_S2_NSTART=20
    MOSEI_TRIALS=80;    MOSEI_NSTART=55
    SIMSV2_TRIALS=80;   SIMSV2_NSTART=55
    ;;
  phase2)  # tier 1+2: 15 params, medium-D
    MOSI_S1_TRIALS=50;  MOSI_S2_TRIALS=80;  MOSI_S2_TOP_K=8;  MOSI_S2_NSTART=20
    MOSEI_TRIALS=100;   MOSEI_NSTART=55
    SIMSV2_TRIALS=100;  SIMSV2_NSTART=55
    ;;
  phase3)  # tier 1+2+3: 22 params, full search
    MOSI_S1_TRIALS=60;  MOSI_S2_TRIALS=140; MOSI_S2_TOP_K=8;  MOSI_S2_NSTART=25
    MOSEI_TRIALS=150;   MOSEI_NSTART=55
    SIMSV2_TRIALS=150;  SIMSV2_NSTART=55
    ;;
esac

PYTHON="${PYTHON:-/root/autodl-tmp/anaconda3/envs/ITHP5090/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: PYTHON='$PYTHON' is not executable." >&2; exit 1
fi

ROOT="$(pwd)"
PHASE_ROOT="$ROOT/logs/optuna/4090D_restart/$PHASE"
mkdir -p "$PHASE_ROOT/db" "$PHASE_ROOT/run" "$PHASE_ROOT/train_logs" "$PHASE_ROOT/checkpoints"

echo "[$(date -Iseconds)] launching $PHASE (tier=$TIER, ONLY=$ONLY)"
echo "  PHASE_ROOT=$PHASE_ROOT"
echo "  PYTHON=$PYTHON"

db_uri () { echo "sqlite:///${PHASE_ROOT}/db/${1}.db"; }

launch_mosi () {
  local gpu="$1"
  local logfile="$PHASE_ROOT/run/mosi.log"
  local study="infogate_mosi_${PHASE}_4090d"
  local db; db="$(db_uri mosi)"
  echo "  [$(date -Iseconds)] mosi   -> GPU${gpu}  ${logfile}"
  # NB: use `>>` (append) so that a resume does not clobber the prior driver
  # summary; the DB is the source of truth but keeping a continuous driver
  # log across restarts is convenient for auditing.
  {
    echo ""
    echo "######## [$(date -Iseconds)] driver restart (mosi) ########"
  } >> "${logfile}"
  nohup env CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u optuna_search_v2.py \
    --dataset mosi \
    --gpu 0 \
    --search_tier "${TIER}" \
    --n_trials $((MOSI_S1_TRIALS + MOSI_S2_TRIALS)) \
    --n_startup_trials "${MOSI_S2_NSTART}" \
    --stage1_trials "${MOSI_S1_TRIALS}" \
    --stage2_trials "${MOSI_S2_TRIALS}" \
    --stage2_top_k "${MOSI_S2_TOP_K}" \
    --selection_metric mae \
    --no_dataset_overrides \
    --study_name "${study}" \
    --db "${db}" \
    >> "${logfile}" 2>&1 &
  echo "    PID=$!"
}

launch_mosei () {
  local gpu="$1"
  local logfile="$PHASE_ROOT/run/mosei.log"
  local study="infogate_mosei_${PHASE}_4090d"
  local db; db="$(db_uri mosei)"
  echo "  [$(date -Iseconds)] mosei  -> GPU${gpu}  ${logfile}"
  {
    echo ""
    echo "######## [$(date -Iseconds)] driver restart (mosei) ########"
  } >> "${logfile}"
  nohup env CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u optuna_search_v2.py \
    --dataset mosei \
    --gpu 0 \
    --search_tier "${TIER}" \
    --n_trials "${MOSEI_TRIALS}" \
    --n_startup_trials "${MOSEI_NSTART}" \
    --selection_metric mae \
    --no_dataset_overrides \
    --study_name "${study}" \
    --db "${db}" \
    --stage_label "${PHASE}" \
    >> "${logfile}" 2>&1 &
  echo "    PID=$!"
}

launch_simsv2 () {
  local gpu="$1"
  local logfile="$PHASE_ROOT/run/simsv2.log"
  local study="infogate_simsv2_${PHASE}_4090d"
  local db; db="$(db_uri simsv2)"
  echo "  [$(date -Iseconds)] simsv2 -> GPU${gpu}  ${logfile}"
  {
    echo ""
    echo "######## [$(date -Iseconds)] driver restart (simsv2) ########"
  } >> "${logfile}"
  nohup env CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u optuna_search_v2.py \
    --dataset simsv2 \
    --gpu 0 \
    --search_tier "${TIER}" \
    --n_trials "${SIMSV2_TRIALS}" \
    --n_startup_trials "${SIMSV2_NSTART}" \
    --selection_metric mae \
    --no_dataset_overrides \
    --study_name "${study}" \
    --db "${db}" \
    --stage_label "${PHASE}" \
    >> "${logfile}" 2>&1 &
  echo "    PID=$!"
}

# GPU layout: MOSI on GPU0 (alongside UR-FUNNY); MOSEI + SIMSv2 on GPU1
if [[ "$ONLY" == "all" || "$ONLY" == "mosi"   ]]; then launch_mosi   0; fi
if [[ "$ONLY" == "all" || "$ONLY" == "mosei"  ]]; then launch_mosei  1; fi
if [[ "$ONLY" == "all" || "$ONLY" == "simsv2" ]]; then launch_simsv2 1; fi

echo
echo "[$(date -Iseconds)] all driver(s) launched for $PHASE"
echo "  monitor: tail -F $PHASE_ROOT/run/*.log"
echo "  trials : ls -la $PHASE_ROOT/train_logs/"
