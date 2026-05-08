#!/usr/bin/env bash
# Progressive Optuna restart on the 4090D machine.
#
# Layout (chosen by user):
#   GPU0 = MOSI (ONLY=mosi) or SIMSv2 (ONLY=simsv2); BOTH on GPU0 if ONLY=all
#   GPU1 = MOSEI
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
# Classification (UR-FUNNY / MUStARD) uses the same top-level folder::
#   logs/optuna/4090D_restart/classification/<RUN_TAG>/...
#   via ./run_optuna_classify.sh (or run_optuna_ur_funny_v2.sh).
#
# Usage:
#   ./run_optuna_4090d_restart.sh phase1
#   ./run_optuna_4090d_restart.sh phase2     # only after phase1 has converged
#   ./run_optuna_4090d_restart.sh phase3     # only after phase2 has converged
#   ./run_optuna_4090d_restart.sh phase4     # SIMSv2: local search around phase3 trial 148 (no tight hull)
#   ./run_optuna_4090d_restart.sh phase4_mosi  # MOSI micro-local: phase4_mosi trial 8 + phase3 s2 {89,43,75}
#   ./run_optuna_4090d_restart.sh phase5_mosi  # Clean MOSI micro-local DB (same anchors as phase4_mosi)
#   ./run_optuna_4090d_restart.sh phase5_simsv2  # SIMS: merge anchors phase3 #148 + phase4 #0 (needs phase4 db)
#
#   ONLY=mosi  ./run_optuna_4090d_restart.sh phase1   # launch one dataset only
#   ONLY=simsv2 ./run_optuna_4090d_restart.sh phase4
#   ONLY=simsv2 ./run_optuna_4090d_restart.sh phase5_simsv2  # after phase4 SIMS db exists
#   ONLY=mosi ./run_optuna_4090d_restart.sh phase4_mosi  # or phase5_mosi (clean db)
#
# Optional env (defaults in-script): MOSI_N_EPOCHS_CAP, MOSI_EARLY_STOP_PATIENCE;
# SIMSV2_N_EPOCHS_CAP, SIMSV2_EARLY_STOP_PATIENCE (phase4 / phase5_simsv2 only).
# Resume trial budget: MOSI_MICRO_N_TRIALS (phase4_mosi|phase5_mosi), SIMSV2_N_TRIALS (phase4|phase5_simsv2).
# New Optuna study name (same DB file allows multiple studies): set MOSI_STUDY_NAME / SIMSV2_STUDY_NAME,
# or MOSI_STUDY_SUFFIX / SIMSV2_STUDY_SUFFIX (appended to default name) when parameter distributions changed.
# Parallel two drivers: wrap each in `( cd .../My_creation && ONLY=... ./run_optuna_4090d_restart.sh ... ) &`
# so both see the script directory (plain `cd && A & B &` only applies `cd` to the first background job).
#
set -euo pipefail
cd "$(dirname "$0")"

PHASE="${1:?Usage: $0 <phase1|phase2|phase3|phase4|phase4_mosi|phase5_mosi|phase5_simsv2>}"
ONLY="${ONLY:-all}"   # all | mosi | mosei | simsv2

case "$PHASE" in
  phase1) TIER=1 ;;
  phase2) TIER=2 ;;
  phase3) TIER=3 ;;
  phase4) TIER=3 ;;
  phase4_mosi) TIER=3 ;;
  phase5_mosi) TIER=3 ;;
  phase5_simsv2) TIER=3 ;;
  *) echo "ERROR: unknown phase '$PHASE'; expected phase1|phase2|phase3|phase4|phase4_mosi|phase5_mosi|phase5_simsv2" >&2; exit 1 ;;
esac

# MOSI micro-local (phase4_mosi / phase5_mosi): Optuna n_epochs cap + train dev early stop
MOSI_N_EPOCHS_CAP="${MOSI_N_EPOCHS_CAP:-110}"
MOSI_EARLY_STOP_PATIENCE="${MOSI_EARLY_STOP_PATIENCE:-15}"

# SIMSv2 tier-3 local hull (phase4 / phase5_simsv2): same knobs via optuna_search_v2 -> train.py
SIMSV2_N_EPOCHS_CAP="${SIMSV2_N_EPOCHS_CAP:-75}"
SIMSV2_EARLY_STOP_PATIENCE="${SIMSV2_EARLY_STOP_PATIENCE:-15}"

MOSI_STUDY_NAME="${MOSI_STUDY_NAME:-}"
MOSI_STUDY_SUFFIX="${MOSI_STUDY_SUFFIX:-}"
SIMSV2_STUDY_NAME="${SIMSV2_STUDY_NAME:-}"
SIMSV2_STUDY_SUFFIX="${SIMSV2_STUDY_SUFFIX:-}"

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
  phase4)  # tier 3; SIMSv2 micro-local around phase3 trial 148 (see launch_simsv2)
    MOSI_S1_TRIALS=60;  MOSI_S2_TRIALS=140; MOSI_S2_TOP_K=8;  MOSI_S2_NSTART=25
    MOSEI_TRIALS=150;   MOSEI_NSTART=55
    SIMSV2_TRIALS=120;   SIMSV2_NSTART=22
    ;;
  phase4_mosi)  # tier 3; MOSI single-study micro-local (see launch_mosi_micro_local)
    # Target total trials for Optuna resume (load_if_exists); increase to add budget on same DB.
    MOSI_MICRO_TRIALS=120
    MOSI_MICRO_NSTART=15
    MOSEI_TRIALS=150
    MOSEI_NSTART=55
    ;;
  phase5_mosi)  # tier 3; fresh MOSI micro-local dir (same anchor policy as phase4_mosi)
    MOSI_MICRO_TRIALS=60
    MOSI_MICRO_NSTART=22
    MOSEI_TRIALS=150
    MOSEI_NSTART=55
    ;;
  phase5_simsv2)  # tier 3; dual-basin SIMS hull (phase3 #148 + phase4 #0)
    MOSI_S1_TRIALS=60;  MOSI_S2_TRIALS=140; MOSI_S2_TOP_K=8;  MOSI_S2_NSTART=25
    MOSEI_TRIALS=150;   MOSEI_NSTART=55
    SIMSV2_TRIALS=90;   SIMSV2_NSTART=22
    ;;
esac

# Resume / add budget on same DB without editing case values:
#   MOSI_MICRO_N_TRIALS   — overrides MOSI_MICRO_TRIALS for phase4_mosi|phase5_mosi
#   SIMSV2_N_TRIALS       — overrides SIMSV2_TRIALS for phase4|phase5_simsv2
case "$PHASE" in
  phase4)
    SIMSV2_TRIALS="${SIMSV2_N_TRIALS:-$SIMSV2_TRIALS}"
    ;;
  phase4_mosi|phase5_mosi)
    MOSI_MICRO_TRIALS="${MOSI_MICRO_N_TRIALS:-$MOSI_MICRO_TRIALS}"
    ;;
  phase5_simsv2)
    SIMSV2_TRIALS="${SIMSV2_N_TRIALS:-$SIMSV2_TRIALS}"
    ;;
esac

PYTHON="${PYTHON:-/root/autodl-tmp/anaconda3/envs/ITHP5090/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: PYTHON='$PYTHON' is not executable." >&2; exit 1
fi

ROOT="$(pwd)"
# phase4_mosi / phase5_mosi / phase5_simsv2 use their own directories (no collision with phase4 SIMS).
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
  # Warm-start tier>=2 from the previous phase's two-stage MOSI DBs (S1+S2).
  local enqueue_args=()
  case "$PHASE" in
    phase2)
      # Pull warm-starts from tier-1 run (phase1), not from the current phase2 study name.
      # NB: SQLite absolute path needs ``sqlite:///`` + `/abs/path` (4 slashes total).
      enqueue_args=(
        --enqueue_top_from
        "sqlite:///${ROOT}/logs/optuna/4090D_restart/phase1/db/mosi_infogate_mosi_phase1_4090d_s1_random.db,sqlite:///${ROOT}/logs/optuna/4090D_restart/phase1/db/mosi_infogate_mosi_phase1_4090d_s2_local.db"
        --enqueue_top_k 12
      )
      ;;
    phase3)
      enqueue_args=(
        --enqueue_top_from
        "sqlite:///${ROOT}/logs/optuna/4090D_restart/phase2/db/mosi_infogate_mosi_phase2_4090d_s1_random.db,sqlite:///${ROOT}/logs/optuna/4090D_restart/phase2/db/mosi_infogate_mosi_phase2_4090d_s2_local.db"
        --enqueue_top_k 12
      )
      ;;
    phase4)
      enqueue_args=(
        --enqueue_top_from
        "sqlite:///${ROOT}/logs/optuna/4090D_restart/phase3/db/mosi_infogate_mosi_phase3_4090d_s1_random.db,sqlite:///${ROOT}/logs/optuna/4090D_restart/phase3/db/mosi_infogate_mosi_phase3_4090d_s2_local.db"
        --enqueue_top_k 12
      )
      ;;
  esac

  # phase3 uses the refitted MOSI override in apply_dataset_bounds_overrides
  # (derived from phase1 s1+s2 + phase2 s1+s2 top-32). Earlier phases keep the
  # base search space for honest comparability and reproducibility on resume.
  local override_args=(--no_dataset_overrides)
  if [[ "$PHASE" == "phase3" || "$PHASE" == "phase4" ]]; then
    override_args=()
  fi

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
    "${override_args[@]}" \
    --study_name "${study}" \
    --db "${db}" \
    --artefact_root "${PHASE_ROOT}" \
    "${enqueue_args[@]}" \
    >> "${logfile}" 2>&1 &
  echo "    PID=$!"
}

# Single-study TPE (requires --disable_two_stage_mosi). Hull merges phase4_mosi best #8
# with phase3 s2_local {89,43,75}; wider hull (no --local_space_tight) per MAE search plan.
launch_mosi_micro_local () {
  local gpu="$1"
  local logfile="$PHASE_ROOT/run/mosi.log"
  local study_default="infogate_mosi_${PHASE}_4090d${MOSI_STUDY_SUFFIX}"
  local study="${MOSI_STUDY_NAME:-$study_default}"
  local db; db="$(db_uri mosi)"
  local p3_s2_db="sqlite:///${ROOT}/logs/optuna/4090D_restart/phase3/db/mosi_infogate_mosi_phase3_4090d_s2_local.db"
  local p3_s2_study="infogate_mosi_phase3_4090d_s2_local"
  local p4_mosi_db="sqlite:///${ROOT}/logs/optuna/4090D_restart/phase4_mosi/db/mosi.db"
  local p4_mosi_study="infogate_mosi_phase4_mosi_4090d"

  echo "  [$(date -Iseconds)] mosi (${PHASE} micro-local) -> GPU${gpu}  ${logfile}"
  {
    echo ""
    echo "######## [$(date -Iseconds)] driver restart (mosi ${PHASE} micro-local) ########"
  } >> "${logfile}"
  nohup env CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u optuna_search_v2.py \
    --dataset mosi \
    --gpu 0 \
    --disable_two_stage_mosi \
    --search_tier "${TIER}" \
    --n_trials "${MOSI_MICRO_TRIALS}" \
    --n_startup_trials "${MOSI_MICRO_NSTART}" \
    --n_epochs "${MOSI_N_EPOCHS_CAP}" \
    --early_stop_patience "${MOSI_EARLY_STOP_PATIENCE}" \
    --selection_metric mae \
    --study_name "${study}" \
    --db "${db}" \
    --artefact_root "${PHASE_ROOT}" \
    --stage_label "${PHASE}" \
    --local_space_anchor_storage "${p4_mosi_db}" \
    --local_space_anchor_study "${p4_mosi_study}" \
    --local_space_anchor_trials "8" \
    --local_space_anchor_extra "${p3_s2_db}::${p3_s2_study}::89,43,75" \
    --enqueue_trials_storage "${p4_mosi_db}" \
    --enqueue_trials_study "${p4_mosi_study}" \
    --enqueue_trials_numbers "8" \
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
  local study_default="infogate_simsv2_${PHASE}_4090d${SIMSV2_STUDY_SUFFIX}"
  local study="${SIMSV2_STUDY_NAME:-$study_default}"
  local db; db="$(db_uri simsv2)"

  # phase3 uses the narrowed SIMSv2 overrides derived from phase2 top-15
  # (see apply_dataset_bounds_overrides in optuna_search_v2.py). Earlier
  # phases keep the base search space for honest comparability.
  local override_args=(--no_dataset_overrides)
  local enqueue_args=()
  local local_space_args=()
  local artefact_args=()
  local simsv2_train_caps=()
  if [[ "$PHASE" == "phase3" ]]; then
    override_args=()
    # SQLite absolute URI: 3 slashes + absolute path = 4 slashes total.
    enqueue_args=(
      --enqueue_top_from
      "sqlite:///${ROOT}/logs/optuna/4090D_restart/phase2/db/simsv2.db"
      --enqueue_top_k 12
    )
  elif [[ "$PHASE" == "phase4" ]]; then
    simsv2_train_caps=(
      --n_epochs "${SIMSV2_N_EPOCHS_CAP}"
      --early_stop_patience "${SIMSV2_EARLY_STOP_PATIENCE}"
    )
    override_args=()
    artefact_args=(--artefact_root "${PHASE_ROOT}")
    local p3_db="sqlite:///${ROOT}/logs/optuna/4090D_restart/phase3/db/simsv2.db"
    local p3_study="infogate_simsv2_phase3_4090d"
    local_space_args=(
      --local_space_anchor_storage "${p3_db}"
      --local_space_anchor_study "${p3_study}"
      --local_space_anchor_trials "148"
      --enqueue_trials_storage "${p3_db}"
      --enqueue_trials_study "${p3_study}"
      --enqueue_trials_numbers "148"
    )
  elif [[ "$PHASE" == "phase5_simsv2" ]]; then
    simsv2_train_caps=(
      --n_epochs "${SIMSV2_N_EPOCHS_CAP}"
      --early_stop_patience "${SIMSV2_EARLY_STOP_PATIENCE}"
    )
    override_args=()
    artefact_args=(--artefact_root "${PHASE_ROOT}")
    local p3_db="sqlite:///${ROOT}/logs/optuna/4090D_restart/phase3/db/simsv2.db"
    local p3_study="infogate_simsv2_phase3_4090d"
    local p4_db="sqlite:///${ROOT}/logs/optuna/4090D_restart/phase4/db/simsv2.db"
    local p4_study="infogate_simsv2_phase4_4090d"
    local p4_path="${ROOT}/logs/optuna/4090D_restart/phase4/db/simsv2.db"
    if [[ ! -f "${p4_path}" ]]; then
      echo "ERROR: phase5_simsv2 requires existing ${p4_path} (run phase4 SIMSv2 first)." >&2
      exit 1
    fi
    local_space_args=(
      --local_space_anchor_storage "${p3_db}"
      --local_space_anchor_study "${p3_study}"
      --local_space_anchor_trials "148"
      --local_space_anchor_extra "${p4_db}::${p4_study}::0"
      --enqueue_trials_storage "${p3_db}"
      --enqueue_trials_study "${p3_study}"
      --enqueue_trials_numbers "148"
    )
  fi

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
    "${override_args[@]}" \
    --study_name "${study}" \
    --db "${db}" \
    --stage_label "${PHASE}" \
    "${artefact_args[@]}" \
    "${enqueue_args[@]}" \
    "${local_space_args[@]}" \
    "${simsv2_train_caps[@]}" \
    >> "${logfile}" 2>&1 &
  echo "    PID=$!"
}

# GPU layout: MOSI GPU0; MOSEI GPU1; SIMSv2 GPU0 (use ONLY= to avoid MOSI+SIMSv2 on same card).
# phase4_mosi: physical GPU via MOSI_GPU (default 0).
# SIMSv2: SIMS_GPU (default 0). Set SIMS_GPU=1 when a second GPU is available for true parallelism.
MOSI_GPU="${MOSI_GPU:-0}"
SIMS_GPU="${SIMS_GPU:-0}"
if [[ "$PHASE" == "phase4_mosi" || "$PHASE" == "phase5_mosi" ]]; then
  if [[ "$ONLY" != "mosi" && "$ONLY" != "all" ]]; then
    echo "ERROR: ${PHASE} only launches MOSI micro-local; use ONLY=mosi or ONLY=all." >&2
    exit 1
  fi
  launch_mosi_micro_local "${MOSI_GPU}"
elif [[ "$PHASE" == "phase5_simsv2" ]]; then
  if [[ "$ONLY" != "simsv2" && "$ONLY" != "all" ]]; then
    echo "ERROR: phase5_simsv2 only launches SIMSv2 dual-anchor search; use ONLY=simsv2 or ONLY=all." >&2
    exit 1
  fi
  launch_simsv2 "${SIMS_GPU}"
else
  if [[ "$ONLY" == "all" || "$ONLY" == "mosi"   ]]; then launch_mosi   0; fi
  if [[ "$ONLY" == "all" || "$ONLY" == "mosei"  ]]; then launch_mosei  1; fi
  if [[ "$ONLY" == "all" || "$ONLY" == "simsv2" ]]; then launch_simsv2 "${SIMS_GPU}"; fi
fi

echo
echo "[$(date -Iseconds)] all driver(s) launched for $PHASE"
echo "  monitor: tail -F $PHASE_ROOT/run/*.log"
echo "  trials : ls -la $PHASE_ROOT/train_logs/"
