#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# BERT-base-uncased: single-stage Tier-3 search for align_mix_floor (0..0.6).
# Uses shared My_creation/optuna_search_v2.py + bert_infogate.py (--pretrained_model).
#
# Layout:
#   bert_optuna/logs/new_space_search/regression/{db,run,train_logs,checkpoints}
#
# Warm-start: seeds gold MOSI/MOSEI hparams from ablation/fixed_experiment modules
# (trial 121/220/234, trial 70). Override with ENQUEUE_TOP_* if old DBs exist.
#
# Usage:
#   export MOSI_GPU=1 MOSEI_GPU=0
#   bash My_creation/bert_optuna/run_bert_new_space_align_floor.sh
#
# Optional:
#   SERIAL=1          # MOSI then MOSEI on one GPU schedule
#   RUN_MOSI=0|RUN_MOSEI=0
#   SEED_ONLY=1       # only enqueue warm-starts, do not launch drivers
# -----------------------------------------------------------------------------
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
SEED_ONLY="${SEED_ONLY:-0}"
ENQUEUE_K="${ENQUEUE_TOP_K:-10}"

if [[ ! -d "$BERT_MODEL" ]]; then
  echo "ERROR: BERT weights not found: $BERT_MODEL" >&2
  echo "  Run: bash $SCRIPT_DIR/download_bert_base_uncased.sh" >&2
  exit 1
fi
BERT_MODEL="$(cd "$(dirname "$BERT_MODEL")" && pwd)/$(basename "$BERT_MODEL")"

R_BASE="$SCRIPT_DIR/logs/new_space_search/regression"
mkdir -p "$R_BASE"/{db,run,train_logs,checkpoints}

MOSI_DB="${R_BASE}/db/infogate_mosi_bert_new_space_align_floor.db"
MOSEI_DB="${R_BASE}/db/infogate_mosei_bert_new_space_align_floor.db"

echo "[$(date -Iseconds)] bert_optuna align_floor"
echo "  MY_CREATION=$MY_CREATION"
echo "  BERT_MODEL=$BERT_MODEL"
echo "  R_BASE=$R_BASE"
echo "  MOSI_GPU=$MOSI_GPU MOSEI_GPU=$MOSEI_GPU SERIAL=$SERIAL"

seed_studies() {
  "$PYTHON" - <<'PY'
from pathlib import Path
import importlib.util
import optuna

root = Path("/root/autodl-tmp/ITHP_MODS_CyIN_Ect/My_creation")
base = root / "bert_optuna/logs/new_space_search/regression/db"

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod

mosi121 = load("mosi121", root / "ablation_study/mosi_space2_trial121_hparams.py")
mosi220 = load("mosi220", root / "ablation_study/mosi_trial220_hparams.py")
mosi234 = load("mosi234", root / "ablation_study/mosi_trial234_hparams.py")
mosei70 = load("mosei70", root / "fixed_experiment/mosei_phase1_trial70_hparams.py")

keep = {
    "n_epochs", "learning_rate", "ig_learning_rate", "beta_ib",
    "num_infogate_layers", "bottleneck_dim", "mse_weight", "dropout_prob",
    "alpha_ib", "stage1_epochs", "warmup_proportion", "weight_decay",
    "ema_decay", "selector_target_temp", "selector_rib_weight",
    "align_mix_floor", "gumbel_tau_start", "gumbel_tau_end", "num_heads",
    "unified_dim", "ema_start_epoch", "seed",
}

def pack(src, batch_config):
    p = {k: v for k, v in dict(src).items() if k in keep}
    p["batch_config"] = int(batch_config)
    p.setdefault("align_mix_floor", 0.3)
    return p

def seed(study_name, db_path, entries):
    storage = f"sqlite:///{db_path}"
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="minimize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=128),
    )
    for label, raw, bc in entries:
        study.enqueue_trial(
            pack(raw, bc),
            user_attrs={"warm_start_source": label},
            skip_if_exists=True,
        )
    study = optuna.load_study(study_name=study_name, storage=storage)
    waiting = sum(t.state == optuna.trial.TrialState.WAITING for t in study.trials)
    print(f"{study_name}: waiting={waiting} db={db_path}")

seed(
    "infogate_mosi_bert_new_space_align_floor_tpe",
    base / "infogate_mosi_bert_new_space_align_floor.db",
    [
        ("phase4_space2_trial121", mosi121.TRIAL_121_PARAMS, 0),
        ("phase4_trial220", mosi220.TRIAL_220_PARAMS, 0),
        ("phase4_trial234", mosi234.TRIAL_234_PARAMS, 0),
    ],
)
seed(
    "infogate_mosei_bert_new_space_align_floor_tpe",
    base / "infogate_mosei_bert_new_space_align_floor.db",
    [("phase1_trial70", mosei70.TRIAL_70_PARAMS, 0)],
)
PY
}

seed_studies

if [[ "$SEED_ONLY" == "1" ]]; then
  echo "[$(date -Iseconds)] SEED_ONLY=1; done."
  exit 0
fi

enqueue_v2() {
  local uris="$1"
  if [[ -n "${uris}" ]]; then
    printf '%s\n' "--enqueue_top_from" "${uris}" "--enqueue_top_k" "${ENQUEUE_K}"
  fi
}

run_mosi() {
  local log="$R_BASE/run/mosi_bert_align_floor.log"
  {
    echo ""
    echo "######## [$(date -Iseconds)] MOSI BERT align_floor (GPU $MOSI_GPU) ########"
  } >> "$log"
  nohup env CUDA_VISIBLE_DEVICES="$MOSI_GPU" "$PYTHON" -u optuna_search_v2.py \
    --dataset mosi \
    --gpu 0 \
    --search_tier 3 \
    --micro_refine none \
    --disable_two_stage_mosi \
    --n_trials 80 \
    --n_startup_trials 20 \
    --selection_metric mae \
    --study_name infogate_mosi_bert_new_space_align_floor_tpe \
    --db "sqlite:///${MOSI_DB}" \
    --artefact_root "$R_BASE" \
    --pretrained_model "$BERT_MODEL" \
    $(enqueue_v2 "${ENQUEUE_TOP_MOSI:-}") \
    >>"$log" 2>&1 &
  MOSI_PID=$!
}

run_mosei() {
  local log="$R_BASE/run/mosei_bert_align_floor.log"
  {
    echo ""
    echo "######## [$(date -Iseconds)] MOSEI BERT align_floor (GPU $MOSEI_GPU) ########"
  } >> "$log"
  nohup env CUDA_VISIBLE_DEVICES="$MOSEI_GPU" "$PYTHON" -u optuna_search_v2.py \
    --dataset mosei \
    --gpu 0 \
    --search_tier 3 \
    --micro_refine none \
    --n_trials 60 \
    --n_startup_trials 15 \
    --selection_metric mae \
    --study_name infogate_mosei_bert_new_space_align_floor_tpe \
    --db "sqlite:///${MOSEI_DB}" \
    --artefact_root "$R_BASE" \
    --pretrained_model "$BERT_MODEL" \
    $(enqueue_v2 "${ENQUEUE_TOP_MOSEI:-}") \
    >>"$log" 2>&1 &
  MOSEI_PID=$!
}

if [[ "$SERIAL" == "1" ]]; then
  if [[ "$RUN_MOSI" == "1" ]]; then
    run_mosi
    echo "[$(date -Iseconds)] MOSI PID=$MOSI_PID  tail -F $R_BASE/run/mosi_bert_align_floor.log"
    wait "$MOSI_PID"
  fi
  if [[ "$RUN_MOSEI" == "1" ]]; then
    run_mosei
    echo "[$(date -Iseconds)] MOSEI PID=$MOSEI_PID  tail -F $R_BASE/run/mosei_bert_align_floor.log"
    wait "$MOSEI_PID"
  fi
else
  WAIT_PIDS=()
  if [[ "$RUN_MOSI" == "1" ]]; then
    run_mosi
    echo "[$(date -Iseconds)] MOSI PID=$MOSI_PID  tail -F $R_BASE/run/mosi_bert_align_floor.log"
    WAIT_PIDS+=("$MOSI_PID")
  fi
  if [[ "$RUN_MOSEI" == "1" ]]; then
    run_mosei
    echo "[$(date -Iseconds)] MOSEI PID=$MOSEI_PID  tail -F $R_BASE/run/mosei_bert_align_floor.log"
    WAIT_PIDS+=("$MOSEI_PID")
  fi
  if ((${#WAIT_PIDS[@]} > 0)); then
    wait "${WAIT_PIDS[@]}"
  fi
fi

echo "[$(date -Iseconds)] bert_optuna align_floor drivers finished."
