#!/usr/bin/env bash
# One-shot: MOSI + MOSEI + SIMSV2 on physical GPUs 0/1/2 with isolated run dir:
#   logs/<RUN_TAG>/{run,db,train_logs,checkpoints}
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
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="${PYTHON:-/root/autodl-tmp/anaconda3/envs/ITHP5090/bin/python}"
fi

RUN_TAG="${RUN_TAG:-optuna_relaunch_$(date +%Y%m%d_%H%M%S)}"
ROOT="$(pwd)"
RUN_ROOT="$ROOT/logs/$RUN_TAG"
mkdir -p "$RUN_ROOT/run" "$RUN_ROOT/db"
echo "$RUN_TAG" > "$RUN_ROOT/RUN_TAG.txt"

N_TRIALS="${N_TRIALS:-200}"
N_START="${N_START:-55}"
SEARCH_TIER="${SEARCH_TIER:-2}"
STAGE1_TRIALS="${STAGE1_TRIALS:-60}"
STAGE2_TRIALS="${STAGE2_TRIALS:-140}"
STAGE2_TOP_K="${STAGE2_TOP_K:-8}"
GPU_BASE="${GPU_BASE:-0}"

# sqlite absolute URI: "sqlite:///" + "/abs/path" -> sqlite:////abs/path
db_uri () {
  local name="$1"
  echo "sqlite:///${RUN_ROOT}/db/${name}.db"
}

echo "RUN_TAG=$RUN_TAG"
echo "RUN_ROOT=$RUN_ROOT"
echo "PYTHON=$PYTHON"

launch () {
  local dataset="$1"
  local phys_gpu="$2"
  local logfile="$3"
  local study="infogate_${dataset}_${RUN_TAG}"
  local db
  db="$(db_uri "${dataset}")"
  local extra=()
  if [[ "$dataset" == "mosi" ]]; then
    extra+=(--stage1_trials "${STAGE1_TRIALS}" --stage2_trials "${STAGE2_TRIALS}" --stage2_top_k "${STAGE2_TOP_K}")
  fi
  echo "[$(date -Iseconds)] start ${dataset} CUDA_VISIBLE_DEVICES=${phys_gpu} -> ${logfile}" >&2
  nohup env CUDA_VISIBLE_DEVICES="${phys_gpu}" "${PYTHON}" -u optuna_search_v2.py \
    --dataset "${dataset}" \
    --gpu 0 \
    --n_trials "${N_TRIALS}" \
    --n_startup_trials "${N_START}" \
    --search_tier "${SEARCH_TIER}" \
    "${extra[@]}" \
    --study_name "${study}" \
    --db "${db}" \
    > "${logfile}" 2>&1 &
  echo $!
}

p1=$(launch mosi   $((GPU_BASE + 0)) "$RUN_ROOT/run/mosi_gpu0.log")
p2=$(launch mosei  $((GPU_BASE + 1)) "$RUN_ROOT/run/mosei_gpu1.log")
p3=$(launch simsv2 $((GPU_BASE + 2)) "$RUN_ROOT/run/simsv2_gpu2.log")

echo "PIDs: mosi=${p1} mosei=${p2} simsv2=${p3}"
echo "Tail: tail -F $RUN_ROOT/run/*.log"
