#!/usr/bin/env bash
# Run Optuna on MOSI, MOSEI, SIMSV2: one physical GPU per dataset (three RTX 5090).
# Default GPU_BASE=0 → uses cards 0,1,2 only; card 3 is NOT used by this script.
# Random warmup: n_startup_trials >= 55 by default (see optuna_search_v2.py).
#
# Usage:
#   ./run_optuna_three_5090.sh
#   N_TRIALS=300 N_START=60 SEARCH_TIER=2 ./run_optuna_three_5090.sh
#
# Single-GPU machine (sequential, same card):
#   PARALLEL=0 GPU_BASE=0 ./run_optuna_three_5090.sh

set -euo pipefail
cd "$(dirname "$0")"

# Default: conda env ITHP5090 (override CONDA_BASE / CONDA_ENV if needed)
CONDA_BASE="${CONDA_BASE:-/root/autodl-tmp/anaconda3}"
CONDA_ENV="${CONDA_ENV:-ITHP5090}"
if [[ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
fi
if [[ -z "${PYTHON:-}" && -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
  PYTHON="${CONDA_PREFIX}/bin/python"
fi
PYTHON="${PYTHON:-python3}"
N_TRIALS="${N_TRIALS:-200}"
N_START="${N_START:-55}"
SEARCH_TIER="${SEARCH_TIER:-2}"
PARALLEL="${PARALLEL:-1}"
GPU_BASE="${GPU_BASE:-0}"

mkdir -p logs/optuna

run_optuna () {
  local dataset=$1
  local gpu=$2
  local log="logs/optuna/run_${dataset}_gpu${gpu}.log"
  echo "[$(date -Iseconds)] starting ${dataset} on CUDA ${gpu} -> ${log}" >&2
  nohup env CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u optuna_search_v2.py \
    --dataset "${dataset}" \
    --gpu 0 \
    --n_trials "${N_TRIALS}" \
    --n_startup_trials "${N_START}" \
    --search_tier "${SEARCH_TIER}" \
    >> "${log}" 2>&1 &
  echo $!
}

# After CUDA_VISIBLE_DEVICES=<k>, the visible device is always index 0 for train.py subprocess.
if [[ "${PARALLEL}" == "1" ]]; then
  p1=$(run_optuna mosi $((GPU_BASE + 0)))
  p2=$(run_optuna mosei $((GPU_BASE + 1)))
  p3=$(run_optuna simsv2 $((GPU_BASE + 2)))
  echo "Launched 3 jobs (PIDs: ${p1} ${p2} ${p3}). Logs under logs/optuna/run_*.log"
else
  for ds in mosi mosei simsv2; do
    env CUDA_VISIBLE_DEVICES="${GPU_BASE}" "${PYTHON}" -u optuna_search_v2.py \
      --dataset "${ds}" \
      --gpu 0 \
      --n_trials "${N_TRIALS}" \
      --n_startup_trials "${N_START}" \
      --search_tier "${SEARCH_TIER}" \
      2>&1 | tee "logs/optuna/run_${ds}_sequential.log"
  done
fi
