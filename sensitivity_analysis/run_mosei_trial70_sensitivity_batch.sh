#!/usr/bin/env bash
# MOSEI trial70: sweep one hyper-parameter axis (default grid), one train job per GPU slot, waves serial.
#
# Usage:
#   bash sensitivity_analysis/run_mosei_trial70_sensitivity_batch.sh beta_ib
#   bash sensitivity_analysis/run_mosei_trial70_sensitivity_batch.sh selector_rib_weight
#   RUNS_ROOT=/path/to/runs AXIS=mse_weight bash sensitivity_analysis/run_mosei_trial70_sensitivity_batch.sh
#
# Env:
#   PYTHON       default: ITHP5090 conda python (same as fixed_experiment/*.sh); override if needed
#   GPU_LIST     e.g. "0,1" — unset on 2-GPU machine defaults to 0,1
#   EXCLUDE_GPUS default 1 on multi-GPU; to use only GPU 1 set GPU_LIST=1 and EXCLUDE_GPUS= (empty)
#   JOBS_PER_GPU unset → 1 job per listed GPU (same semantics as PRISM batch scripts)
#   RUNS_ROOT    default: My_creation/runs/sensitivity_mosei/trial70
#   AXIS         default: first CLI arg, else beta_ib
set -euo pipefail
MY_CREATION="$(cd "$(dirname "$0")/.." && pwd)"
cd "$MY_CREATION"

_DEFAULT_PY="/root/autodl-tmp/anaconda3/envs/ITHP5090/bin/python"
if [[ -x "${_DEFAULT_PY}" ]]; then
  PYTHON="${PYTHON:-${_DEFAULT_PY}}"
else
  PYTHON="${PYTHON:-python3}"
fi
unset _DEFAULT_PY
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

AXIS="${AXIS:-${1:-beta_ib}}"
RUNS_ROOT="${RUNS_ROOT:-${MY_CREATION}/runs/sensitivity_mosei/trial70}"

if [[ ! "${EXCLUDE_GPUS+x}" ]]; then
  EXCLUDE_GPUS=1
fi

is_excluded() {
  local id="$1"
  [[ -z "${EXCLUDE_GPUS// /}" ]] && return 1
  local tok
  IFS=',' read -ra _ex <<< "$(echo "$EXCLUDE_GPUS" | tr -d '[:space:]')"
  for tok in "${_ex[@]}"; do
    [[ "$tok" == "$id" ]] && return 0
  done
  return 1
}

ng=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d '[:space:]')
if [[ -z "$ng" || "$ng" -lt 1 ]]; then
  ng=1
fi

GPU_LIST="${GPU_LIST:-}"
if [[ -n "$GPU_LIST" ]]; then
  IFS=',' read -ra GPU_IDS <<< "$(echo "$GPU_LIST" | tr -d '[:space:]')"
else
  GPU_IDS=()
  if [[ "$ng" -eq 2 ]]; then
    GPU_IDS=(0 1)
  else
    for ((i = 0; i < ng; i++)); do
      if is_excluded "$i"; then
        continue
      fi
      GPU_IDS+=("$i")
    done
    if [[ ${#GPU_IDS[@]} -eq 0 ]]; then
      echo "ERROR: no GPUs left after EXCLUDE_GPUS=${EXCLUDE_GPUS:-∅}" >&2
      exit 1
    fi
  fi
fi

NUM_GPUS=${#GPU_IDS[@]}

validate_int() {
  [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

JOBS_ARR=()
if [[ "${JOBS_PER_GPU+x}" && "$JOBS_PER_GPU" == *","* ]]; then
  IFS=',' read -ra JOBS_ARR <<< "$(echo "$JOBS_PER_GPU" | tr -d '[:space:]')"
  if [[ ${#JOBS_ARR[@]} -ne $NUM_GPUS ]]; then
    echo "ERROR: JOBS_PER_GPU has ${#JOBS_ARR[@]} entries but GPU_LIST has ${NUM_GPUS} GPUs" >&2
    exit 1
  fi
  for j in "${JOBS_ARR[@]}"; do
    if ! validate_int "$j"; then
      echo "ERROR: invalid JOBS_PER_GPU component: ${j}" >&2
      exit 1
    fi
  done
elif [[ ! "${JOBS_PER_GPU+x}" ]] && [[ "$NUM_GPUS" -eq 2 ]] && [[ "${GPU_IDS[0]}" == "0" ]] && [[ "${GPU_IDS[1]}" == "1" ]]; then
  JOBS_ARR=(1 1)
elif [[ "${JOBS_PER_GPU+x}" ]]; then
  j="${JOBS_PER_GPU}"
  if ! validate_int "$j"; then
    echo "ERROR: JOBS_PER_GPU must be a positive integer or comma list, got ${j}" >&2
    exit 1
  fi
  for ((i = 0; i < NUM_GPUS; i++)); do
    JOBS_ARR+=("$j")
  done
else
  for ((i = 0; i < NUM_GPUS; i++)); do
    JOBS_ARR+=(1)
  done
fi

SLOT_GPUS=()
for ((gi = 0; gi < NUM_GPUS; gi++)); do
  g="${GPU_IDS[$gi]}"
  nj="${JOBS_ARR[$gi]}"
  for ((k = 0; k < nj; k++)); do
    SLOT_GPUS+=("$g")
  done
done
PARALLEL=${#SLOT_GPUS[@]}

mapfile -t VALS < <("$PYTHON" sensitivity_analysis/run_mosei_trial70_sensitivity.py list-values --axis "$AXIS")
n_vals=${#VALS[@]}

echo "======== $(date -Is) MOSEI trial70 sensitivity axis=${AXIS} RUNS_ROOT=${RUNS_ROOT} GPUs=[${GPU_IDS[*]}] PARALLEL=${PARALLEL} n_vals=${n_vals} ========"

for ((start = 0; start < n_vals; start += PARALLEL)); do
  pids=()
  echo "-------- wave start_index=${start} $(date -Is) --------"
  for ((slot = 0; slot < PARALLEL; slot++)); do
    idx=$((start + slot))
    if ((idx >= n_vals)); then
      break
    fi
    gpu="${SLOT_GPUS[$slot]}"
    v="${VALS[$idx]}"
    echo "  $(date -Is) axis=${AXIS} value=${v} CUDA_VISIBLE_DEVICES=${gpu}"
    CUDA_VISIBLE_DEVICES="${gpu}" "$PYTHON" -u sensitivity_analysis/run_mosei_trial70_sensitivity.py train \
      --axis "$AXIS" \
      --value "$v" \
      --runs-root "$RUNS_ROOT" &
    pid=$!
    pids+=("$pid")
    echo "  Started PID=${pid} (GPU ${gpu})"
  done
  for pid in "${pids[@]}"; do
    wait "${pid}"
    echo "  Finished PID=${pid} at $(date -Is)"
  done
  echo "-------- wave done $(date -Is) --------"
done

echo "======== $(date -Is) axis=${AXIS} finished. Aggregate with: ========"
echo "$PYTHON sensitivity_analysis/run_mosei_trial70_sensitivity.py aggregate --runs-root \"$RUNS_ROOT\" \\"
echo "  --summary-out \"${MY_CREATION}/sensitivity_analysis/results/mosei_trial70/summary.csv\""
