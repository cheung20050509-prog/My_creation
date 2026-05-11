#!/usr/bin/env bash
# MOSI trial234: six PRISM --ablation modes, separate runs/*/ dirs.
# Multi-GPU: each physical GPU can run a different number of concurrent trainings;
# wait for the whole batch, then the next chunk (parallel within batch, ordered across batches).
#
# Baseline (--ablation none) certification vs frozen fixed_experiment/runs/mosi_trial234/train.log:
#   Do NOT run none on a GPU shared with another training job in the same wave (e.g. avoid
#   JOBS_PER_GPU>1 stacking none + another mode on one card). For a strict reproduction, run
#   none alone: CUDA_VISIBLE_DEVICES=0 GPU_LIST=0 JOBS_PER_GPU=1 bash … or only the launcher
#   with no other CUDA python on that device.
#
# Env:
#   GPU_LIST       "0,1" — if unset and nvidia-smi reports 2 GPUs, uses GPUs 0 and 1
#   JOBS_PER_GPU   single int (same on every GPU), or per-GPU list matching GPU_LIST length,
#                  e.g. "2,1" → two processes on GPU0, one on GPU1.
#                  Unset with GPUs [0,1]: defaults to 2,1
#   EXCLUDE_GPUS   used only when auto-selecting GPUs on machines with != 2 GPUs (unset → default 1)
set -euo pipefail
MY_CREATION="$(cd "$(dirname "$0")/.." && pwd)"
cd "$MY_CREATION"

PYTHON="${PYTHON:-/root/autodl-tmp/anaconda3/envs/ITHP5090/bin/python}"
if ! "$PYTHON" -c "import torch" 2>/dev/null; then
  PYTHON="${PYTHON:-python3}"
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

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
  # Default when JOBS_PER_GPU unset and GPUs are 0,1: two ablations on GPU0, one on GPU1.
  JOBS_ARR=(2 1)
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
  j=2
  for ((i = 0; i < NUM_GPUS; i++)); do
    JOBS_ARR+=("$j")
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

echo "======== $(date -Is) PRISM ablation batch: GPUs=[${GPU_IDS[*]}] JOBS_PER_GPU=[${JOBS_ARR[*]}] PARALLEL=${PARALLEL} (EXCLUDE_GPUS auto-only: ${EXCLUDE_GPUS:-∅}) ========"
echo "NOTE: Baseline none vs frozen — run none as the sole CUDA job on that GPU (no second training on the same card)."

MODES=(none no_infogate no_mselector no_ib no_conf_gating no_adaptive_gate)
n_modes=${#MODES[@]}

for ((start = 0; start < n_modes; start += PARALLEL)); do
  pids=()
  echo "-------- batch start_index=${start} $(date -Is) --------"
  for ((slot = 0; slot < PARALLEL; slot++)); do
    idx=$((start + slot))
    if ((idx >= n_modes)); then
      break
    fi
    gpu="${SLOT_GPUS[$slot]}"
    m="${MODES[$idx]}"
    if [[ "$m" == "none" ]]; then
      OUT="${MY_CREATION}/ablation_study/runs/mosi_trial234"
    else
      OUT="${MY_CREATION}/ablation_study/runs/mosi_trial234_${m}"
    fi
    mkdir -p "${OUT}/checkpoints"
    : >"${OUT}/train.log"
    echo "  $(date -Is) ablation=${m} CUDA_VISIBLE_DEVICES=${gpu} slot=${slot} OUT=${OUT}"
    CUDA_VISIBLE_DEVICES="${gpu}" nohup "$PYTHON" -u ablation_study/train_fixed_mosi_trial234.py \
      --ablation "$m" \
      --checkpoint-dir "${OUT}/checkpoints" \
      >>"${OUT}/train.log" 2>&1 &
    pid=$!
    pids+=("$pid")
    echo "  Started PID=${pid} (GPU ${gpu}) log=${OUT}/train.log"
  done
  for pid in "${pids[@]}"; do
    wait "${pid}"
    echo "  Finished PID=${pid} at $(date -Is)"
  done
  echo "-------- batch done $(date -Is) --------"
done

echo "======== $(date -Is) all modes finished ========"
