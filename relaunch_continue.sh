#!/usr/bin/env bash
# Continue Optuna search where the 2026-04-20 20:59 shutdown left off.
# Usage:  GROUP=mosi_v3 ./relaunch_continue.sh
#         GROUP=mosei_t2 ./relaunch_continue.sh
#         GROUP=mosei_t3 ./relaunch_continue.sh
#         GROUP=simsv2_v3 ./relaunch_continue.sh
#         GROUP=all ./relaunch_continue.sh        # mosei_t3 + simsv2_v3 (multi-GPU layout)
#         GROUP=all3 ./relaunch_continue.sh       # mosei_t3 + mosi_v3 + simsv2_v3, ALL on GPU $GPU_PIN
#
# Env overrides:
#   GPU_BASE=0          physical GPU id; mosei always uses GPU+1, simsv2 uses GPU+0,
#                       mosi_v3 uses GPU+1; with GROUP=all they run in parallel.
#   GPU_PIN=0           used only by GROUP=all3: pin every job to this single physical GPU.
#   N_TRIALS=200        total trials cap (sees existing complete trials in the DB)
#   EXTRA_ARGS=""       any extra flags to forward
#
# GROUP=all3 single-GPU notes:
#   - 4090D / 4090 (24 GB) fits 3 trials concurrently for typical hparams (~17-21 GB),
#     but trials with unified_dim=384 + num_infogate_layers=5 + batch=64 may OOM.
#     If OOM hits one job, kill it (`kill <pid>`) and the others keep running.
#   - All jobs share the same nohup PID line; tail -F all 3 *_resume_*.log to monitor.
#   - Sets PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to reduce fragmentation.
#   - Each job is staggered by 10 s to avoid CUDA init races.
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
[[ -x "$PYTHON" ]] || PYTHON="${PYTHON:-/root/autodl-tmp/anaconda3/envs/ITHP5090/bin/python}"

GROUP="${GROUP:-}"
N_TRIALS="${N_TRIALS:-200}"
GPU_BASE="${GPU_BASE:-0}"
GPU_PIN="${GPU_PIN:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

# For multi-process single-GPU sharing: reduce caching-allocator fragmentation
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

RUN_ROOT="$PWD/logs/optuna_relaunch_20260415_120746"
DB_DIR="$RUN_ROOT/db"
RUN_LOG_DIR="$RUN_ROOT/run"
mkdir -p "$RUN_LOG_DIR"

# Mark interrupted trials as FAIL so TPE doesn't get confused on resume
mark_failed_running () {
  local db="$1"
  "$PYTHON" -c "
import sqlite3
con = sqlite3.connect('$db')
n = con.execute(\"UPDATE trials SET state='FAIL' WHERE state='RUNNING'\").rowcount
con.commit()
print(f'  cleaned {n} stale RUNNING trial(s) in $(basename "$db")')
"
}

run_continue () {
  local group="$1"
  local dataset="$2"
  local phys_gpu="$3"
  local db_uri="$4"
  local study_name="$5"
  local extra=("${!6}")  # nameref to array
  local logfile="$RUN_LOG_DIR/${group}_resume_$(date -u +%Y%m%d_%H%M%S).log"

  echo "================================================================"
  echo "[${group}] dataset=$dataset  CUDA=$phys_gpu"
  echo "  study : $study_name"
  echo "  db    : $db_uri"
  echo "  log   : $logfile"
  echo "  extra : ${extra[*]}"
  echo "================================================================"

  # Sanitize stale RUNNING from the active DB
  local db_path="${db_uri#sqlite:///}"
  if [[ -f "$db_path" ]]; then mark_failed_running "$db_path"; fi

  nohup env CUDA_VISIBLE_DEVICES="${phys_gpu}" "${PYTHON}" -u optuna_search_v2.py \
    --dataset "${dataset}" --gpu 0 \
    --n_trials "${N_TRIALS}" --n_startup_trials 0 \
    --study_name "${study_name}" --db "${db_uri}" \
    "${extra[@]}" $EXTRA_ARGS \
    > "${logfile}" 2>&1 &

  echo "  PID=$!  (tail -F $logfile)"
}

# ---------------- MOSI v3_t3: resume Tier-3 widened search -----------------
launch_mosi_v3 () {
  local extra=(--search_tier 3 --stage_label v3_t3 --disable_two_stage_mosi)
  run_continue mosi_v3 mosi $((GPU_BASE + 1)) \
    "sqlite:///$DB_DIR/mosi_v3.db" \
    "infogate_mosi_v3_t3_widened" extra[@]
}

# ---------------- MOSEI: KEEP TIER 2 (cheaper, more trials) ----------------
launch_mosei_t2 () {
  local extra=(--search_tier 2)
  run_continue mosei_t2 mosei $((GPU_BASE + 1)) \
    "sqlite:///$DB_DIR/mosei.db" \
    "infogate_mosei_optuna_relaunch_20260415_120746" extra[@]
}

# ---------------- MOSEI: ESCALATE TO TIER 3 (RECOMMENDED, NEW STUDY) -------
# Pulls top-10 from existing mosei.db as warm-start seeds.
launch_mosei_t3 () {
  local extra=(--search_tier 3 --stage_label v3_t3
               --enqueue_top_from "sqlite:///$DB_DIR/mosei.db"
               --enqueue_top_k 10)
  # Use a NEW DB so it doesn't pollute the Tier-2 study.
  run_continue mosei_t3 mosei $((GPU_BASE + 1)) \
    "sqlite:///$DB_DIR/mosei_v3.db" \
    "infogate_mosei_v3_t3_widened" extra[@]
}

# ---------------- SIMSV2 v3_t3: resume Tier-3 widened search ----------------
launch_simsv2_v3 () {
  local extra=(--search_tier 3 --stage_label v3_t3
               --enqueue_top_from "sqlite:///$DB_DIR/simsv2.db,sqlite:///$DB_DIR/simsv2_v2.db"
               --enqueue_top_k 5)
  run_continue simsv2_v3 simsv2 $((GPU_BASE + 0)) \
    "sqlite:///$DB_DIR/simsv2_v3.db" \
    "infogate_simsv2_v3_t3_widened" extra[@]
}

# Single-GPU variants: pin every job to GPU_PIN
launch_mosi_v3_pinned () {
  local extra=(--search_tier 3 --stage_label v3_t3 --disable_two_stage_mosi)
  run_continue mosi_v3 mosi "$GPU_PIN" \
    "sqlite:///$DB_DIR/mosi_v3.db" \
    "infogate_mosi_v3_t3_widened" extra[@]
}
launch_mosei_t3_pinned () {
  local extra=(--search_tier 3 --stage_label v3_t3
               --enqueue_top_from "sqlite:///$DB_DIR/mosei.db"
               --enqueue_top_k 10)
  run_continue mosei_t3 mosei "$GPU_PIN" \
    "sqlite:///$DB_DIR/mosei_v3.db" \
    "infogate_mosei_v3_t3_widened" extra[@]
}
launch_simsv2_v3_pinned () {
  local extra=(--search_tier 3 --stage_label v3_t3
               --enqueue_top_from "sqlite:///$DB_DIR/simsv2.db,sqlite:///$DB_DIR/simsv2_v2.db"
               --enqueue_top_k 5)
  run_continue simsv2_v3 simsv2 "$GPU_PIN" \
    "sqlite:///$DB_DIR/simsv2_v3.db" \
    "infogate_simsv2_v3_t3_widened" extra[@]
}

case "$GROUP" in
  mosi_v3)   launch_mosi_v3 ;;
  mosei_t2)  launch_mosei_t2 ;;
  mosei_t3)  launch_mosei_t3 ;;
  simsv2_v3) launch_simsv2_v3 ;;
  all)
    # Multi-GPU layout: skip mosi_v3 (already converged), focus on mosei_t3 + simsv2_v3
    launch_mosei_t3
    sleep 2
    launch_simsv2_v3
    ;;
  all3)
    # SINGLE-GPU layout: 3 jobs share GPU $GPU_PIN
    echo
    echo ">>> GROUP=all3: launching mosei_t3 + mosi_v3 + simsv2_v3 on GPU $GPU_PIN"
    echo ">>> Estimated VRAM: ~17-21 GB on a 24 GB card. Watch with:"
    echo "      watch -n 5 nvidia-smi"
    echo ">>> If OOM hits one job, just kill that PID; the other two keep running."
    echo
    launch_mosei_t3_pinned
    sleep 10
    launch_simsv2_v3_pinned
    sleep 10
    launch_mosi_v3_pinned
    echo
    echo ">>> All 3 launched. Monitor logs:"
    echo "      tail -F $RUN_LOG_DIR/{mosei_t3,mosi_v3,simsv2_v3}_resume_*.log"
    ;;
  all3_2gpu)
    # TWO-GPU layout (preferred when 2x 4090D available):
    #   GPU $GPU_MOSEI = mosei_t3 alone (heaviest per-trial; gets dedicated card)
    #   GPU $GPU_SHARE = mosi_v3 + simsv2_v3 share (mosi is mid, simsv2 is light)
    GPU_MOSEI="${GPU_MOSEI:-0}"
    GPU_SHARE="${GPU_SHARE:-1}"
    echo
    echo ">>> GROUP=all3_2gpu: 2-GPU layout"
    echo "      GPU $GPU_MOSEI  : mosei_t3 (alone)"
    echo "      GPU $GPU_SHARE : mosi_v3 + simsv2_v3 (share)"
    echo ">>> Watch with: watch -n 5 nvidia-smi"
    echo

    # mosei_t3 alone on GPU $GPU_MOSEI
    GPU_PIN="$GPU_MOSEI" launch_mosei_t3_pinned
    sleep 10
    # mosi_v3 + simsv2_v3 share GPU $GPU_SHARE (heavy + light pair)
    GPU_PIN="$GPU_SHARE" launch_mosi_v3_pinned
    sleep 10
    GPU_PIN="$GPU_SHARE" launch_simsv2_v3_pinned
    echo
    echo ">>> All 3 launched. Monitor logs:"
    echo "      tail -F $RUN_LOG_DIR/{mosei_t3,mosi_v3,simsv2_v3}_resume_*.log"
    ;;
  *)
    echo "Usage: GROUP=<mosi_v3|mosei_t2|mosei_t3|simsv2_v3|all|all3|all3_2gpu> $0" >&2
    echo "  all       = mosei_t3 + simsv2_v3 (legacy multi-GPU)" >&2
    echo "  all3      = 3 jobs share single GPU \$GPU_PIN (1-GPU mode)" >&2
    echo "  all3_2gpu = mosei alone on \$GPU_MOSEI; mosi+simsv2 share \$GPU_SHARE (2-GPU mode)" >&2
    exit 1
    ;;
esac
