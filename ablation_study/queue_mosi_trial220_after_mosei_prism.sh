#!/usr/bin/env bash
# Wait until the MOSEI trial70 PRISM ablation master exits, then run MOSI trial220 PRISM ablations.
#
# Semantics: MOSEI batch bash runs all waves (six modes in batches); only when it exits have all MOSEI
# ablations finished. Then this script runs MOSI trial220's six PRISM modes (same names as MOSEI).
# Blocks until MOSEI completes — may take many hours — then runs MOSI (also batched per MOSI script).
#
# Typical launch (after MOSEI was started with PID saved):
#   cd My_creation/ablation_study
#   nohup bash queue_mosi_trial220_after_mosei_prism.sh >> runs/mosi_trial220_prism_after_mosei.log 2>&1 &
#   echo $! > runs/mosi_trial220_prism_after_mosei.pid
#
# Env:
#   MOSEI_MASTER_PID           bash PID running run_prism_ablations_mosei_phase1_trial70.sh (overrides pid file)
#   MOSEI_PRISM_MASTER_PID_FILE  path to file containing that PID (default: runs/mosei_phase1_trial70_prism_master.pid)
#   MOSI_QUEUE_LOG             append wrapper + child stdout/stderr here (default: runs/mosi_trial220_prism_after_mosei.log)
#   POLL_SEC                   sleep between kill -0 checks (default: 30)
#   SKIP_MOSEI_WAIT            if set to 1, do not wait — run MOSI batch immediately (debug only)
#   GPU_LIST, JOBS_PER_GPU     forwarded to run_prism_ablations_mosi_trial220.sh (default JOBS_PER_GPU=1,1 if unset)
set -euo pipefail

MY_CREATION="$(cd "$(dirname "$0")/.." && pwd)"
cd "$MY_CREATION"

echo $$ >"${MY_CREATION}/ablation_study/runs/mosi_trial220_prism_after_mosei.pid"

PID_FILE="${MOSEI_PRISM_MASTER_PID_FILE:-${MY_CREATION}/ablation_study/runs/mosei_phase1_trial70_prism_master.pid}"
QUEUE_LOG="${MOSI_QUEUE_LOG:-${MY_CREATION}/ablation_study/runs/mosi_trial220_prism_after_mosei.log}"
POLL_SEC="${POLL_SEC:-30}"

mkdir -p "$(dirname "$QUEUE_LOG")"

log() {
  printf '%s\n' "$*" >>"$QUEUE_LOG"
}

wait_pid=""
if [[ "${SKIP_MOSEI_WAIT:-0}" == "1" ]]; then
  log "$(date -Is) SKIP_MOSEI_WAIT=1 — starting MOSI trial220 without waiting for MOSEI."
elif [[ -n "${MOSEI_MASTER_PID:-}" ]]; then
  wait_pid="${MOSEI_MASTER_PID}"
elif [[ -f "$PID_FILE" ]]; then
  wait_pid="$(tr -d '[:space:]' <"$PID_FILE" || true)"
fi

if [[ "${SKIP_MOSEI_WAIT:-0}" != "1" ]]; then
  if [[ -z "$wait_pid" ]] || ! [[ "$wait_pid" =~ ^[1-9][0-9]*$ ]]; then
    log "ERROR: need a MOSEI master PID. Either:"
    log "  - export MOSEI_MASTER_PID=<pid of bash running run_prism_ablations_mosei_phase1_trial70.sh>, or"
    log "  - write that PID to ${PID_FILE} (one line), or"
    log "  - SKIP_MOSEI_WAIT=1 for immediate MOSI run."
    exit 1
  fi

  log "$(date -Is) queued: waiting for MOSEI prism master PID=${wait_pid}"
  while kill -0 "$wait_pid" 2>/dev/null; do
    sleep "$POLL_SEC"
  done
  log "$(date -Is) MOSEI prism master PID=${wait_pid} has exited; launching MOSI trial220 PRISM batch."
fi

if [[ ! "${JOBS_PER_GPU+x}" ]]; then
  export JOBS_PER_GPU="1,1"
fi

bash "${MY_CREATION}/ablation_study/run_prism_ablations_mosi_trial220.sh" >>"$QUEUE_LOG" 2>&1
