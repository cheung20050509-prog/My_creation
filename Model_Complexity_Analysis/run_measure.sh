#!/usr/bin/env bash
# Run model-complexity driver with conda env ITHP5090 (same default as run_optuna_4090d_restart.sh).
# Driver: PYTHON=/path/to/python ./run_measure.sh [args...]
# Workers default to ITHP5090 inside measure_fixed_cases.py; override with MODEL_COMPLEXITY_PYTHON=...
set -euo pipefail
_MY_CREATION="$(cd "$(dirname "$0")/.." && pwd)"
cd "${_MY_CREATION}"
PYTHON="${PYTHON:-/root/autodl-tmp/anaconda3/envs/ITHP5090/bin/python}"
exec "${PYTHON}" -u Model_Complexity_Analysis/measure_fixed_cases.py "$@"
