#!/usr/bin/env bash
# All CMU-MOSEI phase1 trial70 PRISM ablations → one facet + one joint t-SNE figure.
# Run from My_creation: bash visualize/run_tsne_mosei_prism_all.sh
set -euo pipefail
export MPLBACKEND="${MPLBACKEND:-Agg}"
MY_CREATION="$(cd "$(dirname "$0")/.." && pwd)"
cd "$MY_CREATION"

PYTHON="${PYTHON:-/root/autodl-tmp/anaconda3/envs/ITHP5090/bin/python}"
if ! "$PYTHON" -c "import torch" 2>/dev/null; then
  PYTHON="${PYTHON:-python3}"
fi

RUNS="$MY_CREATION/ablation_study/runs"
OUT="${OUT:-$MY_CREATION/visualize/figures/mosei_prism_tsne_all}"
SPLIT="${SPLIT:-test}"
MAX_SAMPLES="${MAX_SAMPLES:-2500}"

args=(
  --split "$SPLIT"
  --outdir "$OUT"
  --max-samples "$MAX_SAMPLES"
  --ckpt "PRISM=${RUNS}/mosei_phase1_trial70/checkpoints/infogate_mosei_best.pt"
  --ckpt "w/o InfoGate=${RUNS}/mosei_phase1_trial70_no_infogate/checkpoints/infogate_mosei_best.pt"
  --ckpt "w/o MSelector=${RUNS}/mosei_phase1_trial70_no_mselector/checkpoints/infogate_mosei_best.pt"
  --ckpt "w/o IB=${RUNS}/mosei_phase1_trial70_no_ib/checkpoints/infogate_mosei_best.pt"
  --ckpt "w/o ConfGating=${RUNS}/mosei_phase1_trial70_no_conf_gating/checkpoints/infogate_mosei_best.pt"
  --ckpt "w/o AdaptiveGate=${RUNS}/mosei_phase1_trial70_no_adaptive_gate/checkpoints/infogate_mosei_best.pt"
)

echo "=== facet ==="
"$PYTHON" visualize/tsne_mosei_ablation.py "${args[@]}" --mode facet

echo "=== joint ==="
"$PYTHON" visualize/tsne_mosei_ablation.py "${args[@]}" --mode joint

echo "Done. Figures under $OUT"
