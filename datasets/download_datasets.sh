#!/usr/bin/env bash
# Regression pickles for InfoGate (run from My_creation/datasets/).
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"
PYTHON="${PYTHON:-/root/autodl-tmp/anaconda3/envs/ITHP5090/bin/python}"
GDOWN="${GDOWN:-$PYTHON -m gdown}"

"$PYTHON" -m pip install -q gdown

# CMU-MOSI / CMU-MOSEI (ITHP-style gdown; same as ITHP/datasets/download_datasets.sh)
if [[ ! -f mosi.pkl ]]; then
  echo "[download] mosi.pkl"
  $GDOWN https://drive.google.com/uc?id=12HbavGOtoVCqicvSYWl3zImli5Jz0Nou -O mosi.pkl
fi
if [[ ! -f mosei.pkl ]]; then
  echo "[download] mosei.pkl"
  $GDOWN https://drive.google.com/uc?id=1VJhSc2TGrPU8zJSVTYwn5kfuG47VaNQ3 -O mosei.pkl
fi

# CH-SIMS v2 — MMSA Processed unaligned.pkl → simsv2.pkl (text_bert + lengths)
bash "$DIR/download_simsv2_mmsa.sh"
