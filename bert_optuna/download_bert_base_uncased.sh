#!/usr/bin/env bash
# One-shot download: HuggingFace google-bert/bert-base-uncased -> ../bert-base-uncase
# (local folder name is intentional; HF id remains bert-base-uncased.)
set -euo pipefail
MY_CREATION="$(cd "$(dirname "$0")/.." && pwd)"
export DEST="${BERT_UNCASE_DIR:-$MY_CREATION/bert-base-uncase}"
PYTHON="${PYTHON:-/root/autodl-tmp/anaconda3/envs/ITHP5090/bin/python}"
mkdir -p "$DEST"
echo "[download] DEST=$DEST"
exec "$PYTHON" -c "
import os
from huggingface_hub import snapshot_download
dest = os.environ['DEST']
os.makedirs(dest, exist_ok=True)
snapshot_download(repo_id='google-bert/bert-base-uncased', local_dir=dest)
print('OK:', dest)
"
