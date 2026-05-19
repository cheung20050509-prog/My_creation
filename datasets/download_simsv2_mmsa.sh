#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Download CH-SIMS v2 (SIMSv2) MMSA-style supervised features for InfoGate.
#
# Source (same as prior session / MMSA README):
#   Google Drive folder: https://drive.google.com/drive/folders/1A2S4pqCHryGmiqnNSPLv7rEg63WvjCSk
#   File path in folder: CH-SIMS v2(s) / Processed / unaligned.pkl
#   Direct gdown id:     13JdO6GbPHOGZ8yLBFUrNR8c2FvHN-E_O
#
# Output (under this directory):
#   simsv2.pkl              — main file used by train.py (datasets/simsv2.pkl)
#   simsv2_mmsa.pkl         — symlink → simsv2.pkl
#   sims_unaligned_mmsa.pkl — symlink → simsv2.pkl (MMSA/KuDA naming)
#
# Pickle must contain per split: text_bert, audio, vision, audio_lengths,
# vision_lengths, regression_labels (train / valid|dev / test).
#
# Usage:
#   bash My_creation/datasets/download_simsv2_mmsa.sh
#   VERIFY_ONLY=1 bash ...   # check existing simsv2.pkl, no download
#   FORCE=1 bash ...         # re-download even if simsv2.pkl exists
# -----------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-/root/autodl-tmp/anaconda3/envs/ITHP5090/bin/python}"
GDOWN="${GDOWN:-$PYTHON -m gdown}"

# MMSA feature pack — CH-SIMS v2(s) Processed unaligned.pkl
MMSA_DRIVE_FOLDER="https://drive.google.com/drive/folders/1A2S4pqCHryGmiqnNSPLv7rEg63WvjCSk"
MMSA_SIMS_FILE_ID="${MMSA_SIMS_FILE_ID:-13JdO6GbPHOGZ8yLBFUrNR8c2FvHN-E_O}"
STAGING="$SCRIPT_DIR/_mmsa_download"
STAGED_PKL="$STAGING/sims_unaligned.pkl"
OUT_PKL="$SCRIPT_DIR/simsv2.pkl"
FORCE="${FORCE:-0}"
VERIFY_ONLY="${VERIFY_ONLY:-0}"

verify_pkl() {
  local p="$1"
  "$PYTHON" - <<PY
import pickle, sys
path = """$p"""
with open(path, "rb") as fh:
    d = pickle.load(fh)
splits = list(d.keys())
need_split = {"train", "test"} & set(splits)
if "valid" not in splits and "dev" not in splits:
    raise SystemExit("missing dev/valid split")
need_keys = {
    "text_bert", "audio", "vision",
    "audio_lengths", "vision_lengths", "regression_labels",
}
for sp in ("train", "valid" if "valid" in splits else "dev", "test"):
    ex = d[sp]
    missing = need_keys - set(ex.keys())
    if missing:
        raise SystemExit(f"split {sp} missing keys: {sorted(missing)}")
    n = len(ex["regression_labels"])
    tb = ex["text_bert"][0]
    aud = ex["audio"][0]
    vis = ex["vision"][0]
    import numpy as np
    tb = np.asarray(tb)
    aud = np.asarray(aud)
    vis = np.asarray(vis)
    print(f"OK {path}")
    print(f"  splits={sorted(splits)}")
    print(f"  {sp}: n={n} text_bert={tb.shape} audio={aud.shape} vision={vis.shape}")
    print(f"  audio_dim={aud.shape[-1]} vision_dim={vis.shape[-1]} (expect 25, 177)")
    if aud.shape[-1] != 25 or vis.shape[-1] != 177:
        print("  WARN: dims differ from global_configs simsv2 (25, 177)", file=sys.stderr)
for sp in splits:
    print(f"  {sp}: {len(d[sp]['regression_labels'])} samples")
PY
}

install_pkl() {
  local src="$1"
  if [[ ! -f "$src" ]]; then
    echo "ERROR: staged file missing: $src" >&2
    exit 1
  fi
  if [[ -f "$OUT_PKL" && "$FORCE" != "1" ]]; then
    if cmp -s "$src" "$OUT_PKL"; then
      echo "[install] $OUT_PKL already matches staged file (skip copy)."
    else
      bak="${OUT_PKL}.bak.$(date +%Y%m%d_%H%M%S)"
      echo "[install] backing up existing $OUT_PKL -> $bak"
      mv "$OUT_PKL" "$bak"
      cp -a "$src" "$OUT_PKL"
    fi
  else
    cp -a "$src" "$OUT_PKL"
  fi
  ln -sf simsv2.pkl simsv2_mmsa.pkl
  ln -sf simsv2.pkl sims_unaligned_mmsa.pkl
  ls -lh "$OUT_PKL" simsv2_mmsa.pkl sims_unaligned_mmsa.pkl
}

echo "[$(date -Iseconds)] SIMSv2 MMSA download"
echo "  OUT=$OUT_PKL"
echo "  MMSA folder: $MMSA_DRIVE_FOLDER"
echo "  gdown id:    $MMSA_SIMS_FILE_ID"

if [[ -f "$OUT_PKL" ]]; then
  echo "[check] existing $OUT_PKL"
  verify_pkl "$OUT_PKL"
  if [[ "$VERIFY_ONLY" == "1" ]]; then
    echo "VERIFY_ONLY=1; done."
    exit 0
  fi
  if [[ "$FORCE" != "1" ]]; then
    echo "simsv2.pkl present. Set FORCE=1 to re-download, or VERIFY_ONLY=1 to exit."
    exit 0
  fi
fi

if [[ "$VERIFY_ONLY" == "1" ]]; then
  echo "ERROR: VERIFY_ONLY=1 but $OUT_PKL not found" >&2
  exit 1
fi

"$PYTHON" -m pip install -q gdown

mkdir -p "$STAGING"
if [[ -f "$STAGED_PKL" && "$FORCE" != "1" ]]; then
  echo "[download] reuse staged $STAGED_PKL"
else
  echo "[download] gdown -> $STAGED_PKL (~3.4 GiB, may take several minutes)"
  rm -f "$STAGED_PKL"
  $GDOWN "$MMSA_SIMS_FILE_ID" -O "$STAGED_PKL"
  ls -lh "$STAGED_PKL"
fi

verify_pkl "$STAGED_PKL"
install_pkl "$STAGED_PKL"

# Drop duplicate staging copy if it matches output (save ~3.4G)
if [[ -f "$STAGED_PKL" ]] && cmp -s "$STAGED_PKL" "$OUT_PKL"; then
  rm -f "$STAGED_PKL"
  echo "[cleanup] removed duplicate $STAGED_PKL (same as simsv2.pkl)"
fi

echo "[$(date -Iseconds)] done. train.py reads: datasets/simsv2.pkl (--simsv2_feature_mode mmsa default)"
