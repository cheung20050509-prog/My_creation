#!/usr/bin/env bash
# Populate My_creation/fixed_experiment/paper_repro_bundle/ with a physical copy of
# code, pickles, HF weights, and frozen hparams. Then patch bundle code paths.
#
# Usage (from anywhere):
#   bash My_creation/scripts/bundle_paper_repro_to_fixed_experiment.sh
#
# Partial copy (saves disk when iterating on layout):
#   ONLY=code|data|weights|frozen  bash My_creation/scripts/...
#   ONLY=mosi|mosei|simsv2         copy only that dataset pickle (+ code/weights/frozen as needed)
#   ONLY=all   (default)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MY_CREATION="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUNDLE="${MY_CREATION}/fixed_experiment/paper_repro_bundle"
ONLY="${ONLY:-all}"

copy_code() {
  mkdir -p "${BUNDLE}/code"
  local f
  for f in train.py deberta_infogate.py bert_infogate.py infogate_modules.py \
           global_configs.py selection_utils.py simsv2_metrics.py optuna_search_v2.py; do
    cp -a "${MY_CREATION}/${f}" "${BUNDLE}/code/"
  done
  python3 "${MY_CREATION}/scripts/patch_paper_repro_bundle.py" "${BUNDLE}"
}

copy_data() {
  mkdir -p "${BUNDLE}/data/datasets"
  local p
  for p in mosi.pkl mosei.pkl simsv2.pkl; do
    if [[ -f "${MY_CREATION}/datasets/${p}" ]]; then
      cp -a "${MY_CREATION}/datasets/${p}" "${BUNDLE}/data/datasets/"
    else
      echo "warn: missing ${MY_CREATION}/datasets/${p}" >&2
    fi
  done
}

copy_data_one() {
  local name="$1"
  mkdir -p "${BUNDLE}/data/datasets"
  local p="${name}.pkl"
  if [[ -f "${MY_CREATION}/datasets/${p}" ]]; then
    cp -a "${MY_CREATION}/datasets/${p}" "${BUNDLE}/data/datasets/"
  else
    echo "warn: missing ${MY_CREATION}/datasets/${p}" >&2
  fi
}

copy_weights() {
  mkdir -p "${BUNDLE}/weights"
  if [[ -d "${MY_CREATION}/deberta-v3-base" ]]; then
    rsync -a --delete --exclude='__pycache__' "${MY_CREATION}/deberta-v3-base/" \
      "${BUNDLE}/weights/deberta-v3-base/"
  else
    echo "warn: missing ${MY_CREATION}/deberta-v3-base" >&2
  fi
  if [[ -d "${MY_CREATION}/bert-base-chinese" ]]; then
    rsync -a --delete --exclude='__pycache__' "${MY_CREATION}/bert-base-chinese/" \
      "${BUNDLE}/weights/bert-base-chinese/"
  else
    echo "warn: missing ${MY_CREATION}/bert-base-chinese" >&2
  fi
}

copy_frozen() {
  mkdir -p "${BUNDLE}/frozen"
  local f
  for f in mosi_trial234_hparams.py mosei_phase1_trial70_hparams.py simsv2_phase4_trial52_hparams.py; do
    cp -a "${MY_CREATION}/fixed_experiment/${f}" "${BUNDLE}/frozen/"
  done
}

write_launchers() {
  mkdir -p "${BUNDLE}/frozen" "${BUNDLE}/runs"
  # Python launchers (cwd = bundle root; train entry = code/train.py)
  cat >"${BUNDLE}/frozen/run_mosi_trial234.py" <<'PY'
#!/usr/bin/env python3
"""Launch code/train.py with frozen MOSI trial-234 argv (paper repro bundle)."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

def main() -> int:
    bundle = Path(__file__).resolve().parent.parent
    os.environ["PYTHONPATH"] = str(bundle / "code") + (
        os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else ""
    )
    sys.path.insert(0, str(bundle / "frozen"))
    from mosi_trial234_hparams import build_train_argv

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--checkpoint-dir",
        default=str(bundle / "runs/mosi_trial234/checkpoints"),
        help="Passed to train.py --checkpoint_dir",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    argv = build_train_argv(checkpoint_dir=args.checkpoint_dir)
    cmd = [sys.executable, "-u", str(bundle / "code" / "train.py"), *argv]
    if args.dry_run:
        print("cwd:", bundle)
        print("exec:", " ".join(cmd))
        return 0
    subprocess.run(cmd, cwd=str(bundle), check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
PY

  cat >"${BUNDLE}/frozen/run_mosei_phase1_trial70.py" <<'PY'
#!/usr/bin/env python3
"""Launch code/train.py with frozen MOSEI phase1 trial-70 argv."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

def main() -> int:
    bundle = Path(__file__).resolve().parent.parent
    os.environ["PYTHONPATH"] = str(bundle / "code") + (
        os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else ""
    )
    sys.path.insert(0, str(bundle / "frozen"))
    from mosei_phase1_trial70_hparams import build_train_argv

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--checkpoint-dir",
        default=str(bundle / "runs/mosei_phase1_trial70/checkpoints"),
        help="Passed to train.py --checkpoint_dir",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    argv = build_train_argv(checkpoint_dir=args.checkpoint_dir)
    cmd = [sys.executable, "-u", str(bundle / "code" / "train.py"), *argv]
    if args.dry_run:
        print("cwd:", bundle)
        print("exec:", " ".join(cmd))
        return 0
    subprocess.run(cmd, cwd=str(bundle), check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
PY

  cat >"${BUNDLE}/frozen/run_simsv2_phase4_trial52.py" <<'PY'
#!/usr/bin/env python3
"""Launch code/train.py with frozen SIMSv2 phase4 trial-52 argv (paper row)."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

def main() -> int:
    bundle = Path(__file__).resolve().parent.parent
    os.environ["PYTHONPATH"] = str(bundle / "code") + (
        os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else ""
    )
    sys.path.insert(0, str(bundle / "frozen"))
    from simsv2_phase4_trial52_hparams import build_train_argv

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--checkpoint-dir",
        default=str(bundle / "runs/simsv2_phase4_trial52/checkpoints"),
        help="Passed to train.py --checkpoint_dir",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    argv = build_train_argv(checkpoint_dir=args.checkpoint_dir)
    cmd = [sys.executable, "-u", str(bundle / "code" / "train.py"), *argv]
    if args.dry_run:
        print("cwd:", bundle)
        print("exec:", " ".join(cmd))
        return 0
    subprocess.run(cmd, cwd=str(bundle), check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
PY
  chmod +x "${BUNDLE}/frozen/run_"*.py

  # Thin shell wrappers (plan: run_*.sh)
  cat >"${BUNDLE}/run_mosi_trial234.sh" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${ROOT}/code${PYTHONPATH:+:${PYTHONPATH}}"
exec python3 "${ROOT}/frozen/run_mosi_trial234.py" "$@"
SH
  cat >"${BUNDLE}/run_mosei_phase1_trial70.sh" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${ROOT}/code${PYTHONPATH:+:${PYTHONPATH}}"
exec python3 "${ROOT}/frozen/run_mosei_phase1_trial70.py" "$@"
SH
  cat >"${BUNDLE}/run_simsv2_phase4_trial52.sh" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${ROOT}/code${PYTHONPATH:+:${PYTHONPATH}}"
exec python3 "${ROOT}/frozen/run_simsv2_phase4_trial52.py" "$@"
SH
  chmod +x "${BUNDLE}/run_"*.sh
}

manifest() {
  local mf="${BUNDLE}/MANIFEST.sha256"
  (
    cd "${BUNDLE}"
    find . -type f ! -path './MANIFEST.sha256' ! -path './.git/*' -print0 |
      sort -z | xargs -0 sha256sum
  ) >"${mf}.tmp"
  mv "${mf}.tmp" "${mf}"
  echo "Wrote ${mf}"
}

du_report() {
  du -sh "${BUNDLE}/code" "${BUNDLE}/data" "${BUNDLE}/weights" "${BUNDLE}/frozen" 2>/dev/null || true
  du -sh "${BUNDLE}" || true
}

case "${ONLY}" in
  all)
    copy_code
    copy_data
    copy_weights
    copy_frozen
    write_launchers
    ;;
  code)
    copy_code
    write_launchers
    ;;
  data) copy_data ;;
  weights) copy_weights ;;
  frozen)
    copy_frozen
    write_launchers
    ;;
  mosi)
    copy_code
    copy_data_one mosi
    rsync -a --delete --exclude='__pycache__' "${MY_CREATION}/deberta-v3-base/" \
      "${BUNDLE}/weights/deberta-v3-base/"
    copy_frozen
    write_launchers
    ;;
  mosei)
    copy_code
    copy_data_one mosei
    rsync -a --delete --exclude='__pycache__' "${MY_CREATION}/deberta-v3-base/" \
      "${BUNDLE}/weights/deberta-v3-base/"
    copy_frozen
    write_launchers
    ;;
  simsv2)
    copy_code
    copy_data_one simsv2
    rsync -a --delete --exclude='__pycache__' "${MY_CREATION}/deberta-v3-base/" \
      "${BUNDLE}/weights/deberta-v3-base/"
    rsync -a --delete --exclude='__pycache__' "${MY_CREATION}/bert-base-chinese/" \
      "${BUNDLE}/weights/bert-base-chinese/"
    copy_frozen
    write_launchers
    ;;
  *)
    echo "ONLY must be all|code|data|weights|frozen|mosi|mosei|simsv2, got: ${ONLY}" >&2
    exit 1
    ;;
esac

mkdir -p "${BUNDLE}/runs" "${BUNDLE}/env"
manifest
du_report
echo "Bundle root: ${BUNDLE}"
