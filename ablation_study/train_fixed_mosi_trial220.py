#!/usr/bin/env python3
"""Launch ``ablation_study/train.py`` with frozen MOSI phase4_mosi trial 220 hyperparameters.

All modes (including ``--ablation none``) use this directory's ``train.py`` and
``mosi_trial220_hparams.build_train_argv`` so code and argv stay under
``ablation_study/``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from mosi_trial220_hparams import build_train_argv


def _default_checkpoint_dir(my_creation: Path, ablation: str) -> Path:
    base = my_creation / "ablation_study/runs"
    if ablation == "none":
        return base / "mosi_trial220/checkpoints"
    return base / f"mosi_trial220_{ablation}/checkpoints"


def main() -> int:
    here = Path(__file__).resolve().parent
    my_creation = here.parent

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--ablation",
        default="none",
        choices=(
            "none",
            "no_infogate",
            "no_mselector",
            "no_ib",
            "no_ib_no_mselector_no_infogate",
            "no_conf_gating",
            "no_adaptive_gate",
        ),
        help="PRISM ablation: ``no_mselector``=w/o DPR, ``no_ib``=w/o VTB (same trial220 knobs otherwise).",
    )
    ap.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Passed to train.py --checkpoint_dir (default: runs/mosi_trial220[_<ablation>]/checkpoints)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print train.py invocation and exit",
    )
    args = ap.parse_args()

    ckpt = (
        Path(args.checkpoint_dir).expanduser().resolve()
        if args.checkpoint_dir
        else _default_checkpoint_dir(my_creation, args.ablation)
    )

    train_py = here / "train.py"
    argv = build_train_argv(checkpoint_dir=str(ckpt), ablation=args.ablation)
    cmd = [sys.executable, "-u", str(train_py), *argv]
    if args.dry_run:
        print("cwd:", my_creation)
        print("train:", train_py)
        print("exec:", " ".join(cmd))
        return 0

    subprocess.run(cmd, cwd=str(my_creation), check=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
