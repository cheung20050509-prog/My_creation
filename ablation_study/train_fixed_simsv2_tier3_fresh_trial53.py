#!/usr/bin/env python3
"""Launch ``ablation_study/train.py`` with frozen SIMSv2 trial 53 hparams."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from simsv2_tier3_fresh_trial53_hparams import PAPER_TEST_METRICS, build_train_argv


ABLATION_CHOICES = (
    "none",
    "no_infogate",
    "no_mselector",
    "no_ib",
    "no_ib_no_mselector_no_infogate",
    "no_conf",
    "no_conf_gating",
    "no_adaptive_gate",
)


def _default_checkpoint_dir(my_creation: Path, ablation: str) -> Path:
    base = my_creation / "ablation_study/runs"
    if ablation == "none":
        return base / "simsv2_tier3_fresh_trial53/checkpoints"
    return base / f"simsv2_tier3_fresh_trial53_{ablation}/checkpoints"


def main() -> int:
    here = Path(__file__).resolve().parent
    my_creation = here.parent

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--ablation",
        default="none",
        choices=ABLATION_CHOICES,
        help="PRISM ablation: no_mselector=w/o DPR, no_ib=w/o VTB.",
    )
    ap.add_argument(
        "--checkpoint-dir",
        default=None,
        help=(
            "Passed to train.py --checkpoint_dir. Default: "
            "runs/simsv2_tier3_fresh_trial53[_<ablation>]/checkpoints"
        ),
    )
    ap.add_argument("--dry-run", action="store_true", help="Print invocation and exit")
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
        print("paper test targets:", PAPER_TEST_METRICS)
        print("exec:", " ".join(cmd))
        return 0

    subprocess.run(cmd, cwd=str(my_creation), check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
