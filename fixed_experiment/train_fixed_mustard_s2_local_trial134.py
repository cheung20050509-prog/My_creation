#!/usr/bin/env python3
"""Launch ``My_creation/train_classify.py`` with frozen MUStARD s2_local trial 134 hyperparameters."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from mustard_s2_local_trial134_hparams import PAPER_TEST_METRICS, build_train_argv


def _override_seed_in_argv(argv: list[str], seed: int) -> list[str]:
    out = list(argv)
    i = out.index("--seed")
    out[i + 1] = str(int(seed))
    return out


def main() -> int:
    here = Path(__file__).resolve().parent
    my_creation = here.parent
    train_py = my_creation / "train_classify.py"

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--checkpoint-dir",
        default=str(my_creation / "fixed_experiment/runs/mustard_s2_local_trial134/checkpoints"),
        help="Passed to train_classify.py --checkpoint_dir",
    )
    ap.add_argument("--dry-run", action="store_true", help="Print train_classify.py invocation and exit")
    ap.add_argument("--seed", type=int, default=None, help="Override --seed")
    args = ap.parse_args()

    argv = build_train_argv(checkpoint_dir=args.checkpoint_dir)
    if args.seed is not None:
        argv = _override_seed_in_argv(argv, args.seed)

    cmd = [sys.executable, "-u", str(train_py), *argv]
    if args.dry_run:
        print("cwd:", my_creation)
        print("paper test targets:", PAPER_TEST_METRICS)
        print("exec:", " ".join(cmd))
        return 0

    subprocess.run(cmd, cwd=str(my_creation), check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
