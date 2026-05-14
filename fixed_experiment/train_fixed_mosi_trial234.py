#!/usr/bin/env python3
"""Launch ``My_creation/train.py`` with frozen MOSI phase4_mosi trial 234 hyperparameters."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from mosi_trial234_hparams import build_train_argv


def _override_seed_in_argv(argv: list[str], seed: int) -> list[str]:
    out = list(argv)
    try:
        i = out.index("--seed")
        out[i + 1] = str(int(seed))
    except (ValueError, IndexError) as exc:
        raise RuntimeError("build_train_argv did not emit --seed / --seed value") from exc
    return out


def main() -> int:
    here = Path(__file__).resolve().parent
    my_creation = here.parent
    train_py = here / "train.py"

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--checkpoint-dir",
        default=str(my_creation / "fixed_experiment/runs/mosi_trial234/checkpoints"),
        help="Passed to train.py --checkpoint_dir",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print train.py invocation and exit",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override train.py --seed (default: frozen hparams TRIAL_234_PARAMS['seed'])",
    )
    args = ap.parse_args()

    argv = build_train_argv(checkpoint_dir=args.checkpoint_dir)
    if args.seed is not None:
        argv = _override_seed_in_argv(argv, args.seed)
    cmd = [sys.executable, "-u", str(train_py), *argv]
    if args.dry_run:
        print("cwd:", my_creation)
        print("exec:", " ".join(cmd))
        return 0

    subprocess.run(cmd, cwd=str(my_creation), check=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
