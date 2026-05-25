#!/usr/bin/env python3
"""Launch ``ablation_study/train_classify.py`` with frozen UR-FUNNY v2 trial 162 hyperparameters."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from ur_funny_trial162_hparams import PAPER_TEST_METRICS, build_train_argv


def _override_seed_in_argv(argv: list[str], seed: int) -> list[str]:
    out = list(argv)
    try:
        i = out.index("--seed")
        out[i + 1] = str(int(seed))
    except (ValueError, IndexError) as exc:
        raise RuntimeError("build_train_argv did not emit --seed") from exc
    return out


def _default_checkpoint_dir(my_creation: Path, ablation: str) -> Path:
    base = my_creation / "ablation_study/runs"
    if ablation == "none":
        return base / "ur_funny_trial162/checkpoints"
    return base / f"ur_funny_trial162_{ablation}/checkpoints"


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
            "no_conf",
            "no_conf_gating",
            "no_adaptive_gate",
        ),
        help=(
            "PRISM ablation: ``no_mselector``=w/o DPR, ``no_ib``=w/o VTB; "
            "``no_conf`` aliases ``no_conf_gating`` in train_classify.py."
        ),
    )
    ap.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Passed to train_classify.py --checkpoint_dir (default: runs/ur_funny_trial162[_<ablation>]/checkpoints)",
    )
    ap.add_argument("--dry-run", action="store_true", help="Print train_classify.py invocation and exit")
    ap.add_argument("--seed", type=int, default=None, help="Override train_classify.py --seed")
    args = ap.parse_args()

    ckpt = (
        Path(args.checkpoint_dir).expanduser().resolve()
        if args.checkpoint_dir
        else _default_checkpoint_dir(my_creation, args.ablation)
    )

    train_py = here / "train_classify.py"
    argv = build_train_argv(checkpoint_dir=str(ckpt), ablation=args.ablation)
    if args.seed is not None:
        argv = _override_seed_in_argv(argv, args.seed)

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
