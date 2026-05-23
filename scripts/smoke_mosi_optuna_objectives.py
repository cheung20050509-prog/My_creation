#!/usr/bin/env python3
"""Smoke-test dev vs test MAE parsing for MOSI Optuna objectives (no training)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from optuna_search_v2 import parse_best_results, parse_min_dev_mae_stage2  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "log_path",
        nargs="?",
        default=str(
            SCRIPT_DIR
            / "logs/optuna/4090D_restart/more/mosi/train_logs/mosi_more_mosi_trial_0.log"
        ),
    )
    p.add_argument("--stage1-epochs", type=int, default=7)
    args = p.parse_args()

    log = Path(args.log_path)
    if not log.is_file():
        print(f"ERROR: missing log {log}", file=sys.stderr)
        return 1

    best = parse_best_results(str(log))
    dev = parse_min_dev_mae_stage2(str(log), "mosi", args.stage1_epochs)
    test_mae = best.get("MAE")
    if test_mae is None or dev is None:
        print(f"ERROR: parse failed best={best} dev={dev}", file=sys.stderr)
        return 1

    print(f"log: {log}")
    print(f"  min dev MAE (stage2): {dev:.6f}  -> optuna_objective dev_mae")
    print(f"  Best Results test MAE: {test_mae:.6f}  -> optuna_objective test_mae")
    if dev >= test_mae:
        print("  (dev can be < or > test; both values are valid objectives)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
