#!/usr/bin/env python3
"""Print Optuna best trial when the objective is min dev MAE (selection_metric=mae).

Reads ``trial.value`` (dev MAE), ``user_attrs`` (test_mae, train_log_path), and
re-parses ``Best Results`` from the train log when available.

Example (paths relative to repo clone; DBs live under ``My_creation/logs/optuna/4090D_restart/``)::

  cd My_creation
  python scripts/optuna_summarize_best_devmae.py \\
    --storage sqlite:///$(pwd)/logs/optuna/4090D_restart/phase7_mosi/db/mosi.db \\
    --study-name infogate_mosi_phase7_mosi_4090d_devmae
"""
from __future__ import annotations

import argparse
import os
import sys

import optuna

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_MY_CREATION = os.path.dirname(_SCRIPT_DIR)
if _MY_CREATION not in sys.path:
    sys.path.insert(0, _MY_CREATION)

import optuna_search_v2 as osv  # noqa: E402


def main():
    pa = argparse.ArgumentParser(
        description="Summarize best Optuna trial for dev-MAE HPO (mae objective).",
    )
    pa.add_argument("--storage", required=True, help="Optuna storage URI, e.g. sqlite:///.../mosi.db")
    pa.add_argument("--study-name", required=True, help="Study name inside the storage")
    args = pa.parse_args()

    study = optuna.load_study(study_name=args.study_name, storage=args.storage)
    try:
        bt = study.best_trial
    except ValueError as e:
        print(f"No best trial: {e}", file=sys.stderr)
        sys.exit(2)

    print("=" * 60)
    print(f"Study: {args.study_name}")
    print(f"Storage: {args.storage}")
    print(f"Best trial: #{bt.number}")
    print(f"  Optuna objective (min dev MAE): {bt.value:.6f}")
    ua = dict(bt.user_attrs or {})
    if "test_mae" in ua:
        print(f"  user_attr test_mae (Best Results @ dev-ckpt): {float(ua['test_mae']):.6f}")
    if "dev_mae" in ua:
        print(f"  user_attr dev_mae (should match objective): {float(ua['dev_mae']):.6f}")
    log_path = ua.get("train_log_path")
    if log_path and os.path.isfile(log_path):
        br = osv.parse_best_results(log_path)
        mae = br.get("MAE")
        if mae is not None:
            print(f"  Parsed Best Results MAE from log: {float(mae):.6f}")
        else:
            print("  Parsed Best Results MAE from log: (missing)")
    elif log_path:
        print(f"  train_log_path (missing file): {log_path}")
    print("  Params (subset):")
    for k in sorted(bt.params):
        print(f"    {k}: {bt.params[k]}")
    print("=" * 60)


if __name__ == "__main__":
    main()
