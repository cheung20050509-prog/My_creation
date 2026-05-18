#!/usr/bin/env python3
"""Compare ``train.py`` Best Results (from log) with ``test_results`` inside a checkpoint.

``torch.save`` dict includes ``test_results`` (MAE, Acc-7, …) at the dev-selected
best epoch (see ``train.py``).  This script prints both and exits non-zero on mismatch.

Usage (from ``My_creation/``); 可将标准输出重定向到 ``Qualitative_Evaluation/results/``::

    python Qualitative_Evaluation/verify_ckpt_best_results.py \\
      --checkpoint logs/optuna/4090D_restart/phase1/checkpoints/optuna_mosei_phase1/trial_70/infogate_mosei_best.pt \\
      --train-log logs/optuna/4090D_restart/phase1/train_logs/mosei_phase1_trial_70.log \\
      > Qualitative_Evaluation/results/verify_mosei_trial70.txt
"""

from __future__ import annotations

import argparse
import os
import sys

_QE = os.path.dirname(os.path.abspath(__file__))
_MY = os.path.dirname(_QE)
if _MY not in sys.path:
    sys.path.insert(0, _MY)

import optuna_search_v2 as osv  # noqa: E402
import torch  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--train-log", type=str, default=None,
                    help="Trial train log (optional). If set, parse Best Results MAE and compare.")
    args = ap.parse_args()

    ck = os.path.abspath(args.checkpoint)
    if not os.path.isfile(ck):
        print(f"ERROR: checkpoint not found: {ck}", file=sys.stderr)
        return 1

    obj = torch.load(ck, map_location="cpu", weights_only=False)
    tr = obj.get("test_results")
    if not isinstance(tr, dict) or "mae" not in tr:
        print("ERROR: checkpoint has no test_results dict with 'mae'", file=sys.stderr)
        return 1

    print("=== checkpoint test_results (dev-selected epoch) ===")
    for k in sorted(tr.keys()):
        v = tr[k]
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v}")

    if not args.train_log:
        print("\n(no --train-log; skip log comparison)")
        return 0

    log_path = os.path.abspath(args.train_log)
    if not os.path.isfile(log_path):
        print(f"ERROR: train log not found: {log_path}", file=sys.stderr)
        return 1

    br = osv.parse_best_results(log_path)
    if "MAE" not in br:
        print("ERROR: could not parse MAE from Best Results block in log", file=sys.stderr)
        return 1

    log_mae = float(br["MAE"])
    ck_mae = float(tr["mae"])
    print("\n=== log Best Results MAE ===")
    print(f"  {log_mae:.6f}")

    tol = 1e-4
    if abs(log_mae - ck_mae) > tol:
        print(f"\nMISMATCH: |log_mae - ckpt_mae| = {abs(log_mae - ck_mae):.6f} > {tol}", file=sys.stderr)
        return 2

    print("\nOK: log Best Results MAE matches checkpoint test_results['mae'] within tolerance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
