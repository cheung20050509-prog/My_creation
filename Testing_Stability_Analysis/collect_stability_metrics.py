#!/usr/bin/env python3
"""Collect dev-selected **test** metrics from multi-seed train.log files.

Parses the ``Best Results`` block (same as ``train.py`` final summary) for:
MAE, Corr, Acc-2, Acc-7, F1.

Usage::

    python Testing_Stability_Analysis/collect_stability_metrics.py \\
      My_creation/Testing_Stability_Analysis/runs/mosi_t39_20260524_120000
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics
import sys
from pathlib import Path


def extract_best_results(log_text: str) -> dict[str, float] | None:
    if "Best Results" not in log_text:
        return None
    start = log_text.find("Best Results")
    chunk = log_text[start:]
    le = chunk.find("\nLast Epoch")
    if le != -1:
        chunk = chunk[:le]

    def _pct(name: str) -> float | None:
        m = re.search(rf"^\s+{re.escape(name)}:\s+([0-9.]+)%\s*$", chunk, re.MULTILINE)
        return float(m.group(1)) if m else None

    def _float(name: str) -> float | None:
        m = re.search(rf"^\s+{name}:\s+([0-9.]+)\s*$", chunk, re.MULTILINE)
        return float(m.group(1)) if m else None

    mae = _float("MAE")
    corr = _float("Corr")
    f1 = _float("F1")
    acc2 = _pct("Acc-2")
    acc7 = _pct("Acc-7")
    if mae is None:
        return None
    out: dict[str, float] = {"mae": mae}
    if corr is not None:
        out["corr"] = corr
    if acc2 is not None:
        out["acc2"] = acc2
    if acc7 is not None:
        out["acc7"] = acc7
    if f1 is not None:
        out["f1"] = f1
    return out


def parse_run_dir(name: str) -> tuple[str, int] | None:
    if name.startswith("mosi_seed"):
        return "mosi", int(name[len("mosi_seed") :])
    if name.startswith("mosei_seed"):
        return "mosei", int(name[len("mosei_seed") :])
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_root", type=Path)
    args = ap.parse_args()
    root = args.run_root.expanduser().resolve()
    if not root.is_dir():
        print(f"ERROR: not a directory: {root}", file=sys.stderr)
        return 1

    rows: list[dict[str, str | float | int]] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        parsed = parse_run_dir(child.name)
        if parsed is None:
            continue
        dataset, seed = parsed
        log_path = child / "train.log"
        if not log_path.is_file():
            print(f"WARN: missing {log_path}", file=sys.stderr)
            continue
        metrics = extract_best_results(log_path.read_text(encoding="utf-8", errors="replace"))
        if metrics is None:
            print(f"WARN: no Best Results in {log_path}", file=sys.stderr)
            continue
        row: dict[str, str | float | int] = {
            "dataset": dataset,
            "seed": seed,
            "train_log": str(log_path),
        }
        row.update(metrics)
        rows.append(row)

    if not rows:
        print("ERROR: no mosi_seed*/mosei_seed* rows", file=sys.stderr)
        return 1

    fieldnames = ["dataset", "seed", "mae", "corr", "acc7", "acc2", "f1", "train_log"]
    out_csv = root / "summary_stability_metrics.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda x: (str(x["dataset"]), int(x["seed"]))):
            w.writerow(r)

    print(f"Wrote {len(rows)} rows to {out_csv}\n")
    by_ds: dict[str, list[dict[str, float | int]]] = {}
    for r in rows:
        by_ds.setdefault(str(r["dataset"]), []).append(r)

    metric_keys = [("mae", "MAE", False), ("corr", "Corr", True), ("acc7", "Acc-7", True),
                   ("acc2", "Acc-2", True), ("f1", "F1", True)]

    for ds, ds_rows in sorted(by_ds.items()):
        print(f"=== {ds.upper()} (n={len(ds_rows)}) ===")
        for r in sorted(ds_rows, key=lambda x: int(x["seed"])):
            print(
                f"  seed={r['seed']:4d}  "
                f"MAE={r.get('mae', float('nan')):.4f}  "
                f"Corr={r.get('corr', float('nan')):.4f}  "
                f"Acc-7={r.get('acc7', float('nan')):.2f}  "
                f"Acc-2={r.get('acc2', float('nan')):.2f}  "
                f"F1={r.get('f1', float('nan')):.4f}"
            )
        print("  Mean ± Std (sample stdev):")
        for key, label, _higher_better in metric_keys:
            vals = [float(r[key]) for r in ds_rows if key in r]
            if not vals:
                continue
            mean = statistics.mean(vals)
            std = statistics.stdev(vals) if len(vals) > 1 else 0.0
            print(f"    {label:6s}  {mean:.4f} ± {std:.4f}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
