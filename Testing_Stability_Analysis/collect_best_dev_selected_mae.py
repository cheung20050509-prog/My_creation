#!/usr/bin/env python3
"""Parse train.py final block from each seed run's train.log.

Extracts the **dev-selected** epoch's reported **test** MAE (the ``MAE:`` line under
``Best Results (selection_metric, ...)``), matching ``train.py`` print order.

Usage (cwd arbitrary)::

    python Testing_Stability_Analysis/collect_best_dev_selected_mae.py \\
      My_creation/Testing_Stability_Analysis/runs/multi_seed_20260101_000000

Writes ``summary_dev_selected_test_mae.csv`` next to the run directories.
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics
import sys
from pathlib import Path


def extract_best_results_mae(log_text: str) -> float | None:
    if "Best Results" not in log_text:
        return None
    start = log_text.find("Best Results")
    chunk = log_text[start:]
    le = chunk.find("\nLast Epoch")
    if le != -1:
        chunk = chunk[:le]
    m = re.search(r"^\s+MAE:\s+([0-9.]+)\s*$", chunk, re.MULTILINE)
    return float(m.group(1)) if m else None


def parse_run_dir(name: str) -> tuple[str, int] | None:
    if name.startswith("mosi_seed"):
        return "mosi", int(name[len("mosi_seed") :])
    if name.startswith("mosei_seed"):
        return "mosei", int(name[len("mosei_seed") :])
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "run_root",
        type=Path,
        help="Directory containing mosi_seed*/mosei_seed* subfolders",
    )
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
        text = log_path.read_text(encoding="utf-8", errors="replace")
        mae = extract_best_results_mae(text)
        if mae is None:
            print(f"WARN: could not parse Best Results MAE in {log_path}", file=sys.stderr)
            continue
        rows.append(
            {
                "dataset": dataset,
                "seed": seed,
                "best_dev_selected_test_mae": mae,
                "train_log": str(log_path),
            }
        )

    if not rows:
        print("ERROR: no mosi_seed*/mosei_seed* rows collected", file=sys.stderr)
        return 1

    out_csv = root / "summary_dev_selected_test_mae.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=["dataset", "seed", "best_dev_selected_test_mae", "train_log"],
        )
        w.writeheader()
        for r in sorted(rows, key=lambda x: (str(x["dataset"]), int(x["seed"]))):
            w.writerow(r)

    by_ds: dict[str, list[float]] = {}
    for r in rows:
        by_ds.setdefault(str(r["dataset"]), []).append(float(r["best_dev_selected_test_mae"]))

    print(f"Wrote {len(rows)} rows to {out_csv}")
    for ds, vals in sorted(by_ds.items()):
        mean = statistics.mean(vals)
        std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        print(f"  {ds}: n={len(vals)}  mean_mae={mean:.4f}  std_mae={std:.4f}  min={min(vals):.4f}  max={max(vals):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
