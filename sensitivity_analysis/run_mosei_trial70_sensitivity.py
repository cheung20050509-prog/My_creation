#!/usr/bin/env python3
"""MOSEI phase1 trial 70 hyper-parameter sensitivity: one knob at a time, rest frozen.

Reads frozen defaults from ``fixed_experiment/mosei_phase1_trial70_hparams.py`` (same
float formatting as Optuna / verify script). Writes under ``runs/sensitivity_mosei/`` by default (training checkpoints/logs).
Use ``aggregate --summary-out sensitivity_analysis/results/mosei_trial70/summary.csv``
to place the summary table next to figures under this package.

Commands:
  list-values --axis NAME          Print default grid (one value per line, stdout).
  train --axis NAME --value FLOAT  Run fixed_experiment/train.py once.
  aggregate --runs-root DIR        Scan **/train.log, write summary.csv next to runs-root.

Supported axes include ``selector_rib_weight`` (routing supervision scale; default grid
includes 0 for no ``L_rib`` term).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

# Import trial-70 hparams from fixed_experiment (single source of truth).
_MY = Path(__file__).resolve().parent.parent
_FIXED = _MY / "fixed_experiment"
if str(_FIXED) not in sys.path:
    sys.path.insert(0, str(_FIXED))
from mosei_phase1_trial70_hparams import (  # noqa: E402
    GRADIENT_ACCUMULATION_STEP,
    IB_HIDDEN_DIM,
    SELECTION_METRIC,
    TRAIN_BATCH_SIZE,
    TRIAL_70_PARAMS,
    format_train_float_argv,
)

TRAIN_PY = _FIXED / "train.py"

RESULT_LINE_RE = re.compile(
    r"\s+(Selection score|Acc-2|Acc-7|Acc-5|Acc-3|MAE|Corr|F1):\s+([\d.]+)%?"
)

SUPPORTED_AXES = (
    "beta_ib",
    "mse_weight",
    "alpha_ib",
    "selector_target_temp",
    "selector_rib_weight",
)


def default_grid(axis: str) -> list[float]:
    p = TRIAL_70_PARAMS
    b0 = float(p["beta_ib"])
    m0 = float(p["mse_weight"])
    a0 = float(p["alpha_ib"])
    t0 = float(p["selector_target_temp"])
    w0 = float(p["selector_rib_weight"])
    if axis == "beta_ib":
        return sorted({b0 * f for f in (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)})
    if axis == "mse_weight":
        return sorted({round(x, 6) for x in (0.5, 1.0, 1.5, m0, 2.5, 3.0, 4.0)})
    if axis == "alpha_ib":
        return [0.001, 0.005, a0, 0.02, 0.05, 0.1]
    if axis == "selector_target_temp":
        # Skip extremely small t (EMOE-style router collapse); still include 0.1.
        return [0.1, 0.2, t0, 0.5, 0.7, 1.0]
    if axis == "selector_rib_weight":
        # 0 = no L_rib term; then band around trial70 default (typ. 0.05), up to ~Optuna max.
        return sorted(
            {
                round(x, 6)
                for x in (0.0, 0.01, w0 * 0.5, w0, w0 * 1.5, w0 * 2.0, 0.15)
            }
        )
    raise ValueError(f"unsupported axis: {axis}")


def value_tag(axis: str, value: float) -> str:
    """Directory-safe token (no dots)."""
    if axis == "alpha_ib":
        s = f"{float(value):.6f}".rstrip("0").rstrip(".")
    else:
        s = f"{float(value):.6g}"
    return f"v{s.replace('.', 'p').replace('-', 'm')}"


def build_train_argv(*, checkpoint_dir: str, overrides: dict[str, float]) -> list[str]:
    p = {**TRIAL_70_PARAMS, **overrides}
    fmt = format_train_float_argv(p)
    ne = int(p["n_epochs"])
    return [
        "--dataset",
        "mosei",
        "--n_epochs",
        str(ne),
        "--train_batch_size",
        str(TRAIN_BATCH_SIZE),
        "--gradient_accumulation_step",
        str(GRADIENT_ACCUMULATION_STEP),
        "--checkpoint_dir",
        checkpoint_dir,
        "--selection_metric",
        SELECTION_METRIC,
        "--seed",
        str(int(p["seed"])),
        "--learning_rate",
        fmt["learning_rate"],
        "--ig_learning_rate",
        fmt["ig_learning_rate"],
        "--beta_ib",
        fmt["beta_ib"],
        "--num_infogate_layers",
        str(int(p["num_infogate_layers"])),
        "--bottleneck_dim",
        str(int(p["bottleneck_dim"])),
        "--mse_weight",
        fmt["mse_weight"],
        "--dropout_prob",
        fmt["dropout_prob"],
        "--alpha_ib",
        fmt["alpha_ib"],
        "--stage1_epochs",
        str(int(p["stage1_epochs"])),
        "--warmup_proportion",
        fmt["warmup_proportion"],
        "--weight_decay",
        fmt["weight_decay"],
        "--ema_decay",
        fmt["ema_decay"],
        "--selector_target_temp",
        fmt["selector_target_temp"],
        "--selector_rib_weight",
        fmt["selector_rib_weight"],
        "--align_mix_floor",
        fmt["align_mix_floor"],
        "--gumbel_tau_start",
        fmt["gumbel_tau_start"],
        "--gumbel_tau_end",
        fmt["gumbel_tau_end"],
        "--num_heads",
        str(int(p["num_heads"])),
        "--unified_dim",
        str(int(p["unified_dim"])),
        "--ib_hidden_dim",
        str(IB_HIDDEN_DIM),
        "--ema_start_epoch",
        str(int(p["ema_start_epoch"])),
    ]


def parse_best_results(log_path: Path) -> dict[str, float]:
    """Same contract as ``optuna_search_v2.parse_best_results`` (MOSEI test block)."""
    results: dict[str, float] = {}
    if not log_path.is_file():
        return results
    in_block = False
    text = log_path.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        if line.startswith("Best Results"):
            in_block = True
            continue
        if in_block and line.startswith("Last Epoch"):
            break
        if in_block:
            m = RESULT_LINE_RE.match(line)
            if not m:
                continue
            key, raw = m.groups()
            val = float(raw)
            if key in ("Acc-2", "Acc-7", "Acc-5", "Acc-3"):
                val /= 100.0
            if key == "Selection score":
                key = "SelectionScore"
            results[key] = val
    return results


def cmd_list_values(axis: str) -> int:
    for v in default_grid(axis):
        print(v)
    return 0


def cmd_train(axis: str, value: float, runs_root: Path, dry_run: bool) -> int:
    if axis not in SUPPORTED_AXES:
        print(f"ERROR: axis must be one of {SUPPORTED_AXES}", file=sys.stderr)
        return 2
    overrides = {axis: float(value)}
    out_dir = runs_root / axis / value_tag(axis, float(value))
    ckpt = out_dir / "checkpoints"
    argv = build_train_argv(checkpoint_dir=str(ckpt), overrides=overrides)
    cmd = [sys.executable, "-u", str(TRAIN_PY), *argv]
    if dry_run:
        print("cwd:", _MY)
        print("out: ", out_dir)
        print("exec:", " ".join(cmd))
        return 0
    ckpt.mkdir(parents=True, exist_ok=True)
    meta = {"axis": axis, "value": float(value)}
    (out_dir / "run_meta.json").write_text(json.dumps(meta) + "\n", encoding="utf-8")
    log_path = out_dir / "train.log"
    with log_path.open("w", encoding="utf-8") as logf:
        subprocess.run(cmd, cwd=str(_MY), stdout=logf, stderr=subprocess.STDOUT, check=True)
    return 0


def cmd_aggregate(runs_root: Path, summary_out: Path | None) -> int:
    rows: list[dict[str, object]] = []
    for log_path in sorted(runs_root.glob("*/*/train.log")):
        rel = log_path.relative_to(runs_root)
        parts = rel.parts
        if len(parts) < 3:
            continue
        axis = parts[0]
        tag = parts[1]
        run_dir = log_path.parent
        meta_path = run_dir / "run_meta.json"
        value_str = ""
        if meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                value_str = str(meta.get("value", ""))
                if meta.get("axis"):
                    axis = str(meta["axis"])
            except (json.JSONDecodeError, OSError):
                pass
        if not value_str and tag.startswith("v"):
            value_str = tag[1:].replace("p", ".").replace("m", "-")
        metrics = parse_best_results(log_path)
        row: dict[str, object] = {
            "axis": axis,
            "value": value_str,
            "tag": tag,
            "log_path": str(log_path),
        }
        row.update({k: metrics.get(k, "") for k in ("MAE", "Corr", "F1", "Acc-2", "Acc-7", "SelectionScore")})
        rows.append(row)

    summary = summary_out.expanduser().resolve() if summary_out else (runs_root / "summary.csv")
    summary.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "axis",
        "value",
        "tag",
        "MAE",
        "Corr",
        "F1",
        "Acc-2",
        "Acc-7",
        "SelectionScore",
        "log_path",
    ]
    with summary.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            for fn in fieldnames:
                r.setdefault(fn, "")
            w.writerow(r)
    print(f"Wrote {summary} ({len(rows)} runs)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list-values", help="Print default grid for --axis (one per line)")
    p_list.add_argument("--axis", required=True, choices=SUPPORTED_AXES)

    p_train = sub.add_parser("train", help="Run one training job")
    p_train.add_argument("--axis", required=True, choices=SUPPORTED_AXES)
    p_train.add_argument("--value", type=float, required=True)
    p_train.add_argument(
        "--runs-root",
        type=Path,
        default=_MY / "runs/sensitivity_mosei/trial70",
        help="Parent directory; writes runs-root/AXIS/TAG/",
    )
    p_train.add_argument("--dry-run", action="store_true")

    p_agg = sub.add_parser("aggregate", help="Collect Best Results from train.log under runs-root")
    p_agg.add_argument(
        "--runs-root",
        type=Path,
        required=True,
        help="Directory containing AXIS/TAG/train.log",
    )
    p_agg.add_argument(
        "--summary-out",
        type=Path,
        default=None,
        help=(
            "Write summary.csv here (default: <runs-root>/summary.csv). "
            "Use e.g. sensitivity_analysis/results/mosei_trial70/summary.csv to keep "
            "tables under this package."
        ),
    )

    args = ap.parse_args()
    if args.command == "list-values":
        return cmd_list_values(args.axis)
    if args.command == "train":
        return cmd_train(args.axis, args.value, args.runs_root.expanduser().resolve(), args.dry_run)
    if args.command == "aggregate":
        so = args.summary_out.expanduser().resolve() if args.summary_out else None
        return cmd_aggregate(args.runs_root.expanduser().resolve(), so)
    return 1


if __name__ == "__main__":
    sys.exit(main())
