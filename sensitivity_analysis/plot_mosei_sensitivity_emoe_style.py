#!/usr/bin/env python3
"""EMOE-style sensitivity figure for CMU-MOSEI (trial 70 sweeps).

Reads ``summary.csv`` produced by::

    python sensitivity_analysis/run_mosei_trial70_sensitivity.py aggregate \\
        --runs-root runs/sensitivity_mosei/trial70

Visual layout follows EMOE (CVPR 2025) Fig.4: one row of panels, each with
dual y-axes — Acc-7 (%, left, orange + circles) and MAE (right, light blue +
diamonds), with the MAE axis inverted (lower MAE toward the top).

Example::

    cd My_creation
    python3 sensitivity_analysis/run_mosei_trial70_sensitivity.py aggregate \\
        --runs-root runs/sensitivity_mosei/trial70
    python3 sensitivity_analysis/plot_mosei_sensitivity_emoe_style.py \\
        --summary-csv runs/sensitivity_mosei/trial70/summary.csv \\
        --output runs/sensitivity_mosei/trial70/mosei_sensitivity_emoe_style.png
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt

_MY = Path(__file__).resolve().parent.parent
_DEFAULT_SUMMARY = _MY / "runs/sensitivity_mosei/trial70/summary.csv"
_DEFAULT_OUTPUT = _MY / "runs/sensitivity_mosei/trial70/mosei_trial70_sensitivity_emoe_style.png"

DEFAULT_AXES_ORDER = ("beta_ib", "mse_weight", "alpha_ib", "selector_target_temp")

AXIS_TITLE = {
    "beta_ib": r"$\beta_{\mathrm{IB}}$",
    "mse_weight": r"$\mathrm{mse\_weight}$",
    "alpha_ib": r"$\alpha_{\mathrm{IB}}$",
    "selector_target_temp": r"$t_{\mathrm{sel}}$",
}


def _parse_float(s: str) -> float | None:
    s = (s or "").strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def load_rows(summary_csv: Path) -> list[dict[str, str]]:
    if not summary_csv.is_file():
        print(f"ERROR: summary CSV not found: {summary_csv}", file=sys.stderr)
        sys.exit(1)
    with summary_csv.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def rows_for_axis(rows: list[dict[str, str]], axis: str) -> list[tuple[float, float, float]]:
    """Return list of (x_value, acc7_frac, mae) sorted by x."""
    out: list[tuple[float, float, float]] = []
    for row in rows:
        if (row.get("axis") or "").strip() != axis:
            continue
        xv = _parse_float(str(row.get("value", "")))
        acc7 = _parse_float(str(row.get("Acc-7", "")))
        mae = _parse_float(str(row.get("MAE", "")))
        if xv is None or acc7 is None or mae is None:
            print(f"WARN: skip row (axis={axis!r}): {row}", file=sys.stderr)
            continue
        if not (math.isfinite(xv) and math.isfinite(acc7) and math.isfinite(mae)):
            continue
        out.append((xv, acc7, mae))
    out.sort(key=lambda t: t[0])
    return out


def _acc7_ylim(series_pct: list[float]) -> tuple[float, float]:
    if not series_pct:
        return 25.0, 65.0
    lo, hi = min(series_pct), max(series_pct)
    pad = max(2.0, (hi - lo) * 0.15)
    return lo - pad, hi + pad


def _mae_ylim(
    series: list[float],
    lo_hi: tuple[float, float] | None,
    pad: float = 0.02,
) -> tuple[float, float]:
    """Return (high, low) for set_ylim so MAE increases downward (EMOE style)."""
    if lo_hi is not None:
        lo, hi = lo_hi
        return hi, lo
    if not series:
        return 0.7, 0.5
    lo, hi = min(series), max(series)
    return hi + pad, lo - pad


def plot(
    *,
    summary_csv: Path,
    output_png: Path,
    axes_order: tuple[str, ...],
    xlog_axes: set[str],
    mae_ylim: tuple[float, float] | None,
    strict: bool,
) -> None:
    rows = load_rows(summary_csv)
    fig, axs = plt.subplots(1, len(axes_order), figsize=(4.2 * len(axes_order), 3.8), constrained_layout=True)

    if len(axes_order) == 1:
        axs = [axs]

    any_data = False
    for ax, axis in zip(axs, axes_order):
        data = rows_for_axis(rows, axis)
        if not data:
            msg = f"no data for axis={axis}"
            if strict:
                print(f"ERROR: {msg}", file=sys.stderr)
                sys.exit(1)
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(AXIS_TITLE.get(axis, axis))
            continue

        any_data = True
        xs = [d[0] for d in data]
        acc7_pct = [d[1] * 100.0 for d in data]
        maes = [d[2] for d in data]

        color_acc = "#E07020"
        color_mae = "#7EB6D9"

        (line_acc,) = ax.plot(
            xs,
            acc7_pct,
            color=color_acc,
            marker="o",
            linestyle="-",
            linewidth=1.6,
            markersize=6,
            label=r"$ACC_7$",
        )
        ax.set_ylabel(r"$ACC_7$", color=color_acc)
        ax.tick_params(axis="y", labelcolor=color_acc)
        y_lo, y_hi = _acc7_ylim(acc7_pct)
        ax.set_ylim(y_lo, y_hi)
        if axis in xlog_axes:
            ax.set_xscale("log")

        ax2 = ax.twinx()
        (line_mae,) = ax2.plot(
            xs,
            maes,
            color=color_mae,
            marker="D",
            linestyle="-",
            linewidth=1.6,
            markersize=5,
            label="MAE",
        )
        ax2.set_ylabel("MAE", color=color_mae)
        ax2.tick_params(axis="y", labelcolor=color_mae)
        m_hi, m_lo = _mae_ylim(maes, mae_ylim)
        ax2.set_ylim(m_hi, m_lo)

        ax.set_title(AXIS_TITLE.get(axis, axis))
        ax.grid(True, axis="y", alpha=0.35)
        ax.set_xlabel(axis.replace("_", " "))

        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8, framealpha=0.9)

    fig.suptitle(
        "Sensitivity analysis on CMU-MOSEI (trial 70): vary one hyper-parameter, others fixed.",
        fontsize=10,
    )

    if not any_data:
        print("ERROR: no plottable data in any panel", file=sys.stderr)
        sys.exit(1)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=200)
    plt.close(fig)
    print(f"Wrote {output_png}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--summary-csv",
        type=Path,
        default=_DEFAULT_SUMMARY,
        help=f"Path to summary.csv (default: {_DEFAULT_SUMMARY})",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help=f"Output PNG path (default: {_DEFAULT_OUTPUT})",
    )
    ap.add_argument(
        "--axes",
        type=str,
        default=",".join(DEFAULT_AXES_ORDER),
        help="Comma-separated axis names (default: four standard axes)",
    )
    ap.add_argument(
        "--xlog-axes",
        type=str,
        default="",
        help="Comma-separated axes to use log-scale x (e.g. beta_ib)",
    )
    ap.add_argument(
        "--mae-ylim",
        type=str,
        default="",
        help='Inverted-axis data limits as "lo,hi" in MAE space (e.g. 0.5,0.7); empty = auto from data',
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="Exit with error if any requested axis has no rows",
    )
    args = ap.parse_args()

    axes_order = tuple(a.strip() for a in args.axes.split(",") if a.strip())
    if not axes_order:
        print("ERROR: --axes is empty", file=sys.stderr)
        return 2

    xlog = {a.strip() for a in args.xlog_axes.split(",") if a.strip()}

    mae_ylim: tuple[float, float] | None = None
    if args.mae_ylim.strip():
        parts = [p.strip() for p in args.mae_ylim.split(",")]
        if len(parts) != 2:
            print('ERROR: --mae-ylim must be "lo,hi"', file=sys.stderr)
            return 2
        lo, hi = float(parts[0]), float(parts[1])
        mae_ylim = (lo, hi)

    plot(
        summary_csv=args.summary_csv.expanduser().resolve(),
        output_png=args.output.expanduser().resolve(),
        axes_order=axes_order,
        xlog_axes=xlog,
        mae_ylim=mae_ylim,
        strict=args.strict,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
