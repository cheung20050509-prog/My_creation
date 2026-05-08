#!/usr/bin/env python3
"""Dual-panel classification accuracy bar chart (UR-FUNNY v2 / MUSTARD), PRISM vs baselines.

Run: python plot_classification_prism_bars.py
Output: classification_prism_graph.png (same directory as this script).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, Rectangle


def lerp_rgb(c1: str, c2: str, n: int) -> list[tuple[float, float, float]]:
    a = np.array(to_rgb(c1), dtype=float)
    b = np.array(to_rgb(c2), dtype=float)
    t = np.linspace(0.0, 1.0, n)[:, np.newaxis]
    rgb = a * (1.0 - t) + b * t
    return [tuple(row) for row in rgb]


def draw_rounded_bar(
    ax,
    x: float,
    bottom: float,
    width: float,
    height: float,
    color,
    radius_frac: float = 0.08,
) -> None:
    """Vertical bar with flat bottom and semi-elliptical cap (rounded top)."""
    if height <= 0:
        return
    cap_h = min(width * radius_frac * 2.0, height * 0.35, height)
    body_h = height - cap_h
    ax.add_patch(
        Rectangle(
            (x - width / 2, bottom),
            width,
            max(body_h, 1e-6),
            facecolor=color,
            edgecolor="none",
            zorder=2,
        )
    )
    cy = bottom + body_h + cap_h / 2
    ax.add_patch(
        Ellipse(
            (x, cy),
            width,
            cap_h,
            facecolor=color,
            edgecolor="none",
            zorder=3,
        )
    )


def panel(
    ax,
    names: list[str],
    values: list[float],
    colors: list[tuple[float, float, float]],
    bar_width: float = 0.62,
    y_max: float = 80.0,
    dataset_title: str = "",
    legend_ncol: int = 4,
    legend_y: float = 1.12,
    fmt: str = "{:.1f}",
) -> None:
    x = np.arange(len(names))
    for xi, h, c in zip(x, values, colors):
        draw_rounded_bar(ax, float(xi), 0.0, bar_width, h, c)

    ax.set_xticks([])
    ax.set_xlim(-0.55, len(names) - 0.45)
    ax.set_ylim(0, y_max)
    ax.set_yticks(np.arange(0, int(y_max) + 1, 10))
    ax.set_ylabel("Accuracy(%)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for xi, h in zip(x, values):
        ax.text(
            xi,
            h + 1.2,
            fmt.format(h),
            ha="center",
            va="bottom",
            fontsize=9,
            color="#333333",
        )

    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=7, linestyle="None")
        for c in colors
    ]
    ax.legend(
        handles,
        names,
        loc="upper center",
        bbox_to_anchor=(0.5, legend_y),
        ncol=legend_ncol,
        frameon=False,
        fontsize=8,
        handletextpad=0.4,
        columnspacing=0.9,
    )

    ax.text(
        0.5,
        -0.12,
        dataset_title,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=11,
    )


def main() -> None:
    # Baselines + PRISM (last bar); PRISM from Optuna Best Results test Acc (one decimal).
    ur_names = ["C-MFN", "TFN", "MISA", "BBFN", "MAG-XLNet", "MuLoT", "PRISM"]
    ur_vals = [65.2, 64.7, 70.6, 71.7, 72.4, 74.0, 74.5]

    mu_names = ["MISA", "TFN", "MAG-ALBERT", "C-MFN", "BBFN", "ITHP", "PRISM"]
    mu_vals = [66.2, 68.6, 69.1, 70.0, 71.4, 75.3, 75.0]

    purple_light = "#e8dff5"
    purple_dark = "#4a2c6d"
    blue_light = "#dcecf9"
    blue_dark = "#143d6b"

    ur_colors = lerp_rgb(purple_light, purple_dark, len(ur_names))
    mu_colors = lerp_rgb(blue_light, blue_dark, len(mu_names))

    fig_w, fig_h = 11.0, 4.2
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(fig_w, fig_h), dpi=150)
    fig.patch.set_facecolor("white")

    panel(
        ax_l,
        ur_names,
        ur_vals,
        ur_colors,
        dataset_title="UR-FUNNY v2",
        legend_ncol=4,
    )
    panel(
        ax_r,
        mu_names,
        mu_vals,
        mu_colors,
        dataset_title="MUStARD",
        legend_ncol=4,
    )

    plt.subplots_adjust(left=0.07, right=0.98, top=0.82, bottom=0.18, wspace=0.22)

    out = Path(__file__).resolve().parent / "classification_prism_graph.png"
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
