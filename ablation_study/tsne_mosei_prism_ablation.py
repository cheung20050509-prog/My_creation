#!/usr/bin/env python3
"""t-SNE on CMU-MOSEI test ``h_p`` for PRISM vs VTB/DPR/InfoGate macro ablations.

Loads frozen phase1 trial~70 checkpoints under ``ablation_study/runs/`` (same layout
as ``train_fixed_mosei_phase1_trial70.py``), extracts pooled primary embeddings
before the regression head, fits t-SNE **per variant** (independent 2D or 3D spaces).

**Default coloring:** discrete **7 bins** aligned with ``train.py`` Acc-7:
``clip(round(y), -3, 3)`` → ordinal 0..6. Uses a **high-contrast solid palette**
(no ``viridis``, no continuous gradient by default). **Markers** group polarity:
▼ bins −3..−1, ● bin 0, ▲ bins +1..+3. **Legend** lists all seven levels with
color + shape; optional per-panel bin counts in subtitles.

**Optional ``--color-mode ternary``:** collapse Acc-7 to **negative / neutral /
positive** (bins −3..−1 / 0 / +1..+3) with a **paper-style** look: teal ``×``
(``x`` marker) / solid vivid blue filled ``o`` (no contrasting rim) / magenta ``+``,
**per-panel** legend (lower right), and **min–max normalized** axes to ``[0,1]^2`` or ``[0,1]^3``
when ``--n-components 3``, with 0.2 ticks (each panel scaled independently).

Usage (from ``My_creation/``)::

    conda run -n ITHP5090 python ablation_study/tsne_mosei_prism_ablation.py

    python ablation_study/tsne_mosei_prism_ablation.py --verbose \\
      --dump-label-stats ablation_study/runs/mosei_phase1_trial70_tsne/label_stats.csv

    # Sharper PNG (default ``--dpi 480``, ``--fig-scale 1.35``); e.g. poster:
    python ablation_study/tsne_mosei_prism_ablation.py --dpi 600 --fig-scale 1.6 \\
      --output ablation_study/runs/mosei_phase1_trial70_tsne/ablation_tsne_prism_mosei.png

    # Reproduce tuned 2D ternary figure (perplexity 20, early exaggeration 24):
    python ablation_study/tsne_mosei_prism_ablation.py --n-components 2 --color-mode ternary \\
      --perplexity 20 --early-exaggeration 24 --init pca --learning-rate auto \\
      --output ablation_study/runs/mosei_phase1_trial70_tsne/ablation_tsne_prism_mosei_fig3style.png

    # 3D t-SNE (static matplotlib view); use ``--n-components 2`` for classic 2D:
    python ablation_study/tsne_mosei_prism_ablation.py --n-components 3 --color-mode ternary \\
      --output ablation_study/runs/mosei_phase1_trial70_tsne/ablation_tsne_prism_mosei_fig3style_3d.png

Requires trained ``infogate_mosei_best.pt`` in each run directory.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import math
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split

_HERE = Path(__file__).resolve().parent
_MY = _HERE.parent

# Seven solid colors: strong blue → … → strong red (ColorBrewer-style diverging, not viridis).
DISCRETE7_COLORS = [
    "#053061",
    "#2166ac",
    "#4393c3",
    "#92c5de",
    "#fddbc7",
    "#d6604d",
    "#67001f",
]
ACC7_LABELS = [r"$-3$", r"$-2$", r"$-1$", r"$0$", r"$+1$", r"$+2$", r"$+3$"]
# One marker per Acc-7 bin: negative / neutral / positive (still seven colors).
ACC7_MARKERS = ("v", "v", "v", "o", "^", "^", "^")
# Ternary sentiment (paper-style): marker + colors (Acc-7 collapsed).
# Neutral uses one solid vivid blue (no separate dark rim on ``o`` markers).
TERNARY_NEUTRAL_BLUE = "#0066CC"
TERNARY_CLASSES: tuple[tuple[str, str, str, str], ...] = (
    ("negative", "negative", "x", "#20B2AA"),  # teal ×
    ("neutral", "neutral", "o", TERNARY_NEUTRAL_BLUE),
    ("positive", "positive", "+", "#ae017e"),  # magenta +
)


def acc7_bin_index(y: np.ndarray) -> np.ndarray:
    """Match ``train.py`` ``score()`` Acc-7 binning: clip(round(y), -3, 3) → 0..6."""
    b = np.clip(np.rint(y), -3, 3).astype(np.int32)
    return b + 3


def _label_stats(y: np.ndarray) -> dict[str, float | int]:
    y = np.asarray(y, dtype=np.float64).ravel()
    n = int(y.size)
    if n == 0:
        return {"n": 0}
    qs = np.percentile(y, [5, 25, 50, 75, 95])
    return {
        "n": n,
        "min": float(y.min()),
        "max": float(y.max()),
        "mean": float(y.mean()),
        "std": float(y.std()),
        "p5": float(qs[0]),
        "p25": float(qs[1]),
        "p50": float(qs[2]),
        "p75": float(qs[3]),
        "p95": float(qs[4]),
        "frac_lt0": float((y < 0).mean()),
        "frac_lt_neg1": float((y < -1).mean()),
    }


def _print_label_stats(variant: str, st: dict[str, float | int], counts: np.ndarray) -> None:
    if st.get("n", 0) == 0:
        print(f"[{variant}] no samples")
        return
    print(
        f"[{variant}] n={st['n']}  y∈[{st['min']:.3f},{st['max']:.3f}]  "
        f"mean={st['mean']:.3f} std={st['std']:.3f}  "
        f"p5/50/95={st['p5']:.3f}/{st['p50']:.3f}/{st['p95']:.3f}  "
        f"P(y<0)={100*st['frac_lt0']:.1f}%  P(y<-1)={100*st['frac_lt_neg1']:.1f}%"
    )
    print(f"    Acc-7 bin counts [-3..+3]: {counts.tolist()}")


def _panel_overlap_note(n: int, counts: np.ndarray, st: dict[str, float | int]) -> str:
    """Short on-figure reminder: label mass vs. 2D stacking (t-SNE is not density-preserving)."""
    if n == 0 or st.get("n", 0) == 0:
        return ""
    c = counts.astype(np.int64)
    neg123 = int(c[:3].sum())
    neg123_pct = 100.0 * neg123 / n
    return (
        f"n={n}\n"
        f"P(y<0)={100 * float(st['frac_lt0']):.1f}%  "
        f"bins −3..−1: {neg123} ({neg123_pct:.1f}%)\n"
        "Low-D projection stacks many points"
    )


def _append_label_stats_csv(
    path: Path,
    variant: str,
    st: dict[str, float | int],
    counts: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "variant",
        "n",
        "min",
        "max",
        "mean",
        "std",
        "p5",
        "p25",
        "p50",
        "p75",
        "p95",
        "frac_lt0",
        "frac_lt_neg1",
        "c_m3",
        "c_m2",
        "c_m1",
        "c0",
        "c1",
        "c2",
        "c3",
    ]
    new_file = not path.is_file()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if new_file:
            w.writeheader()
        row: dict[str, object] = {
            "variant": variant,
            "n": st.get("n", ""),
            "min": st.get("min", ""),
            "max": st.get("max", ""),
            "mean": st.get("mean", ""),
            "std": st.get("std", ""),
            "p5": st.get("p5", ""),
            "p25": st.get("p25", ""),
            "p50": st.get("p50", ""),
            "p75": st.get("p75", ""),
            "p95": st.get("p95", ""),
            "frac_lt0": st.get("frac_lt0", ""),
            "frac_lt_neg1": st.get("frac_lt_neg1", ""),
        }
        for i, name in enumerate(["c_m3", "c_m2", "c_m1", "c0", "c1", "c2", "c3"]):
            row[name] = int(counts[i]) if i < len(counts) else 0
        w.writerow(row)


DEFAULT_VARIANTS: tuple[tuple[str, str], ...] = (
    ("none", "PRISM (full)"),
    ("no_ib", "w/o VTB"),
    ("no_mselector", "w/o DPR"),
    ("no_infogate", "w/o InfoGate"),
    ("no_ib_no_mselector_no_infogate", "w/o VTB+DPR+InfoGate"),
)


def _default_ckpt_dir(runs_root: Path, ablation: str) -> Path:
    if ablation == "none":
        return runs_root / "mosei_phase1_trial70" / "checkpoints"
    return runs_root / f"mosei_phase1_trial70_{ablation}" / "checkpoints"


def _configure_train_argv(ablation: str, checkpoint_dir: Path) -> None:
    from mosei_phase1_trial70_hparams import build_train_argv

    sys.argv = ["train.py", *build_train_argv(checkpoint_dir=str(checkpoint_dir), ablation=ablation)]


def _collect_h_p(train_mod, max_samples: int | None, stage: int) -> tuple[np.ndarray, np.ndarray]:
    train_mod.set_seed(train_mod.args.seed)
    _, _, test_dl, n_opt = train_mod.setup_data()
    model, _, _ = train_mod.build_model(n_opt)
    ckpt_path = os.path.join(train_mod.args.checkpoint_dir, f"infogate_{train_mod.args.dataset}_best.pt")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    try:
        blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        blob = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(blob["model_state_dict"], strict=True)
    model.to(train_mod.DEVICE)
    model.eval()

    hs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    n_seen = 0
    with torch.no_grad():
        for batch in test_dl:
            batch = tuple(t.to(train_mod.DEVICE) for t in batch)
            input_ids, visual, acoustic, label_ids = batch
            visual = visual.squeeze(1)
            acoustic = acoustic.squeeze(1)
            _, _, _, h_p = model(input_ids, visual, acoustic, labels=label_ids, stage=stage)
            hs.append(h_p.detach().float().cpu().numpy())
            ys.append(label_ids.detach().cpu().numpy().reshape(-1))
            n_seen += h_p.shape[0]
            if max_samples is not None and n_seen >= max_samples:
                break

    H = np.concatenate(hs, axis=0)
    y = np.concatenate(ys, axis=0)
    if max_samples is not None and H.shape[0] > max_samples:
        H = H[:max_samples]
        y = y[:max_samples]
    return H, y


def _stratified_acc7_indices(idx: np.ndarray, cap: int, seed: int) -> np.ndarray:
    """Row indices for a roughly Acc-7–balanced subset of size ``cap`` (same rows across variants)."""
    n = int(idx.size)
    cap = min(cap, n)
    if cap >= n:
        return np.arange(n, dtype=np.int64)
    ar = np.arange(n, dtype=np.int64)
    try:
        tr, _ = train_test_split(
            ar,
            train_size=cap,
            stratify=idx,
            random_state=seed,
            shuffle=True,
        )
        return np.asarray(tr, dtype=np.int64)
    except ValueError:
        rng = np.random.default_rng(seed)
        return np.sort(rng.choice(n, size=cap, replace=False))


def _unit_box_normalize(x: np.ndarray) -> np.ndarray:
    """Min–max each coordinate axis to [0, 1] (per panel; 2D or 3D)."""
    x = np.asarray(x, dtype=np.float64)
    lo = x.min(axis=0)
    hi = x.max(axis=0)
    span = np.maximum(hi - lo, 1e-9)
    return (x - lo) / span


def _ternary_masks(idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Negative / neutral / positive masks from Acc-7 bin indices 0..6."""
    neg = idx < 3
    neu = idx == 3
    pos = idx > 3
    return neg, neu, pos


def _ternary_legend_handles() -> list[Line2D]:
    handles: list[Line2D] = []
    for _key, label, marker, color in TERNARY_CLASSES:
        if marker == "o":
            handles.append(
                Line2D(
                    [0],
                    [0],
                    linestyle="None",
                    marker=marker,
                    color="none",
                    markerfacecolor=color,
                    markeredgecolor=color,
                    markeredgewidth=0.0,
                    markersize=7.0,
                    label=label,
                )
            )
        else:
            handles.append(
                Line2D(
                    [0],
                    [0],
                    linestyle="None",
                    marker=marker,
                    color="none",
                    markerfacecolor="none",
                    markeredgecolor=color,
                    markeredgewidth=1.15,
                    markersize=8.0,
                    label=label,
                )
            )
    return handles


def _run_tsne(
    H: np.ndarray,
    perplexity: float,
    seed: int,
    n_components: int,
    *,
    early_exaggeration: float,
    learning_rate: str | float,
    init: str,
) -> np.ndarray:
    n = H.shape[0]
    if n < 4:
        raise ValueError(f"Need at least 4 samples for t-SNE, got {n}")
    if n_components not in (2, 3):
        raise ValueError(f"n_components must be 2 or 3, got {n_components}")
    perp = min(perplexity, float(n - 1))
    perp = max(perp, 2.0)
    if early_exaggeration <= 0:
        raise ValueError(f"early_exaggeration must be positive, got {early_exaggeration}")
    # sklearn defaults: max_iter=1000; Barnes–Hut allows dim 2–3.
    return TSNE(
        n_components=n_components,
        perplexity=perp,
        random_state=seed,
        early_exaggeration=early_exaggeration,
        learning_rate=learning_rate,
        init=init,
    ).fit_transform(H.astype(np.float64))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--runs-root",
        type=Path,
        default=_MY / "ablation_study" / "runs",
        help="Parent of mosei_phase1_trial70[_<ablation>]/checkpoints",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=_MY / "ablation_study" / "runs" / "mosei_phase1_trial70_tsne" / "ablation_tsne_prism_mosei.png",
        help="Output figure stem; writes both .png and .pdf",
    )
    ap.add_argument("--perplexity", type=float, default=30.0)
    ap.add_argument(
        "--early-exaggeration",
        type=float,
        default=24.0,
        help="t-SNE early exaggeration; larger often spreads clusters more (sklearn default is 12).",
    )
    ap.add_argument(
        "--learning-rate",
        type=str,
        default="auto",
        help='t-SNE learning rate: ``auto`` (sklearn default) or a positive float, e.g. ``200``.',
    )
    ap.add_argument(
        "--init",
        type=str,
        default="pca",
        choices=("pca", "random"),
        help="t-SNE embedding initialization.",
    )
    ap.add_argument(
        "--n-components",
        type=int,
        choices=(2, 3),
        default=2,
        help="t-SNE output dimension: 2 (flat scatter) or 3 (matplotlib 3D; fixed view angle).",
    )
    ap.add_argument("--seed", type=int, default=42, help="t-SNE ``random_state``")
    ap.add_argument("--stage", type=int, default=2, choices=(1, 2), help="PRISM forward stage for eval")
    ap.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Cap test samples per variant (for quick smoke tests)",
    )
    ap.add_argument(
        "--variants",
        type=str,
        default="none,no_ib,no_mselector,no_infogate",
        help="Comma-separated ablation flags matching train.py (order = subplot order)",
    )
    ap.add_argument(
        "--color-mode",
        choices=("discrete7", "ternary", "continuous"),
        default="discrete7",
        help=(
            "discrete7: Acc-7 colors + ▼/●/▲ (default). "
            "ternary: neg/neu/pos collapsed + paper-style ×/o/+; with ``--n-components 3``, "
            "axes are min–max scaled to [0,1]³. "
            "continuous: RdBu_r + colorbar (debug)."
        ),
    )
    ap.add_argument(
        "--marker-size",
        type=float,
        default=1.35,
        help="Scatter marker area scale (matplotlib ``s``); larger = bigger points.",
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-variant label distribution and Acc-7 bin counts",
    )
    ap.add_argument(
        "--dump-label-stats",
        type=Path,
        default=None,
        help="Append one CSV row per variant with label stats and bin counts",
    )
    ap.add_argument(
        "--dpi",
        type=int,
        default=480,
        help="PNG export resolution (dots per inch). PDF stays vector.",
    )
    ap.add_argument(
        "--fig-scale",
        type=float,
        default=1.35,
        help="Scale default figure width/height; scatter ``s`` scales ~fig_scale² so marks stay visible.",
    )
    args = ap.parse_args()

    lr_raw = str(args.learning_rate).strip().lower()
    if lr_raw == "auto":
        learning_rate: str | float = "auto"
    else:
        try:
            learning_rate = float(lr_raw)
        except ValueError:
            print("error: --learning-rate must be ``auto`` or a float", file=sys.stderr)
            return 2
        if learning_rate <= 0:
            print("error: --learning-rate float must be positive", file=sys.stderr)
            return 2

    runs_root = args.runs_root.expanduser().resolve()
    pairs: list[tuple[str, str]] = []
    for tok in args.variants.split(","):
        tok = tok.strip()
        if not tok:
            continue
        title = next((t for a, t in DEFAULT_VARIANTS if a == tok), tok)
        pairs.append((tok, title))

    os.chdir(_MY)
    if str(_HERE) not in sys.path:
        sys.path.insert(0, str(_HERE))

    n_panels = len(pairs)
    # Two columns → 2×2 for the default four PRISM variants; ⌈n/2⌉ rows in general.
    ncols = 2 if n_panels > 1 else 1
    nrows = int(math.ceil(n_panels / ncols))
    fs = float(args.fig_scale)
    if not (fs > 0.0):
        print("error: --fig-scale must be positive", file=sys.stderr)
        return 2
    if not (72 <= int(args.dpi) <= 1200):
        print("error: --dpi should be between 72 and 1200", file=sys.stderr)
        return 2
    # Area-based scatter ``s`` scales ~fs² so point size matches larger axes.
    s_area_scale = fs * fs
    n_comp = int(args.n_components)
    fig_w = (3.4 * ncols + 0.8) * fs
    fig_h = (3.2 * nrows + 1.4) * fs
    if n_comp == 3:
        fig_h *= 1.12
    slots = nrows * ncols
    if n_comp == 3:
        fig = plt.figure(figsize=(fig_w, fig_h))
        axes_flat = [fig.add_subplot(nrows, ncols, k + 1, projection="3d") for k in range(slots)]
    else:
        fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), constrained_layout=False)
        if n_panels == 1:
            axes_flat = [axes]
        else:
            axes_flat = np.atleast_1d(axes).ravel()

    train_mod = None
    legend_handles: list[Line2D] | None = None
    common_n: int | None = None
    ternary_legend_handles = _ternary_legend_handles() if args.color_mode == "ternary" else None

    for panel_i, (ax, (ablation, title)) in enumerate(zip(axes_flat, pairs)):
        ck_dir = _default_ckpt_dir(runs_root, ablation)
        _configure_train_argv(ablation, ck_dir)
        if train_mod is None:
            import train as train_mod
        else:
            importlib.reload(train_mod)

        H, y = _collect_h_p(train_mod, args.max_samples, args.stage)
        st = _label_stats(y)
        idx = acc7_bin_index(y)
        counts = np.bincount(idx, minlength=7)

        if args.verbose:
            _print_label_stats(ablation, st, counts)
        if args.dump_label_stats is not None:
            _append_label_stats_csv(args.dump_label_stats.expanduser().resolve(), ablation, st, counts)

        if common_n is None:
            common_n = int(st.get("n", 0))

        emb = _run_tsne(
            H,
            args.perplexity,
            args.seed,
            n_comp,
            early_exaggeration=float(args.early_exaggeration),
            learning_rate=learning_rate,
            init=str(args.init),
        )

        sc = None
        if args.color_mode == "ternary":
            pos_plot = _unit_box_normalize(emb)
            neg_m, neu_m, pos_m = _ternary_masks(idx)
            n_neg, n_neu, n_pos = int(neg_m.sum()), int(neu_m.sum()), int(pos_m.sum())
            # Dense test sets: keep ``s`` small to limit ink / overplotting; ×/+ stay legible via lw.
            ms = float(args.marker_size) * s_area_scale
            # Keep ternary marks small for ~4k points; avoid a large ``max(..., s_area_scale)`` floor.
            ternary_s = max(0.35 * s_area_scale, min(7.5, ms * 0.52))
            for _key, _label, marker, color in TERNARY_CLASSES:
                if _key == "negative":
                    sel, lw = neg_m, 0.55
                elif _key == "neutral":
                    sel = neu_m
                else:
                    sel, lw = pos_m, 0.5
                if not np.any(sel):
                    continue
                if n_comp == 3:
                    if marker == "o":
                        sc = ax.scatter(
                            pos_plot[sel, 0],
                            pos_plot[sel, 1],
                            pos_plot[sel, 2],
                            c=[color],
                            marker=marker,
                            s=ternary_s,
                            alpha=0.52,
                            linewidths=0,
                            edgecolors="none",
                        )
                    else:
                        sc = ax.scatter(
                            pos_plot[sel, 0],
                            pos_plot[sel, 1],
                            pos_plot[sel, 2],
                            c=[color],
                            marker=marker,
                            s=ternary_s,
                            alpha=0.52,
                            linewidths=lw,
                        )
                elif marker == "o":
                    sc = ax.scatter(
                        pos_plot[sel, 0],
                        pos_plot[sel, 1],
                        c=[color],
                        marker=marker,
                        s=ternary_s,
                        alpha=0.52,
                        linewidths=0,
                        edgecolors="none",
                    )
                else:
                    sc = ax.scatter(
                        pos_plot[sel, 0],
                        pos_plot[sel, 1],
                        c=[color],
                        marker=marker,
                        s=ternary_s,
                        alpha=0.52,
                        linewidths=lw,
                    )
            panel_tag = f"({chr(ord('a') + panel_i)}) "
            count_line = f"neg {n_neg} | neu {n_neu} | pos {n_pos}"
            ax.set_title(f"{panel_tag}{title}\n{count_line}", fontsize=10, ma="left")
            ticks = np.arange(0.0, 1.01, 0.2)
            if n_comp == 3:
                ax.set_xlim(0.0, 1.0)
                ax.set_ylim(0.0, 1.0)
                ax.set_zlim(0.0, 1.0)
                ax.set_xticks(ticks)
                ax.set_yticks(ticks)
                ax.set_zticks(ticks)
                ax.tick_params(axis="both", labelsize=7)
                try:
                    ax.set_box_aspect((1, 1, 1))
                except AttributeError:
                    pass
                ax.view_init(elev=22, azim=-58)
            else:
                ax.set_xlim(0.0, 1.0)
                ax.set_ylim(0.0, 1.0)
                ax.set_xticks(ticks)
                ax.set_yticks(ticks)
                ax.tick_params(axis="both", labelsize=8)
            if ternary_legend_handles is not None:
                leg = ax.legend(
                    handles=ternary_legend_handles,
                    loc="lower right",
                    fontsize=9,
                    frameon=True,
                    fancybox=False,
                    edgecolor="black",
                    facecolor="white",
                )
                leg.get_frame().set_linewidth(0.6)
        elif args.color_mode == "discrete7":
            for i in range(7):
                sel = idx == i
                if not np.any(sel):
                    continue
                if n_comp == 3:
                    sc = ax.scatter(
                        emb[sel, 0],
                        emb[sel, 1],
                        emb[sel, 2],
                        c=[DISCRETE7_COLORS[i]],
                        marker=ACC7_MARKERS[i],
                        s=args.marker_size * s_area_scale,
                        alpha=0.52,
                        linewidths=0.1,
                        edgecolors="#2a2a2a",
                    )
                else:
                    sc = ax.scatter(
                        emb[sel, 0],
                        emb[sel, 1],
                        c=[DISCRETE7_COLORS[i]],
                        marker=ACC7_MARKERS[i],
                        s=args.marker_size * s_area_scale,
                        alpha=0.52,
                        linewidths=0.1,
                        edgecolors="#2a2a2a",
                    )
            if n_comp == 3:
                try:
                    ax.set_box_aspect((1, 1, 1))
                except AttributeError:
                    pass
                ax.view_init(elev=22, azim=-58)
            if legend_handles is None:
                legend_handles = [
                    Line2D(
                        [0],
                        [0],
                        linestyle="None",
                        marker=ACC7_MARKERS[i],
                        color="none",
                        markerfacecolor=DISCRETE7_COLORS[i],
                        markeredgecolor="#333333",
                        markeredgewidth=0.35,
                        markersize=5.0 * fs,
                        label=ACC7_LABELS[i],
                    )
                    for i in range(7)
                ]
            bin_line = " ".join(f"{i - 3}:{int(counts[i])}" for i in range(7))
            ax.set_title(f"{title}\n{bin_line}", fontsize=10, ma="left")
            note = _panel_overlap_note(int(st.get("n", 0)), counts, st)
            if note:
                if n_comp == 3:
                    ax.text2D(
                        0.02,
                        0.98,
                        note,
                        transform=ax.transAxes,
                        fontsize=6.5,
                        va="top",
                        ha="left",
                        linespacing=1.15,
                        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#bbbbbb", alpha=0.88),
                    )
                else:
                    ax.text(
                        0.02,
                        0.98,
                        note,
                        transform=ax.transAxes,
                        fontsize=6.5,
                        va="top",
                        ha="left",
                        linespacing=1.15,
                        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#bbbbbb", alpha=0.88),
                    )
            ax.set_xticks([])
            ax.set_yticks([])
            if n_comp == 3:
                ax.set_zticks([])
        else:
            for i in range(7):
                sel = idx == i
                if not np.any(sel):
                    continue
                if n_comp == 3:
                    sc = ax.scatter(
                        emb[sel, 0],
                        emb[sel, 1],
                        emb[sel, 2],
                        c=y[sel],
                        cmap="RdBu_r",
                        marker=ACC7_MARKERS[i],
                        s=args.marker_size * s_area_scale,
                        alpha=0.52,
                        linewidths=0.1,
                        edgecolors="#2a2a2a",
                        vmin=-3,
                        vmax=3,
                    )
                else:
                    sc = ax.scatter(
                        emb[sel, 0],
                        emb[sel, 1],
                        c=y[sel],
                        cmap="RdBu_r",
                        marker=ACC7_MARKERS[i],
                        s=args.marker_size * s_area_scale,
                        alpha=0.52,
                        linewidths=0.1,
                        edgecolors="#2a2a2a",
                        vmin=-3,
                        vmax=3,
                    )
            if n_comp == 3:
                try:
                    ax.set_box_aspect((1, 1, 1))
                except AttributeError:
                    pass
                ax.view_init(elev=22, azim=-58)
            ax.set_title(f"{title}\n(continuous debug)", fontsize=10)
            note = _panel_overlap_note(int(st.get("n", 0)), counts, st)
            if note:
                if n_comp == 3:
                    ax.text2D(
                        0.02,
                        0.98,
                        note,
                        transform=ax.transAxes,
                        fontsize=6.5,
                        va="top",
                        ha="left",
                        linespacing=1.15,
                        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#bbbbbb", alpha=0.88),
                    )
                else:
                    ax.text(
                        0.02,
                        0.98,
                        note,
                        transform=ax.transAxes,
                        fontsize=6.5,
                        va="top",
                        ha="left",
                        linespacing=1.15,
                        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#bbbbbb", alpha=0.88),
                    )
            ax.set_xticks([])
            ax.set_yticks([])
            if n_comp == 3:
                ax.set_zticks([])

    for j in range(len(pairs), len(axes_flat)):
        axes_flat[j].set_visible(False)

    n_note = f"Same test split per panel (n={common_n}). " if common_n else ""
    dim_box = r"$[0,1]^3$" if n_comp == 3 else r"$[0,1]^2$"
    if args.color_mode == "ternary":
        view_3d = (
            "Static 3D view (elevation 22°, azimuth −58°); for exploration use an interactive viewer. "
            if n_comp == 3
            else ""
        )
        fig.suptitle(
            "MOSEI test-set $h_p$ t-SNE — PRISM macro ablations (trial 70)\n"
            + n_note
            + "Ternary style: Acc-7 collapsed to negative / neutral / positive; "
            + f"each panel axes are min–max scaled to {dim_box}; t-SNE fit per variant. "
            + view_3d,
            fontsize=11,
            y=1.02,
        )
    else:
        overlap_note = (
            "t-SNE is fit independently per variant; dense 3D overlap in one view is normal."
            if n_comp == 3
            else "t-SNE is fit independently per variant; dense 2D overlap is normal and does not show per-bin mass."
        )
        fig.suptitle(
            "MOSEI test-set $h_p$ t-SNE — PRISM macro ablations (trial 70)\n"
            + (
                "Colors: Acc-7 discrete bins (``train.py`` metric); markers ▼ / ● / ▲ = negative / neutral / positive bins. "
                if args.color_mode == "discrete7"
                else "Colors: continuous (debug); same ▼/●/▲ markers by Acc-7 bin. "
            )
            + n_note
            + overlap_note,
            fontsize=11,
            y=1.02,
        )

    if args.color_mode == "discrete7" and legend_handles is not None:
        fig.legend(
            legend_handles,
            [h.get_label() for h in legend_handles],
            loc="lower center",
            ncol=7,
            fontsize=11,
            frameon=True,
            fancybox=False,
            edgecolor="#888888",
            bbox_to_anchor=(0.5, -0.02),
            borderaxespad=0.8,
            handlelength=1.4,
            handleheight=1.2,
        )
        plt.tight_layout(rect=[0, 0.12, 1, 0.96])
    elif args.color_mode == "ternary":
        plt.tight_layout(rect=[0, 0.04, 1, 0.94])
    else:
        cbar = fig.colorbar(sc, ax=axes_flat[: len(pairs)], shrink=0.65, label="Sentiment (continuous)")
        cbar.ax.tick_params(labelsize=9)
        plt.tight_layout(rect=[0, 0.06, 1, 0.94])

    args.output = args.output.expanduser().resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    stem = args.output.with_suffix("")
    png_path = stem.with_suffix(".png")
    pdf_path = stem.with_suffix(".pdf")
    fig.savefig(png_path, dpi=int(args.dpi), bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {png_path} and {pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
