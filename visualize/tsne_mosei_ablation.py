#!/usr/bin/env python3
"""MOSEI ablation t-SNE from InfoGate checkpoint(s): pooled ``h_p`` vectors.

Loads ``ablation_study/train.py`` with checkpoint ``args`` injected so ``build_model``
and ``setup_data`` match training. **Facet** mode uses a calm paper style (viridis
sentiment map, serif type, no tick clutter, thin black panel frames, horizontal
Negative/Positive colorbar). **Joint** mode keeps a compact sans-serif legend layout.

Example::

    cd My_creation && python visualize/tsne_mosei_ablation.py \\
      --ckpt PRISM=ablation_study/runs/mosei_phase1_trial70/checkpoints/infogate_mosei_best.pt \\
      --ckpt "w/o MSelector"=ablation_study/runs/mosei_phase1_trial70_no_mselector/checkpoints/infogate_mosei_best.pt \\
      --split test --mode facet --max-samples 3000
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import math
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import torch

_MY_CREATION = Path(__file__).resolve().parent.parent
_ABLATION_TRAIN = _MY_CREATION / "ablation_study" / "train.py"

DEFAULT_PALETTE = ["#0173B2", "#FFC857", "#8D37D5", "#029E73"]

def _facet_sentiment_cmap():
    """Perceptually uniform sequential map (EMOE / Figure-style calm look)."""
    try:
        return mpl.colormaps["viridis"]
    except (AttributeError, KeyError):
        return mpl.cm.get_cmap("viridis")


MARKERS_CYCLE = ("o", "s", "^", "v", "P", "X", "D")

ABLATION_DISPLAY = {
    "none": "PRISM",
    "no_infogate": "w/o InfoGate",
    "no_mselector": "w/o MSelector",
    "no_ib": "w/o IB",
    "no_conf_gating": "w/o ConfGating",
    "no_adaptive_gate": "w/o AdaptiveGate",
}

SUBPANEL_LABELS = tuple(chr(ord("a") + i) for i in range(26))


def apply_paper_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "0.35",
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
        }
    )


def apply_facet_calm_style() -> None:
    """EMOE-like calm facet figure: serif, white field, no tick noise, thin black frames."""
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "black",
            "axes.linewidth": 0.55,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.spines.left": True,
            "axes.spines.bottom": True,
            "axes.grid": False,
            "axes.titlesize": 9,
            "axes.titlepad": 6,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "font.family": "serif",
            "font.serif": [
                "DejaVu Serif",
                "Nimbus Roman",
                "Times New Roman",
                "Times",
                "serif",
            ],
            "mathtext.fontset": "dejavuserif",
        }
    )


def load_ablation_train_module(saved_ns: argparse.Namespace):
    """Execute ``ablation_study/train.py`` with ``parse_args`` returning ``saved_ns``."""
    import argparse as argparse_mod

    ab_dir = str(_MY_CREATION / "ablation_study")
    mc_dir = str(_MY_CREATION)
    # Ablation snapshot must shadow My_creation for `deberta_infogate` (insert(0, mc)
    # after insert(0, ab) would otherwise put My_creation first and load the wrong module).
    if ab_dir not in sys.path:
        sys.path.insert(0, ab_dir)
    if mc_dir not in sys.path:
        sys.path.append(mc_dir)

    ns_copy = copy.deepcopy(saved_ns)
    real_parse = argparse_mod.ArgumentParser.parse_args

    def _fake_parse(self, args=None, namespace=None):
        return ns_copy

    argparse_mod.ArgumentParser.parse_args = _fake_parse
    try:
        spec = importlib.util.spec_from_file_location(
            "ablation_train_tsne", str(_ABLATION_TRAIN)
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load {_ABLATION_TRAIN}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        argparse_mod.ArgumentParser.parse_args = real_parse
    return mod


def collect_h_p(model, loader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    hs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = tuple(t.to(device) for t in batch)
            input_ids, visual, acoustic, label_ids = batch
            visual = visual.squeeze(1)
            acoustic = acoustic.squeeze(1)
            _, _, _, h_p = model(input_ids, visual, acoustic, stage=2)
            hs.append(h_p.detach().float().cpu().numpy())
            ys.append(label_ids.detach().cpu().numpy())
    if not hs:
        raise RuntimeError("Empty loader — no embeddings collected.")
    return np.concatenate(hs, axis=0), np.concatenate(ys, axis=0).ravel()


def subsample_indices(n: int, max_samples: int | None, seed: int) -> np.ndarray:
    if max_samples is None or max_samples >= n:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return rng.choice(n, size=max_samples, replace=False)


def parse_ckpt_specs(pairs: list[str]) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    for p in pairs:
        if "=" not in p:
            raise ValueError(f"--ckpt must be name=path, got: {p}")
        name, path = p.split("=", 1)
        name = name.strip()
        path = Path(path.strip()).expanduser()
        if not name:
            raise ValueError(f"Empty name in --ckpt {p}")
        out.append((name, path.resolve()))
    return out


def run_tsne(X: np.ndarray, perplexity: float, seed: int, standard_scale: bool):
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler

    if standard_scale:
        X = StandardScaler().fit_transform(X)
    n = X.shape[0]
    perp = min(perplexity, max(5, n - 1))
    tsne = TSNE(
        n_components=2,
        perplexity=float(perp),
        random_state=seed,
        init="pca",
        learning_rate="auto",
    )
    return tsne.fit_transform(X)


def add_subplot_label(ax, label: str) -> None:
    ax.text(
        -0.10,
        1.06,
        f"({label})",
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def _style_facet_axes_calm(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="both", which="both", length=0, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.55)
        spine.set_edgecolor("black")


def scatter_facet_calm(
    ax,
    z: np.ndarray,
    y: np.ndarray,
    cmap,
    title: str,
    panel_letter: str,
    vmin: float,
    vmax: float,
) -> None:
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    ax.scatter(
        z[:, 0],
        z[:, 1],
        s=10,
        alpha=0.52,
        c=y,
        cmap=cmap,
        norm=norm,
        edgecolors="none",
        rasterized=True,
    )
    _style_facet_axes_calm(ax)
    ax.text(
        0.5,
        1.085,
        f"({panel_letter})",
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        va="bottom",
        ha="center",
    )
    ax.set_title(title, fontsize=9, pad=2)


def scatter_models(ax, z: np.ndarray, model_idx: np.ndarray, names: list[str], palette: list[str]):
    n_models = len(names)
    for m in range(n_models):
        mask = model_idx == m
        if not np.any(mask):
            continue
        color = palette[m % len(palette)]
        marker = MARKERS_CYCLE[m % len(MARKERS_CYCLE)]
        ec = "0.35" if (m % len(palette)) == 1 else "0.22"
        ax.scatter(
            z[mask, 0],
            z[mask, 1],
            s=14,
            alpha=0.65,
            c=color,
            marker=marker,
            edgecolors=ec,
            linewidths=0.2,
            label=names[m],
            rasterized=True,
        )
    ax.set_title("Joint t-SNE (all checkpoints)", fontsize=10)
    ax.set_xlabel("t-SNE-1")
    ax.set_ylabel("t-SNE-2")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ckpt",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="Repeatable: short label and checkpoint path (must contain args + model_state_dict).",
    )
    parser.add_argument("--split", choices=("dev", "test"), default="test")
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--mode", choices=("facet", "joint"), default="facet")
    parser.add_argument("--max-samples", type=int, default=4000)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=128)
    parser.add_argument("--standard-scale", action="store_true")
    parser.add_argument(
        "--palette",
        type=str,
        default=None,
        help="Comma-separated hex colors (default: blue/yellow/purple/green).",
    )
    parser.add_argument(
        "--no-paper-style",
        action="store_true",
        help="Skip matplotlib rc updates.",
    )
    args_cli = parser.parse_args()

    if not args_cli.no_paper_style:
        if args_cli.mode == "facet":
            apply_facet_calm_style()
        else:
            apply_paper_style()

    if args_cli.palette:
        palette = [x.strip() for x in args_cli.palette.split(",") if x.strip()]
        if len(palette) < 1:
            raise SystemExit("--palette must list at least one hex color.")
    else:
        palette = list(DEFAULT_PALETTE)

    ckpt_specs = parse_ckpt_specs(args_cli.ckpt)
    outdir = (
        args_cli.outdir.expanduser().resolve()
        if args_cli.outdir
        else (_MY_CREATION / "visualize" / "figures" / "mosei_ablation_tsne")
    )
    outdir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    series: list[tuple[str, np.ndarray, np.ndarray]] = []

    for name, ckpt_path in ckpt_specs:
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        try:
            bundle = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except TypeError:
            bundle = torch.load(ckpt_path, map_location="cpu")
        if "args" not in bundle or "model_state_dict" not in bundle:
            raise KeyError(f"{ckpt_path}: expected keys 'args', 'model_state_dict'")
        saved_ns = bundle["args"]
        if getattr(saved_ns, "dataset", None) != "mosei":
            print(
                f"WARNING: {name}: ckpt dataset is {getattr(saved_ns, 'dataset', None)!r}, "
                "expected 'mosei'. Continuing anyway.",
                file=sys.stderr,
            )

        train_mod = load_ablation_train_module(saved_ns)
        train_mod.set_seed(args_cli.seed)
        train_mod.global_configs.set_dataset_config(saved_ns.dataset)
        _, dev_dl, test_dl, n_opt = train_mod.setup_data()
        loader = dev_dl if args_cli.split == "dev" else test_dl

        model, _, _ = train_mod.build_model(n_opt)
        missing, unexpected = model.load_state_dict(
            bundle["model_state_dict"], strict=False
        )
        if missing:
            print(f"WARNING {name}: missing keys ({len(missing)}):", missing[:5], file=sys.stderr)
        if unexpected:
            print(f"WARNING {name}: unexpected keys ({len(unexpected)}):", unexpected[:5], file=sys.stderr)
        model.to(device)

        H, y = collect_h_p(model, loader, device)
        series.append((name, H, y))
        print(f"{name}: embeddings shape={H.shape} labels shape={y.shape}")

        del model, bundle
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    n_min = min(H.shape[0] for _, H, _ in series)
    idx = subsample_indices(n_min, args_cli.max_samples, args_cli.seed)

    display_names = [ABLATION_DISPLAY.get(n, n) for n, _, _ in series]

    if args_cli.mode == "facet":
        all_y = np.concatenate([y[idx] for _, _, y in series])
        vmin = float(np.percentile(all_y, 2))
        vmax = float(np.percentile(all_y, 98))

        facet_cmap = _facet_sentiment_cmap()
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

        n_ckpt = len(series)
        ncols = min(3, n_ckpt)
        nrows = math.ceil(n_ckpt / ncols)
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(3.35 * ncols, 3.15 * nrows), squeeze=False
        )
        for i, (name, H, y) in enumerate(series):
            row, col = divmod(i, ncols)
            ax = axes[row][col]
            Z = run_tsne(
                H[idx],
                perplexity=args_cli.perplexity,
                seed=args_cli.seed,
                standard_scale=args_cli.standard_scale,
            )
            scatter_facet_calm(
                ax,
                Z,
                y[idx],
                facet_cmap,
                display_names[i],
                SUBPANEL_LABELS[i],
                vmin,
                vmax,
            )
        for j in range(len(series), nrows * ncols):
            row, col = divmod(j, ncols)
            axes[row][col].axis("off")

        sm = mpl.cm.ScalarMappable(cmap=facet_cmap, norm=norm)
        sm.set_array([])
        fig.subplots_adjust(
            left=0.05, right=0.98, top=0.91, bottom=0.16, wspace=0.14, hspace=0.36
        )
        cbar_ax = fig.add_axes([0.28, 0.035, 0.44, 0.022])
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
        cbar.set_ticks([vmin, vmax])
        cbar.set_ticklabels(["Negative", "Positive"])
        cbar.ax.tick_params(axis="x", which="major", length=0, pad=4)
        cbar.outline.set_linewidth(0.45)
        cbar.outline.set_edgecolor("black")
    else:
        X_blocks: list[np.ndarray] = []
        model_ids: list[np.ndarray] = []
        for m, (name, H, y) in enumerate(series):
            X_blocks.append(H[idx])
            model_ids.append(np.full(len(idx), m, dtype=np.int32))
        X_all = np.vstack(X_blocks)
        mid = np.concatenate(model_ids)
        Z = run_tsne(
            X_all,
            perplexity=args_cli.perplexity,
            seed=args_cli.seed,
            standard_scale=args_cli.standard_scale,
        )
        fig, ax = plt.subplots(figsize=(5.8, 5.0))
        scatter_models(ax, Z, mid, display_names, palette)
        ax.legend(loc="best", ncol=2, fontsize=8.5)
        add_subplot_label(ax, SUBPANEL_LABELS[0])
        plt.tight_layout()

    stem = f"mosei_tsne_{args_cli.split}_{args_cli.mode}"
    pdf_path = outdir / f"{stem}.pdf"
    png_path = outdir / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight", format="pdf")
    fig.savefig(png_path, bbox_inches="tight", dpi=300, format="png")
    plt.close(fig)
    print(f"Wrote {pdf_path}")
    print(f"Wrote {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
