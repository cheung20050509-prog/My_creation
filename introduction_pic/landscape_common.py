"""Shared helpers for 3D landscapes from real test-set feature matrices."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


def standardize(x: np.ndarray) -> np.ndarray:
    return StandardScaler().fit_transform(np.asarray(x, dtype=np.float64))


def feature_activation_magnitude(x: np.ndarray) -> np.ndarray:
    """Per-sample L2 norm in standardized feature space (real vectors only)."""
    xs = standardize(x)
    mag = np.linalg.norm(xs, axis=1)
    lo, hi = np.percentile(mag, [5.0, 95.0])
    if hi <= lo + 1e-12:
        return np.ones_like(mag)
    return np.clip((mag - lo) / (hi - lo), 0.0, 1.0)


def scott_bandwidth(xy: np.ndarray) -> float:
    """Scott's rule from projected 2D coordinates."""
    xy = np.asarray(xy, dtype=np.float64)
    n, d = xy.shape
    if n < 2:
        return 0.5
    std = np.std(xy, axis=0, ddof=1)
    std = np.maximum(std, 1e-6)
    return float(1.06 * np.min(std) * (n ** (-1.0 / (d + 4))))


def project_2d(
    x: np.ndarray,
    *,
    method: str,
    seed: int,
    perplexity: float,
) -> np.ndarray:
    xs = standardize(x)
    if method == "pca":
        emb = PCA(n_components=2, random_state=seed).fit_transform(xs)
    elif method == "tsne":
        perp = min(float(perplexity), max(2.0, float(xs.shape[0] - 1)))
        emb = TSNE(
            n_components=2,
            perplexity=perp,
            init="pca",
            learning_rate="auto",
            early_exaggeration=12.0,
            random_state=seed,
        ).fit_transform(xs)
    else:
        raise ValueError(f"unknown projection method: {method}")
    return emb


def axis_limits_from_points(xy: np.ndarray, *, pad: float = 0.15) -> tuple[float, float]:
    xy = np.asarray(xy, dtype=np.float64)
    lo = float(np.percentile(xy, 1.0))
    hi = float(np.percentile(xy, 99.0))
    span = max(hi - lo, 1e-9)
    margin = pad * span
    return lo - margin, hi + margin


def binned_surface_from_points(
    xy: np.ndarray,
    weights: np.ndarray,
    *,
    n_bins: int,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """2D histogram of real projected samples; height = sum of sample weights per bin."""
    xy = np.asarray(xy, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    hist, xedges, yedges = np.histogram2d(
        xy[:, 0],
        xy[:, 1],
        bins=int(n_bins),
        range=[[xlim[0], xlim[1]], [ylim[0], ylim[1]]],
        weights=weights,
    )
    xc = 0.5 * (xedges[:-1] + xedges[1:])
    yc = 0.5 * (yedges[:-1] + yedges[1:])
    xx, yy = np.meshgrid(xc, yc)
    zz = hist.T.astype(np.float64)
    occupied = zz > 0
    if not np.any(occupied):
        return xx, yy, zz, 1.0
    peak = float(np.percentile(zz[occupied], 99.5))
    return xx, yy, zz, peak


def kde_surface_from_points(
    xy: np.ndarray,
    weights: np.ndarray,
    *,
    grid_size: int,
    bandwidth: float,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    chunk: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    gx = np.linspace(xlim[0], xlim[1], grid_size)
    gy = np.linspace(ylim[0], ylim[1], grid_size)
    xx, yy = np.meshgrid(gx, gy)
    pts = np.column_stack([xx.ravel(), yy.ravel()])
    z = np.zeros(pts.shape[0], dtype=np.float64)
    xy = np.asarray(xy, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    bw2 = 2.0 * float(bandwidth) ** 2

    for start in range(0, xy.shape[0], chunk):
        sub_xy = xy[start : start + chunk]
        sub_w = weights[start : start + chunk]
        diff = pts[:, None, :] - sub_xy[None, :, :]
        kern = np.exp(-(diff * diff).sum(axis=2) / bw2)
        z += kern @ sub_w

    z = z.reshape(xx.shape)
    peak = float(np.percentile(z, 99.5))
    return xx, yy, z, peak


def draw_real_binned_panel(
    ax,
    x: np.ndarray,
    *,
    title: str,
    projection: str,
    seed: int,
    perplexity: float,
    n_bins: int,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
):
    xy = project_2d(x, method=projection, seed=seed, perplexity=perplexity)
    weights = feature_activation_magnitude(x)
    if xlim is None:
        xlim = axis_limits_from_points(xy[:, 0])
    if ylim is None:
        ylim = axis_limits_from_points(xy[:, 1])

    xx, yy, zz_raw, peak = binned_surface_from_points(
        xy,
        weights,
        n_bins=n_bins,
        xlim=xlim,
        ylim=ylim,
    )
    zz = np.where(zz_raw > 0, zz_raw / max(peak, 1e-12), np.nan)

    surf = ax.plot_surface(
        xx,
        yy,
        zz,
        cmap=cm.jet,
        norm=Normalize(vmin=0.0, vmax=1.0),
        linewidth=0,
        antialiased=True,
        rstride=1,
        cstride=1,
        alpha=0.98,
    )
    ax.contour(
        xx,
        yy,
        np.nan_to_num(zz, nan=0.0),
        zdir="z",
        offset=0.0,
        cmap=cm.jet,
        levels=10,
        linewidths=0.35,
    )
    ax.set_title(title, fontsize=11, pad=8)
    ax.set_xlabel("Latent Dimension 1", fontsize=8, labelpad=4)
    ax.set_ylabel("Latent Dimension 2", fontsize=8, labelpad=4)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(0.0, 1.0)
    ax.view_init(elev=27, azim=-58)
    ax.tick_params(axis="both", labelsize=7, pad=1)
    ax.tick_params(axis="z", labelsize=7, pad=1)
    try:
        ax.set_box_aspect((1.2, 1.0, 0.68))
    except AttributeError:
        pass
    return surf, {
        "projection": projection,
        "n_bins": int(n_bins),
        "z_panel_norm_raw": peak,
        "z_peak_raw": peak,
        "n_samples": int(x.shape[0]),
        "feature_dim": int(x.shape[1]),
        "n_occupied_bins": int(np.sum(zz_raw > 0)),
        "xlim": list(xlim),
        "ylim": list(ylim),
    }


def plot_single_fused_landscape(
    *,
    fused: np.ndarray,
    output: Path,
    title: str,
    panel_title: str,
    projection: str = "pca",
    seed: int = 42,
    perplexity: float = 30.0,
    n_bins: int = 48,
    dpi: int = 480,
) -> dict:
    """One 3D figure from real fused features only (no concat, no KDE grid fill)."""
    fig = plt.figure(figsize=(7.2, 5.8))
    ax = fig.add_subplot(111, projection="3d")
    surf, stats = draw_real_binned_panel(
        ax,
        fused,
        title=panel_title,
        projection=projection,
        seed=seed,
        perplexity=perplexity,
        n_bins=n_bins,
    )
    stats["method"] = "real_fused_features_pca_weighted_histogram"
    stats["data"] = "MOSEI test samples only; empty bins masked (no synthetic fill)"

    cbar = fig.colorbar(surf, ax=ax, shrink=0.72, pad=0.08, fraction=0.05)
    cbar.set_label("Binned activation (normalized)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    fig.suptitle(title, fontsize=12, y=0.98)

    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=int(dpi), bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return {
        "output_png": str(output),
        "output_pdf": str(output.with_suffix(".pdf")),
        "plot": stats,
    }


def plot_single_kde_fused_landscape(
    *,
    fused: np.ndarray,
    output: Path,
    title: str,
    panel_title: str,
    projection: str = "pca",
    seed: int = 42,
    perplexity: float = 30.0,
    grid_size: int = 130,
    bandwidth_scale: float = 1.0,
    dpi: int = 480,
) -> dict:
    """One smooth 3D surface estimated from real fused test features."""
    xy = project_2d(fused, method=projection, seed=seed, perplexity=perplexity)
    weights = feature_activation_magnitude(fused)
    xlim = axis_limits_from_points(xy[:, 0])
    ylim = axis_limits_from_points(xy[:, 1])
    bw = scott_bandwidth(xy) * float(bandwidth_scale)

    xx, yy, zz_raw, peak = kde_surface_from_points(
        xy,
        weights,
        grid_size=grid_size,
        bandwidth=bw,
        xlim=xlim,
        ylim=ylim,
    )
    zz = zz_raw / max(peak, 1e-12)

    fig = plt.figure(figsize=(7.2, 5.8))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(
        xx,
        yy,
        zz,
        cmap=cm.jet,
        norm=Normalize(vmin=0.0, vmax=1.0),
        linewidth=0,
        antialiased=True,
        rstride=1,
        cstride=1,
        alpha=0.98,
    )
    ax.contour(xx, yy, zz, zdir="z", offset=0.0, cmap=cm.jet, levels=10, linewidths=0.35)
    ax.set_title(panel_title, fontsize=11, pad=8)
    ax.set_xlabel("Latent Dimension 1", fontsize=8, labelpad=4)
    ax.set_ylabel("Latent Dimension 2", fontsize=8, labelpad=4)
    ax.set_zlabel("Latent Activation Magnitude", fontsize=8, labelpad=4)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(0.0, 1.0)
    ax.view_init(elev=27, azim=-58)
    ax.tick_params(axis="both", labelsize=7, pad=1)
    ax.tick_params(axis="z", labelsize=7, pad=1)
    try:
        ax.set_box_aspect((1.2, 1.0, 0.68))
    except AttributeError:
        pass

    cbar = fig.colorbar(surf, ax=ax, shrink=0.72, pad=0.08, fraction=0.05)
    cbar.set_label("Latent Activation Magnitude", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    fig.suptitle(title, fontsize=12, y=0.98)

    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=int(dpi), bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return {
        "output_png": str(output),
        "output_pdf": str(output.with_suffix(".pdf")),
        "plot": {
            "method": "real_fused_features_weighted_kde_scott",
            "projection": projection,
            "bandwidth": bw,
            "bandwidth_scale": float(bandwidth_scale),
            "grid_size": int(grid_size),
            "z_panel_norm_raw": peak,
            "z_peak_raw": peak,
            "n_samples": int(fused.shape[0]),
            "feature_dim": int(fused.shape[1]),
            "xlim": list(xlim),
            "ylim": list(ylim),
        },
    }


def plot_landscape_compare_figure(
    *,
    concat: np.ndarray,
    fused: np.ndarray,
    output: Path,
    suptitle: str,
    concat_title: str,
    fused_title: str,
    projection: str = "pca",
    seed: int = 42,
    perplexity: float = 30.0,
    grid_size: int = 130,
    layout: str = "vertical",
    dpi: int = 480,
) -> dict:
    if layout == "horizontal":
        fig = plt.figure(figsize=(11.2, 4.3))
        axes = [fig.add_subplot(1, 2, 1, projection="3d"), fig.add_subplot(1, 2, 2, projection="3d")]
    else:
        fig = plt.figure(figsize=(6.6, 8.4))
        axes = [fig.add_subplot(2, 1, 1, projection="3d"), fig.add_subplot(2, 1, 2, projection="3d")]

    xy_c = project_2d(concat, method=projection, seed=seed, perplexity=perplexity)
    xy_f = project_2d(fused, method=projection, seed=seed, perplexity=perplexity)
    xlim_c = axis_limits_from_points(xy_c[:, 0])
    ylim_c = axis_limits_from_points(xy_c[:, 1])
    xlim_f = axis_limits_from_points(xy_f[:, 0])
    ylim_f = axis_limits_from_points(xy_f[:, 1])

    n_bins = max(24, int(grid_size) // 3)
    stats: dict = {}
    surf, stats["concat"] = draw_real_binned_panel(
        axes[0],
        concat,
        title=concat_title,
        projection=projection,
        seed=seed,
        perplexity=perplexity,
        n_bins=n_bins,
        xlim=xlim_c,
        ylim=ylim_c,
    )
    _, stats["fused"] = draw_real_binned_panel(
        axes[1],
        fused,
        title=fused_title,
        projection=projection,
        seed=seed,
        perplexity=perplexity,
        n_bins=n_bins,
        xlim=xlim_f,
        ylim=ylim_f,
    )
    stats["shared"] = {
        "method": "real_features_pca_weighted_histogram",
        "n_bins": n_bins,
    }

    cbar = fig.colorbar(surf, ax=axes, shrink=0.58, pad=0.10, fraction=0.045)
    cbar.set_label("Binned activation (normalized)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    fig.suptitle(suptitle, fontsize=12, y=0.985)

    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=int(dpi), bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return {
        "output_png": str(output),
        "output_pdf": str(output.with_suffix(".pdf")),
        "plot": stats,
    }


def load_npz_features(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, dict]:
    data = np.load(path.expanduser().resolve(), allow_pickle=False)
    concat = data["concat"]
    if "prism" in data.files:
        fused = data["prism"]
    elif "ithp" in data.files:
        fused = data["ithp"]
    else:
        raise KeyError(f"npz missing fused key (prism/ithp): {path}")
    preds = data["preds"] if "preds" in data.files else None
    labels = data["labels"] if "labels" in data.files else None
    meta = json.loads(str(data["meta"])) if "meta" in data.files else {}
    return concat, fused, preds, labels, meta
