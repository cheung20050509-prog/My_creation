#!/usr/bin/env python3
"""Plot PRISM MOSEI fused landscape as a smooth 3D surface from real test features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from landscape_common import load_npz_features, plot_single_kde_fused_landscape

HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE / "outputs" / "mosei_trial70_intro_features.npz"
DEFAULT_OUTPUT = HERE / "outputs" / "mosei_trial70_fused_landscape.png"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--projection", choices=("pca", "tsne"), default="pca")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--perplexity", type=float, default=30.0)
    ap.add_argument("--grid-size", type=int, default=130)
    ap.add_argument("--bandwidth-scale", type=float, default=3.0)
    ap.add_argument("--dpi", type=int, default=480)
    args = ap.parse_args()

    _, prism, _, _, meta = load_npz_features(args.input)
    report = plot_single_kde_fused_landscape(
        fused=prism,
        output=args.output,
        title="PRISM Fused Representation Landscape (MOSEI trial70, real test data)",
        panel_title="PRISM Fusion (h_p, stage 2)",
        projection=args.projection,
        seed=args.seed,
        perplexity=args.perplexity,
        grid_size=args.grid_size,
        bandwidth_scale=args.bandwidth_scale,
        dpi=args.dpi,
    )
    report["input"] = str(args.input.expanduser().resolve())
    report["source_meta"] = meta
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
