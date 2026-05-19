#!/usr/bin/env python3
"""Export MOSEI trial~70, test ``global_index`` 2856: (A) optional matplotlib table PDF/PNG
for drafts, (B) token conf heatmap PDF/PNG for the paper.

The ACL manuscript uses an **inline LaTeX table** for (a); see ``overleaf_69e83a58/acl_latex.tex`` (Qualitative Evaluation). By default only the heatmap PDF is copied to Overleaf ``graph/``.

Run from ``My_creation/``::

    python Qualitative_Evaluation/plot_mosei_case_table_and_heatmap.py
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys

_THIS = os.path.dirname(os.path.abspath(__file__))
_MY = os.path.dirname(_THIS)
_REPO_ROOT = os.path.dirname(_MY)
_DEFAULT_RESULTS = os.path.join(_MY, "Qualitative_Evaluation", "results")
_DEFAULT_CSV = os.path.join(_DEFAULT_RESULTS, "mosei_trial70_dpr_test.csv")
_DEFAULT_TSV = os.path.join(_DEFAULT_RESULTS, "mosei_trial70_idx2856_ib_conf_tokens.tsv")
_DEFAULT_OVERLEAF_GRAPH = os.path.join(_REPO_ROOT, "overleaf_69e83a58", "graph")


def _find_csv_row(csv_path: str, global_index: int) -> dict[str, str]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            if int(row["global_index"]) == global_index:
                return row
    raise SystemExit(f"ERROR: no row global_index={global_index} in {csv_path}")


def _fmt(v: str, key: str) -> str:
    if key in ("label", "logit", "abs_err", "route_entropy", "h_p_l2norm"):
        try:
            return f"{float(v):.6g}"
        except ValueError:
            return v
    if key.startswith("w_") or key.startswith("ib_conf_"):
        try:
            return f"{float(v):.6f}"
        except ValueError:
            return v
    return v


def _table_rows_from_csv(row: dict[str, str]) -> list[tuple[str, str]]:
    order = [
        ("Dataset", "CMU-MOSEI"),
        ("Optuna trial", "70"),
        ("Test global_index", row.get("global_index", "")),
        ("Test split_index", row.get("split_index", "")),
        ("Label (sentiment)", _fmt(row["label"], "label")),
        ("Prediction (logit)", _fmt(row["logit"], "logit")),
        ("Absolute error", _fmt(row["abs_err"], "abs_err")),
        ("DPR primary (name)", row.get("primary_name", "")),
        ("DPR primary (idx)", row.get("primary_idx", "")),
        ("Route entropy", _fmt(row.get("route_entropy", ""), "route_entropy")),
        ("||h_p|| (router)", _fmt(row.get("h_p_l2norm", ""), "h_p_l2norm")),
        ("w_l (language)", _fmt(row.get("w_l", ""), "w_l")),
        ("w_a (acoustic)", _fmt(row.get("w_a", ""), "w_a")),
        ("w_v (visual)", _fmt(row.get("w_v", ""), "w_v")),
        ("w_max", _fmt(row.get("w_max", ""), "w_max")),
        ("ib_conf_t (sample mean)", _fmt(row.get("ib_conf_t", ""), "ib_conf_t")),
        ("ib_conf_a", _fmt(row.get("ib_conf_a", ""), "ib_conf_a")),
        ("ib_conf_v", _fmt(row.get("ib_conf_v", ""), "ib_conf_v")),
        ("ib_conf_fused", _fmt(row.get("ib_conf_fused", ""), "ib_conf_fused")),
        ("Transcript / frames (manual)", "—"),
        ("Video / audio strip (manual)", "—"),
    ]
    return order


def _save_table_figure(out_base: str, pairs: list[tuple[str, str]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_w, fig_h = 6.2, 7.8
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=150)
    ax.axis("off")
    ax.set_title("MOSEI case summary (tabular placeholder; trial 70, idx 2856)", fontsize=11, pad=12)

    cell_text = [[k, v] for k, v in pairs]
    table = ax.table(
        cellText=cell_text,
        colLabels=["Field", "Value"],
        loc="center",
        cellLoc="left",
        colWidths=[0.42, 0.58],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.35)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#e6e6e6")
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_base)) or ".", exist_ok=True)
    for ext in (".pdf", ".png"):
        path = out_base + ext
        fig.savefig(path, bbox_inches="tight")
        print(f"Wrote {os.path.abspath(path)}")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=str, default=_DEFAULT_CSV, help="mosei_trial70_dpr_test.csv")
    ap.add_argument("--tsv", type=str, default=_DEFAULT_TSV, help="token-level ib_conf TSV")
    ap.add_argument("--global-index", type=int, default=2856)
    ap.add_argument(
        "--out-dir",
        type=str,
        default=_DEFAULT_RESULTS,
        help="Directory for outputs (table + heatmap basenames)",
    )
    ap.add_argument("--cmap", type=str, default="viridis")
    copy_g = ap.add_mutually_exclusive_group()
    copy_g.add_argument(
        "--copy-to-overleaf",
        action="store_true",
        dest="copy_to_overleaf",
        help="Copy generated PDFs to overleaf graph (default unless --no-copy-to-overleaf)",
    )
    copy_g.add_argument(
        "--no-copy-to-overleaf",
        action="store_false",
        dest="copy_to_overleaf",
        help="Do not copy PDFs to Overleaf",
    )
    ap.set_defaults(copy_to_overleaf=True)
    ap.add_argument(
        "--overleaf-graph",
        type=str,
        default=_DEFAULT_OVERLEAF_GRAPH,
        help="Destination for --copy-to-overleaf",
    )
    ap.add_argument(
        "--heatmap-minimal",
        action="store_true",
        help="Pass-through to plot_ib_conf_token_tsv: minimal heatmap decoration",
    )
    args = ap.parse_args()

    if not os.path.isfile(args.csv):
        print(f"ERROR: missing {args.csv}", file=sys.stderr)
        return 1
    if not os.path.isfile(args.tsv):
        print(f"ERROR: missing {args.tsv}", file=sys.stderr)
        return 1

    sys.path.insert(0, _THIS)
    from plot_ib_conf_token_tsv import (
        load_ib_conf_token_rows,
        resolve_ib_conf_plot_columns,
        save_heatmap_figures,
    )

    row = _find_csv_row(args.csv, args.global_index)
    pairs = _table_rows_from_csv(row)
    table_base = os.path.join(args.out_dir, "mosei_trial70_idx2856_case_table")
    _save_table_figure(table_base, pairs)

    rows = load_ib_conf_token_rows(args.tsv, include_pad=False)
    plot_cols, notes = resolve_ib_conf_plot_columns(rows)
    hm_base = os.path.join(args.out_dir, "mosei_trial70_idx2856_ib_conf_tokens_heatmap_standalone")
    save_heatmap_figures(
        [hm_base + ".png", hm_base + ".pdf"],
        rows,
        plot_cols,
        notes,
        title=os.path.basename(args.tsv),
        cmap=args.cmap,
        vscale="data",
        vmin=None,
        vmax=None,
        heatmap_minimal=args.heatmap_minimal,
    )

    if args.copy_to_overleaf:
        dest = args.overleaf_graph
        os.makedirs(dest, exist_ok=True)
        # Paper embeds case fields as LaTeX tabular; only heatmap PDF is shipped to Overleaf.
        for name in ("mosei_trial70_idx2856_ib_conf_tokens_heatmap_standalone.pdf",):
            src = os.path.join(args.out_dir, name)
            shutil.copy2(src, os.path.join(dest, name))
            print(f"Copied -> {os.path.join(dest, name)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
