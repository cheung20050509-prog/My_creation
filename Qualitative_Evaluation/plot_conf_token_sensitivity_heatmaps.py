#!/usr/bin/env python3
"""Token-level confidence and prediction-sensitivity heatmaps.

For one sample, the script masks the auxiliary confidence at one valid token at
a time, reruns the model, and records the prediction/error change. It then draws
two heatmaps:

1. original token order: confidence, signed prediction shift, error change
2. confidence-sorted order: the same rows sorted by auxiliary confidence

Run from ``My_creation/``::

    CUDA_VISIBLE_DEVICES="" python Qualitative_Evaluation/plot_conf_token_sensitivity_heatmaps.py \
      --checkpoint logs/optuna/4090D_restart/phase1/checkpoints/optuna_mosei_phase1/trial_70/infogate_mosei_best.pt \
      --dataset mosei --split test --global-index 750 \
      --output-prefix Qualitative_Evaluation/results/mosei_trial70_idx750_conf_token_sensitivity
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from types import MethodType
from typing import Callable

import numpy as np
import torch

_QE_DIR = os.path.dirname(os.path.abspath(__file__))
_MY = os.path.dirname(_QE_DIR)
if _MY not in sys.path:
    sys.path.insert(0, _MY)
if _QE_DIR not in sys.path:
    sys.path.insert(0, _QE_DIR)

import global_configs  # noqa: E402
from deberta_infogate import InfoGate_DeBertaForSequenceClassification  # noqa: E402
from global_configs import DEVICE  # noqa: E402
from ib_conf_utils import install_infogate_ib_trace, per_token_ib_conf_mean_over_bottleneck  # noqa: E402
from plot_conf_intervention_prediction import (  # noqa: E402
    _default_cli,
    _load_deberta_dump_helpers,
    _load_single_sample,
)


SLOT_NAMES = {0: "acoustic", 1: "language", 2: "visual"}
SLOT_CONF_KEYS = {0: "a", 1: "t", 2: "v"}


def _install_single_token_mask(ig_module, token_pos: int | None) -> Callable[[], None]:
    original_forward = ig_module.forward

    def patched_forward(self, B_p, conf_p, B_aux_list, conf_aux_list, tok_mask=None):
        if token_pos is None:
            new_conf = conf_aux_list
        else:
            new_conf = []
            for conf in conf_aux_list:
                out = conf.clone()
                if 0 <= token_pos < out.size(1):
                    out[:, token_pos, :] = 0.0
                new_conf.append(out)
        return original_forward(B_p, conf_p, B_aux_list, new_conf, tok_mask)

    ig_module.forward = MethodType(patched_forward, ig_module)

    def cleanup() -> None:
        ig_module.forward = original_forward

    return cleanup


def _forward_with_optional_mask(model, batch, token_pos: int | None, parse_mselector):
    ig = model.dberta.infogate
    pad_id = model.config.pad_token_id if model.config.pad_token_id is not None else 0
    captured = []

    def mselector_hook(_mod, _inp, out):
        weights, primary = parse_mselector(out)
        captured.append((weights.detach().cpu() if weights is not None else None, primary.detach().cpu()))

    hook = ig.mselector.register_forward_hook(mselector_hook)
    cleanup_ib, ib_store = install_infogate_ib_trace(ig)
    cleanup_mask = _install_single_token_mask(ig.infogate, token_pos)

    try:
        input_ids, visual, acoustic, labels = batch
        visual = visual.squeeze(1)
        acoustic = acoustic.squeeze(1)
        attention_mask = input_ids.ne(pad_id).float()
        ib_store.clear()
        captured.clear()
        with torch.no_grad():
            logits, _, _, _ = model(input_ids, visual, acoustic, labels=None, stage=2)
        if not captured:
            raise RuntimeError("mselector hook did not fire")
        weights, primary = captured[-1]
        if weights is None:
            raise RuntimeError("could not parse DPR weights")
        pred = float(logits.view(-1)[0].detach().cpu().item())
        label = float(labels.view(-1)[0].detach().cpu().item())
        primary_idx = int(primary.view(-1)[0].item())
        return {
            "pred": pred,
            "label": label,
            "abs_err": abs(pred - label),
            "weights": weights[0].detach().cpu().numpy(),
            "primary_idx": primary_idx,
            "attention_mask": attention_mask[0].detach().cpu().numpy(),
            "input_ids": input_ids[0].detach().cpu().tolist(),
            "ib_store": {k: v.detach().clone() for k, v in ib_store.items()},
            "attention_mask_tensor": attention_mask.detach().clone(),
        }
    finally:
        cleanup_mask()
        cleanup_ib()
        hook.remove()


def _token_conf_arrays(original, tokenizer) -> tuple[list[dict[str, float | str]], list[int]]:
    attn = original["attention_mask_tensor"]
    store = original["ib_store"]
    input_ids = original["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    valid_positions = [i for i, v in enumerate(original["attention_mask"]) if float(v) > 0.5]
    primary_idx = int(original["primary_idx"])
    aux_indices = [i for i in (0, 1, 2) if i != primary_idx]

    conf_by_slot = {}
    for idx, key in SLOT_CONF_KEYS.items():
        conf_by_slot[idx] = per_token_ib_conf_mean_over_bottleneck(store[key], attn)[0].detach().cpu().numpy()

    rows = []
    for pos in valid_positions:
        aux_vals = [float(conf_by_slot[idx][pos]) for idx in aux_indices]
        row = {
            "pos": pos,
            "token_id": int(input_ids[pos]),
            "token": tokens[pos].replace("\t", " "),
            "primary_name": SLOT_NAMES[primary_idx],
            "aux_conf_mean": float(np.mean(aux_vals)),
            "conf_acoustic": float(conf_by_slot[0][pos]),
            "conf_language": float(conf_by_slot[1][pos]),
            "conf_visual": float(conf_by_slot[2][pos]),
        }
        rows.append(row)
    return rows, valid_positions


def _write_csv(path: str, rows: list[dict[str, float | str]]) -> None:
    fieldnames = [
        "pos", "token_id", "token", "primary_name",
        "aux_conf_mean", "conf_acoustic", "conf_language", "conf_visual",
        "pred_masked", "delta_pred", "abs_delta_pred", "abs_err_masked", "delta_abs_err",
    ]
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _short_token(tok: str, max_len: int = 12) -> str:
    tok = tok.replace("▁", "").replace("Ġ", "").strip() or tok
    if len(tok) > max_len:
        return tok[: max_len - 1] + "…"
    return tok


def _heatmap_matrix(rows: list[dict[str, float | str]]) -> np.ndarray:
    return np.array(
        [
            [float(r["aux_conf_mean"]) for r in rows],
            [float(r["delta_pred"]) for r in rows],
            [float(r["delta_abs_err"]) for r in rows],
        ],
        dtype=np.float64,
    )


def _plot_one(path: str, rows: list[dict[str, float | str]], *, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mat = _heatmap_matrix(rows)
    labels = [_short_token(str(r["token"])) for r in rows]

    fig, axes = plt.subplots(3, 1, figsize=(max(9.5, len(rows) * 0.34), 4.8), dpi=180, sharex=True)
    specs = [
        ("Aux. conf. mean", mat[0], "viridis", None, None),
        (r"Signed $\Delta\hat{y}$", mat[1], "coolwarm", -max(abs(mat[1]).max(), 1e-6), max(abs(mat[1]).max(), 1e-6)),
        (r"$\Delta |\hat{y}-y|$", mat[2], "coolwarm", -max(abs(mat[2]).max(), 1e-6), max(abs(mat[2]).max(), 1e-6)),
    ]
    for ax, (ylabel, values, cmap, vmin, vmax) in zip(axes, specs):
        im = ax.imshow(values[np.newaxis, :], aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_yticks([])
        ax.set_ylabel(ylabel, rotation=0, ha="right", va="center", labelpad=78, fontsize=9)
        cbar = fig.colorbar(im, ax=ax, fraction=0.015, pad=0.008)
        cbar.ax.tick_params(labelsize=7)
        for x, val in enumerate(values):
            text_color = "white" if (vmin is None and val < np.nanmean(values)) else "black"
            if vmin is not None:
                text_color = "black"
            ax.text(x, 0, f"{val:+.3f}" if ylabel != "Aux. conf. mean" else f"{val:.2f}",
                    ha="center", va="center", fontsize=6, color=text_color)

    axes[0].set_title(title, fontsize=11)
    axes[-1].set_xticks(np.arange(len(rows)))
    axes[-1].set_xticklabels(labels, rotation=55, ha="right", fontsize=7)
    axes[-1].set_xlabel("Token")
    fig.tight_layout()
    for ext in (".png", ".pdf"):
        fig.savefig(path + ext, bbox_inches="tight")
    plt.close(fig)


def _plot_heatmaps(prefix: str, rows: list[dict[str, float | str]]) -> None:
    _plot_one(prefix + "_ordered", rows, title="Token confidence and prediction sensitivity (original order)")
    sorted_rows = sorted(rows, key=lambda r: float(r["aux_conf_mean"]))
    _plot_one(prefix + "_sorted", sorted_rows, title="Token confidence and prediction sensitivity (sorted by confidence)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, choices=("mosi", "mosei"), default="mosei")
    parser.add_argument("--split", type=str, choices=("dev", "test"), default="test")
    parser.add_argument("--global-index", type=int, default=750)
    parser.add_argument("--output-prefix", type=str, required=True)
    args = parser.parse_args()

    ckpt_path = os.path.abspath(args.checkpoint)
    if not os.path.isfile(ckpt_path):
        raise SystemExit(f"checkpoint not found: {ckpt_path}")

    cli = _default_cli(args.dataset)
    _, _, apply_ckpt_arch, parse_mselector = _load_deberta_dump_helpers()
    apply_ckpt_arch(cli, ckpt_path)
    if not getattr(cli, "ablation", None):
        cli.ablation = "none"
    global_configs.set_dataset_config(cli.dataset)

    model = InfoGate_DeBertaForSequenceClassification.from_pretrained(
        cli.model, multimodal_config=cli, num_labels=1
    )
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(DEVICE)
    model.eval()

    batch, tokenizer = _load_single_sample(cli, args.split, args.global_index)
    original = _forward_with_optional_mask(model, batch, None, parse_mselector)
    token_rows, valid_positions = _token_conf_arrays(original, tokenizer)

    for row, pos in zip(token_rows, valid_positions):
        masked = _forward_with_optional_mask(model, batch, pos, parse_mselector)
        row["pred_masked"] = masked["pred"]
        row["delta_pred"] = masked["pred"] - original["pred"]
        row["abs_delta_pred"] = abs(masked["pred"] - original["pred"])
        row["abs_err_masked"] = masked["abs_err"]
        row["delta_abs_err"] = masked["abs_err"] - original["abs_err"]

    csv_path = args.output_prefix + ".csv"
    _write_csv(csv_path, token_rows)
    _plot_heatmaps(args.output_prefix, token_rows)

    print(f"Original pred={original['pred']:+.6f} label={original['label']:+.6f} err={original['abs_err']:.6f}")
    print(f"Primary={SLOT_NAMES[int(original['primary_idx'])]} weights={original['weights']}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {args.output_prefix}_ordered.png/.pdf")
    print(f"Wrote {args.output_prefix}_sorted.png/.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
