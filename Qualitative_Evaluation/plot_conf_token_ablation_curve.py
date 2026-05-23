#!/usr/bin/env python3
"""Plot prediction sensitivity to confidence-sorted auxiliary token masking.

For a single sample, this keeps the model and input fixed, then masks a growing
fraction of auxiliary confidence positions sorted by VTB confidence. Two curves
are produced: masking the lowest-confidence positions and masking the
highest-confidence positions.

Run from ``My_creation/``::

    CUDA_VISIBLE_DEVICES="" python Qualitative_Evaluation/plot_conf_token_ablation_curve.py \
      --checkpoint logs/optuna/4090D_restart/phase1/checkpoints/optuna_mosei_phase1/trial_70/infogate_mosei_best.pt \
      --dataset mosei --split test --global-index 750 \
      --output-prefix Qualitative_Evaluation/results/mosei_trial70_idx750_conf_token_ablation
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
from ib_conf_utils import install_infogate_ib_trace, per_sample_ib_conf_mean  # noqa: E402
from plot_conf_intervention_prediction import (  # noqa: E402
    _default_cli,
    _load_deberta_dump_helpers,
    _load_single_sample,
)


def _mask_conf_by_rank(
    conf: torch.Tensor,
    tok_mask: torch.Tensor | None,
    ratio: float,
    direction: str,
) -> torch.Tensor:
    """Set selected token confidence vectors to zero by confidence rank."""
    if ratio <= 0:
        return conf
    out = conf.clone()
    pos_conf = conf.mean(dim=-1)  # [B, T]
    for b in range(conf.size(0)):
        if tok_mask is None:
            valid_idx = torch.arange(conf.size(1), device=conf.device)
        else:
            valid_idx = torch.nonzero(tok_mask[b].bool(), as_tuple=False).flatten()
        if valid_idx.numel() == 0:
            continue
        k = int(round(float(ratio) * int(valid_idx.numel())))
        k = max(0, min(k, int(valid_idx.numel())))
        if k == 0:
            continue
        vals = pos_conf[b, valid_idx]
        order = torch.argsort(vals, descending=(direction == "high"))
        selected = valid_idx[order[:k]]
        out[b, selected, :] = 0.0
    return out


def _install_token_ablation(ig_module, ratio: float, direction: str) -> Callable[[], None]:
    original_forward = ig_module.forward

    def patched_forward(self, B_p, conf_p, B_aux_list, conf_aux_list, tok_mask=None):
        new_conf = [
            _mask_conf_by_rank(conf, tok_mask, ratio, direction)
            for conf in conf_aux_list
        ]
        return original_forward(B_p, conf_p, B_aux_list, new_conf, tok_mask)

    ig_module.forward = MethodType(patched_forward, ig_module)

    def cleanup() -> None:
        ig_module.forward = original_forward

    return cleanup


def _run_one(model, batch, ratio: float, direction: str, parse_mselector) -> dict[str, float | str]:
    ig = model.dberta.infogate
    pad_id = model.config.pad_token_id if model.config.pad_token_id is not None else 0
    captured = []

    def mselector_hook(_mod, _inp, out):
        weights, primary = parse_mselector(out)
        captured.append((weights.detach().cpu() if weights is not None else None, primary.detach().cpu()))

    hook = ig.mselector.register_forward_hook(mselector_hook)
    cleanup_ib, ib_store = install_infogate_ib_trace(ig)
    cleanup_ablation = _install_token_ablation(ig.infogate, ratio, direction)

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
        row = {
            "direction": direction,
            "mask_ratio": float(ratio),
            "label": label,
            "pred": pred,
            "abs_err": abs(pred - label),
            "primary_idx": int(primary.view(-1)[0].item()),
            "primary_name": {0: "acoustic", 1: "language", 2: "visual"}.get(
                int(primary.view(-1)[0].item()), "unknown"
            ),
            "w_a": float(weights[0, 0].item()),
            "w_l": float(weights[0, 1].item()),
            "w_v": float(weights[0, 2].item()),
        }
        for key, out_key in (("t", "ib_conf_t"), ("a", "ib_conf_a"), ("v", "ib_conf_v"), ("conf_p", "ib_conf_fused")):
            if key in ib_store:
                row[out_key] = float(per_sample_ib_conf_mean(ib_store[key], attention_mask)[0].detach().cpu().item())
        return row
    finally:
        cleanup_ablation()
        cleanup_ib()
        hook.remove()


def _write_csv(path: str, rows: list[dict[str, float | str]]) -> None:
    fieldnames = [
        "direction", "mask_ratio", "label", "pred", "abs_err",
        "primary_idx", "primary_name", "w_a", "w_l", "w_v",
        "ib_conf_t", "ib_conf_a", "ib_conf_v", "ib_conf_fused",
    ]
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot(prefix: str, rows: list[dict[str, float | str]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    label = float(rows[0]["label"])
    by_dir = {"low": [], "high": []}
    for row in rows:
        by_dir[str(row["direction"])].append(row)
    for direction in by_dir:
        by_dir[direction].sort(key=lambda r: float(r["mask_ratio"]))

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.2), dpi=180)
    style = {
        "low": dict(color="#4c78a8", marker="o", label="mask low-conf first"),
        "high": dict(color="#f58518", marker="s", label="mask high-conf first"),
    }

    for direction, items in by_dir.items():
        xs = [float(r["mask_ratio"]) * 100.0 for r in items]
        preds = [float(r["pred"]) for r in items]
        errs = [float(r["abs_err"]) for r in items]
        axes[0].plot(xs, preds, linewidth=2.0, **style[direction])
        axes[1].plot(xs, errs, linewidth=2.0, **style[direction])

    axes[0].axhline(label, color="#d62728", linestyle="--", linewidth=1.2, label=f"label={label:.3g}")
    axes[0].set_title("Prediction under confidence-sorted masking")
    axes[0].set_ylabel("Prediction score")
    axes[0].set_xlabel("Masked auxiliary tokens (%)")

    axes[1].set_title("Error sensitivity")
    axes[1].set_ylabel(r"Absolute error $|\hat{y}-y|$")
    axes[1].set_xlabel("Masked auxiliary tokens (%)")

    for ax in axes:
        ax.grid(axis="y", linestyle=":", alpha=0.35)
        ax.set_xticks([0, 10, 20, 30, 40, 50])
        ax.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    for ext in (".png", ".pdf"):
        fig.savefig(prefix + ext, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, choices=("mosi", "mosei"), default="mosei")
    parser.add_argument("--split", type=str, choices=("dev", "test"), default="test")
    parser.add_argument("--global-index", type=int, default=750)
    parser.add_argument("--output-prefix", type=str, required=True)
    parser.add_argument("--ratios", type=str, default="0,0.1,0.2,0.3,0.4,0.5")
    parser.add_argument("--seed", type=int, default=128)
    args = parser.parse_args()

    ckpt_path = os.path.abspath(args.checkpoint)
    if not os.path.isfile(ckpt_path):
        raise SystemExit(f"checkpoint not found: {ckpt_path}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

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

    batch, _ = _load_single_sample(cli, args.split, args.global_index)
    ratios = [float(x.strip()) for x in args.ratios.split(",") if x.strip()]

    rows = []
    for direction in ("low", "high"):
        for ratio in ratios:
            rows.append(_run_one(model, batch, ratio, direction, parse_mselector))

    csv_path = args.output_prefix + ".csv"
    _write_csv(csv_path, rows)
    _plot(args.output_prefix, rows)

    print(f"Wrote {csv_path}")
    print(f"Wrote {args.output_prefix}.png")
    print(f"Wrote {args.output_prefix}.pdf")
    for row in rows:
        print(
            f"{row['direction']:>4s} {float(row['mask_ratio']):.1%}: "
            f"pred={float(row['pred']):+.6f} err={float(row['abs_err']):.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
