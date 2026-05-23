#!/usr/bin/env python3
"""Plot how confidence interventions change one sample's prediction.

This script is intended for qualitative analysis. It keeps the trained model
fixed and only changes the auxiliary confidence tensors consumed by InfoGate
during a single forward pass.

Usage from ``My_creation/``::

    python Qualitative_Evaluation/plot_conf_intervention_prediction.py \
      --checkpoint logs/optuna/4090D_restart/phase1/checkpoints/optuna_mosei_phase1/trial_70/infogate_mosei_best.pt \
      --dataset mosei --split test --global-index 750 \
      --output-prefix Qualitative_Evaluation/results/mosei_trial70_idx750_conf_intervention
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import pickle
import random
import sys
from types import MethodType, SimpleNamespace
from typing import Callable

import numpy as np
import torch

_QE_DIR = os.path.dirname(os.path.abspath(__file__))
_MY = os.path.dirname(_QE_DIR)
if _MY not in sys.path:
    sys.path.insert(0, _MY)

import global_configs  # noqa: E402
from deberta_infogate import InfoGate_DeBertaForSequenceClassification  # noqa: E402
from global_configs import DEVICE  # noqa: E402
from ib_conf_utils import install_infogate_ib_trace, per_sample_ib_conf_mean  # noqa: E402
from simsv2_qual_utils import (  # noqa: E402
    apply_ckpt_arch_simsv2,
    default_simsv2_cli,
    forward_simsv2,
    load_simsv2_model,
    load_single_simsv2_batch,
    simsv2_infogate,
    unpack_simsv2_batch,
)


def _load_deberta_dump_helpers():
    path = os.path.join(_QE_DIR, "dump_dpr_primary_on_split.py")
    spec = importlib.util.spec_from_file_location("_dpr_helpers", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod.convert_to_features, mod.build_tensor_dataset, mod._apply_ckpt_arch, mod._weights_and_primary_from_mselector_output


def _default_cli(dataset: str) -> SimpleNamespace:
    if dataset == "simsv2":
        return default_simsv2_cli()
    return SimpleNamespace(
        model=os.path.join(_MY, "deberta-v3-base"),
        dataset=dataset,
        max_seq_length=50,
        unified_dim=256,
        ib_hidden_dim=256,
        bottleneck_dim=128,
        num_heads=4,
        num_infogate_layers=3,
        dropout_prob=0.1,
        beta_ib=32.0,
        alpha_ib=0.01,
        mse_weight=0.5,
        selector_target_temp=0.35,
        selector_balance_weight=0.0,
        selector_rib_weight=0.05,
        gumbel_tau_start=1.0,
        gumbel_tau_end=0.5,
        ablation="none",
        use_l_lib=True,
        use_l_rib=True,
    )


def _masked_mean_conf(conf: torch.Tensor, tok_mask: torch.Tensor | None) -> torch.Tensor:
    if tok_mask is None:
        return conf.mean(dim=(1, 2), keepdim=True)
    mask = tok_mask.float().unsqueeze(-1)
    denom = mask.sum(dim=(1, 2), keepdim=True).clamp_min(1.0) * conf.size(-1)
    return (conf * mask).sum(dim=(1, 2), keepdim=True) / denom


def _mode_transform(conf: torch.Tensor, tok_mask: torch.Tensor | None, mode: str, *, seed: int) -> torch.Tensor:
    if mode == "original":
        return conf
    if mode == "ones":
        return torch.ones_like(conf)
    if mode == "mean":
        return _masked_mean_conf(conf, tok_mask).expand_as(conf)
    if mode == "invert":
        return (1.0 - conf).clamp(0.0, 1.0)
    if mode == "shuffle":
        # Deterministic token-level shuffle per sample; bottleneck dimensions remain paired.
        out = conf.clone()
        gen = torch.Generator(device=conf.device)
        gen.manual_seed(seed)
        for b in range(conf.size(0)):
            if tok_mask is None:
                valid_idx = torch.arange(conf.size(1), device=conf.device)
            else:
                valid_idx = torch.nonzero(tok_mask[b].bool(), as_tuple=False).flatten()
            if valid_idx.numel() <= 1:
                continue
            perm = valid_idx[torch.randperm(valid_idx.numel(), generator=gen, device=conf.device)]
            out[b, valid_idx, :] = conf[b, perm, :]
        return out
    if mode == "low_mask":
        mean_pos = conf.mean(dim=-1, keepdim=True)
        if tok_mask is not None:
            valid = tok_mask.bool().unsqueeze(-1)
            safe_vals = mean_pos.masked_fill(~valid, float("inf"))
            thresh = torch.quantile(safe_vals.flatten(1), 0.35, dim=1, keepdim=True).view(-1, 1, 1)
        else:
            thresh = torch.quantile(mean_pos.flatten(1), 0.35, dim=1, keepdim=True).view(-1, 1, 1)
        keep = (mean_pos >= thresh).float()
        return conf * keep
    raise ValueError(f"unknown confidence mode: {mode}")


def _install_conf_intervention(ig_module, mode: str, seed: int) -> Callable[[], None]:
    original_forward = ig_module.forward

    def patched_forward(self, B_p, conf_p, B_aux_list, conf_aux_list, tok_mask=None):
        if mode == "original":
            new_conf = conf_aux_list
        else:
            new_conf = [
                _mode_transform(conf, tok_mask, mode, seed=seed + i * 9973)
                for i, conf in enumerate(conf_aux_list)
            ]
        return original_forward(B_p, conf_p, B_aux_list, new_conf, tok_mask)

    ig_module.forward = MethodType(patched_forward, ig_module)

    def cleanup() -> None:
        ig_module.forward = original_forward

    return cleanup


def _load_single_sample(cli: SimpleNamespace, split: str, global_index: int):
    from torch.utils.data import DataLoader
    from transformers import BertTokenizer, DebertaV2Tokenizer

    if cli.dataset == "simsv2":
        batch, _inner = load_single_simsv2_batch(cli, split, global_index, batch_size=1)
        tokenizer = BertTokenizer.from_pretrained(cli.model)
        return batch, tokenizer

    convert_to_features, build_tensor_dataset, _, _ = _load_deberta_dump_helpers()
    data_path = os.path.join(_MY, "datasets", f"{cli.dataset}.pkl")
    with open(data_path, "rb") as handle:
        data = pickle.load(handle)
    if split == "dev":
        split_key = "dev" if "dev" in data else "valid"
    else:
        split_key = "test"
    raw = data[split_key]
    if global_index < 0 or global_index >= len(raw):
        raise SystemExit(f"global_index {global_index} out of range [0,{len(raw)-1}] for {split_key}")

    tokenizer = DebertaV2Tokenizer.from_pretrained(cli.model)
    feats = convert_to_features([raw[global_index]], cli.max_seq_length, tokenizer, cli.dataset)
    ds = build_tensor_dataset(feats)
    batch = next(iter(DataLoader(ds, batch_size=1, shuffle=False)))
    return tuple(t.to(DEVICE) for t in batch), tokenizer


def _run_one_mode(model, batch, mode: str, seed: int, parse_mselector) -> dict[str, float | str]:
    is_simsv2 = len(batch) >= 6
    ig = simsv2_infogate(model) if is_simsv2 else model.dberta.infogate
    pad_id = model.config.pad_token_id if model.config.pad_token_id is not None else 0

    captured = []

    def mselector_hook(_mod, _inp, out):
        weights, primary = parse_mselector(out)
        captured.append((weights.detach().cpu() if weights is not None else None, primary.detach().cpu()))

    hook = ig.mselector.register_forward_hook(mselector_hook)
    cleanup_ib, ib_store = install_infogate_ib_trace(ig)
    cleanup_intervention = _install_conf_intervention(ig.infogate, mode, seed)

    try:
        if is_simsv2:
            input_ids, visual, acoustic, labels, input_mask, segment_ids = unpack_simsv2_batch(batch)
            attention_mask = input_mask.float()
        else:
            input_ids, visual, acoustic, labels = batch
            visual = visual.squeeze(1)
            acoustic = acoustic.squeeze(1)
            attention_mask = input_ids.ne(pad_id).float()
        ib_store.clear()
        captured.clear()
        with torch.no_grad():
            if is_simsv2:
                logits, _, _, _ = forward_simsv2(
                    model, input_ids, visual, acoustic,
                    labels=None, input_mask=input_mask, segment_ids=segment_ids,
                )
            else:
                logits, _, _, _ = model(input_ids, visual, acoustic, labels=None, stage=2)
        if not captured:
            raise RuntimeError("mselector hook did not fire")
        weights, primary = captured[-1]
        if weights is None:
            raise RuntimeError("could not parse DPR weights")

        pred = float(logits.view(-1)[0].detach().cpu().item())
        label = float(labels.view(-1)[0].detach().cpu().item())
        row = {
            "mode": mode,
            "label": label,
            "pred": pred,
            "abs_err": abs(pred - label),
            "delta_pred": 0.0,
            "primary_idx": int(primary.view(-1)[0].item()),
            "primary_name": {0: "acoustic", 1: "language", 2: "visual"}.get(int(primary.view(-1)[0].item()), "unknown"),
            "w_a": float(weights[0, 0].item()),
            "w_l": float(weights[0, 1].item()),
            "w_v": float(weights[0, 2].item()),
        }
        for key, out_key in (("t", "ib_conf_t"), ("a", "ib_conf_a"), ("v", "ib_conf_v"), ("conf_p", "ib_conf_fused")):
            if key in ib_store:
                row[out_key] = float(per_sample_ib_conf_mean(ib_store[key], attention_mask)[0].detach().cpu().item())
        return row
    finally:
        cleanup_intervention()
        cleanup_ib()
        hook.remove()


def _write_csv(path: str, rows: list[dict[str, float | str]]) -> None:
    fieldnames = [
        "mode", "label", "pred", "abs_err", "delta_pred",
        "primary_idx", "primary_name", "w_a", "w_l", "w_v",
        "ib_conf_t", "ib_conf_a", "ib_conf_v", "ib_conf_fused",
    ]
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot(path_prefix: str, rows: list[dict[str, float | str]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    modes = [str(r["mode"]) for r in rows]
    preds = [float(r["pred"]) for r in rows]
    errors = [float(r["abs_err"]) for r in rows]
    label = float(rows[0]["label"])

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.2), dpi=180)
    colors = ["#4c78a8" if m == "original" else "#f58518" for m in modes]

    axes[0].bar(modes, preds, color=colors)
    axes[0].axhline(label, color="#d62728", linestyle="--", linewidth=1.3, label=f"label={label:.3g}")
    axes[0].set_ylabel("Prediction score")
    axes[0].set_title("Prediction shift")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].bar(modes, errors, color=colors)
    axes[1].set_ylabel(r"Absolute error $|\hat{y}-y|$")
    axes[1].set_title("Error after confidence intervention")

    for ax in axes:
        ax.tick_params(axis="x", rotation=28)
        ax.grid(axis="y", linestyle=":", alpha=0.35)

    fig.tight_layout()
    for ext in (".png", ".pdf"):
        fig.savefig(path_prefix + ext, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, choices=("mosi", "mosei", "simsv2"), default="mosei")
    parser.add_argument("--split", type=str, choices=("dev", "test"), default="test")
    parser.add_argument("--global-index", type=int, default=750)
    parser.add_argument("--output-prefix", type=str, required=True)
    parser.add_argument(
        "--modes",
        type=str,
        default="original,ones,mean,shuffle,invert,low_mask",
        help="Comma-separated confidence modes.",
    )
    parser.add_argument("--seed", type=int, default=128)
    args = parser.parse_args()

    ckpt_path = os.path.abspath(args.checkpoint)
    if not os.path.isfile(ckpt_path):
        raise SystemExit(f"checkpoint not found: {ckpt_path}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cli = _default_cli(args.dataset)
    _, _, apply_ckpt_arch, parse_mselector = _load_deberta_dump_helpers()
    if args.dataset == "simsv2":
        apply_ckpt_arch_simsv2(cli, ckpt_path)
    else:
        apply_ckpt_arch(cli, ckpt_path)
    if not getattr(cli, "ablation", None):
        cli.ablation = "none"
    global_configs.set_dataset_config(cli.dataset)

    if args.dataset == "simsv2":
        model = load_simsv2_model(cli, ckpt_path)
    else:
        model = InfoGate_DeBertaForSequenceClassification.from_pretrained(
            cli.model, multimodal_config=cli, num_labels=1
        )
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        model.to(DEVICE)
        model.eval()

    batch, _ = _load_single_sample(cli, args.split, args.global_index)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    rows = []
    for mode in modes:
        rows.append(_run_one_mode(model, batch, mode, args.seed, parse_mselector))

    original_pred = float(rows[0]["pred"])
    for row in rows:
        row["delta_pred"] = float(row["pred"]) - original_pred

    csv_path = args.output_prefix + ".csv"
    _write_csv(csv_path, rows)
    _plot(args.output_prefix, rows)

    print(f"Wrote {csv_path}")
    print(f"Wrote {args.output_prefix}.png")
    print(f"Wrote {args.output_prefix}.pdf")
    for row in rows:
        print(
            f"{row['mode']:>9s}: pred={float(row['pred']):+.6f} "
            f"err={float(row['abs_err']):.6f} delta={float(row['delta_pred']):+.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
