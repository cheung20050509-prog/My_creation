#!/usr/bin/env python3
"""Export **token-level** VTB confidences for one split row (qualitative / appendix).

Training dumps aggregate with ``per_sample_ib_conf_mean`` (mask-mean over time, then
over bottleneck ``D``).  IB hooks store **per-position** ``conf = sigmoid(-logvar)``
as ``[B, T, D]``.  This script applies the **same** reduction over ``D`` only, leaving
one scalar per token: ``mean_D conf[b,t,:]``, masked to 0 on padding (see
``ib_conf_utils.per_token_ib_conf_mean_over_bottleneck``).

Supports:

- **DeBERTa** regression (MOSI / MOSEI): same data path as ``dump_dpr_primary_on_split.py``.
- **ALBERT + HCF** humor classification (MUStARD / UR-FUNNY): same hooks as
  ``dump_dpr_primary_classify.py`` (including ``forward4`` wrap when ``num_modalities==4``).

Usage (from ``My_creation/``)::

    conda run -n ITHP5090 python Qualitative_Evaluation/dump_ib_conf_token_level.py \\
      --backbone deberta \\
      --checkpoint logs/optuna/4090D_restart/phase1/checkpoints/optuna_mosei_phase1/trial_70/infogate_mosei_best.pt \\
      --dataset mosei --split test --global-index 2856 \\
      --output Qualitative_Evaluation/results/mosei_trial70_idx2856_ib_conf_tokens.tsv

    conda run -n ITHP5090 python Qualitative_Evaluation/dump_ib_conf_token_level.py \\
      --backbone albert \\
      --checkpoint logs/optuna/4090D_restart/classification/.../infogate_mustard_best.pt \\
      --dataset mustard --split test --global-index 7 \\
      --output Qualitative_Evaluation/results/mustard_idx7_ib_conf_tokens.tsv
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import pickle
import random
import sys
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

_QE_DIR = os.path.dirname(os.path.abspath(__file__))
_MY = os.path.dirname(_QE_DIR)
if _MY not in sys.path:
    sys.path.insert(0, _MY)

import global_configs  # noqa: E402
from global_configs import DEVICE  # noqa: E402

from ib_conf_utils import (  # noqa: E402
    install_infogate_ib_trace,
    per_token_ib_conf_mean_over_bottleneck,
)


def _load_deberta_dump_helpers():
    path = os.path.join(_QE_DIR, "dump_dpr_primary_on_split.py")
    spec = importlib.util.spec_from_file_location("_dpr_os_split", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod.convert_to_features, mod.build_tensor_dataset


def _apply_ckpt_arch_deberta(cli: SimpleNamespace, ckpt_path: str) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    saved = ckpt.get("args")
    if saved is None:
        return
    keys = (
        "model",
        "dataset",
        "max_seq_length",
        "unified_dim",
        "ib_hidden_dim",
        "bottleneck_dim",
        "num_heads",
        "num_infogate_layers",
        "dropout_prob",
        "beta_ib",
        "alpha_ib",
        "mse_weight",
        "selector_target_temp",
        "selector_balance_weight",
        "selector_rib_weight",
        "gumbel_tau_start",
        "gumbel_tau_end",
        "ablation",
    )
    for k in keys:
        if hasattr(saved, k):
            setattr(cli, k, getattr(saved, k))
    for flag, attr in (
        ("disable_l_lib", "use_l_lib"),
        ("disable_l_rib", "use_l_rib"),
    ):
        if hasattr(saved, flag):
            setattr(cli, attr, not bool(getattr(saved, flag)))


def _weights_and_primary_from_mselector_output(out):
    if not isinstance(out, tuple):
        return None, out if torch.is_tensor(out) else out[-1]
    if len(out) >= 6 and torch.is_tensor(out[3]) and out[3].dim() == 2:
        return out[3], out[5]
    if len(out) >= 7 and torch.is_tensor(out[4]) and out[4].dim() == 2:
        return out[4], out[6]
    return None, out[-1]


def _apply_ckpt_arch_albert(cli: SimpleNamespace, ckpt_path: str) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    saved = ckpt.get("args")
    if saved is None:
        return
    keys = (
        "model",
        "dataset",
        "max_seq_length",
        "unified_dim",
        "ib_hidden_dim",
        "bottleneck_dim",
        "num_heads",
        "num_infogate_layers",
        "dropout_prob",
        "beta_ib",
        "alpha_ib",
        "mse_weight",
        "selector_target_temp",
        "selector_balance_weight",
        "selector_rib_weight",
        "gumbel_tau_start",
        "gumbel_tau_end",
        "task_type",
    )
    for k in keys:
        if hasattr(saved, k):
            setattr(cli, k, getattr(saved, k))
    for flag, attr in (
        ("disable_l_lib", "use_l_lib"),
        ("disable_l_rib", "use_l_rib"),
    ):
        if hasattr(saved, flag):
            setattr(cli, attr, not bool(getattr(saved, flag)))


def _run_deberta(cli, ckpt_path: str, split: str, global_index: int, batch_size: int, out_path: str) -> int:
    from deberta_infogate import InfoGate_DeBertaForSequenceClassification  # noqa: E402
    from transformers import DebertaV2Tokenizer  # noqa: E402

    convert_to_features, build_tensor_dataset = _load_deberta_dump_helpers()

    model = InfoGate_DeBertaForSequenceClassification.from_pretrained(
        cli.model, multimodal_config=cli, num_labels=1
    )
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(DEVICE)
    model.eval()

    captured: list[tuple[torch.Tensor | None, torch.Tensor]] = []

    def _hook(_mod, _inp, out):
        w, p = _weights_and_primary_from_mselector_output(out)
        captured.append((w.detach().cpu() if w is not None else None, p.detach().cpu()))

    ig = model.dberta.infogate
    h = ig.mselector.register_forward_hook(_hook)
    cleanup_ib, ib_store = install_infogate_ib_trace(ig)

    ds_path = os.path.join(_MY, "datasets", f"{cli.dataset}.pkl")
    with open(ds_path, "rb") as fh:
        data = pickle.load(fh)
    if split == "dev":
        key = "dev" if "dev" in data else "valid"
        raw = data[key]
    else:
        raw = data["test"]
    if global_index < 0 or global_index >= len(raw):
        print(f"ERROR: global_index {global_index} out of range [0,{len(raw)-1}]", file=sys.stderr)
        return 1

    tok = DebertaV2Tokenizer.from_pretrained(cli.model)
    feats = convert_to_features(raw, cli.max_seq_length, tok, cli.dataset)
    ds = build_tensor_dataset(feats)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    batch_idx = global_index // batch_size
    inner = global_index % batch_size
    batch = None
    for i, b in enumerate(dl):
        if i == batch_idx:
            batch = b
            break
    if batch is None:
        print("ERROR: batch not found", file=sys.stderr)
        return 1

    pad_id = model.config.pad_token_id if model.config.pad_token_id is not None else 0
    try:
        batch = tuple(t.to(DEVICE) for t in batch)
        input_ids, visual, acoustic, labels = batch
        visual = visual.squeeze(1)
        acoustic = acoustic.squeeze(1)
        attn = input_ids.ne(pad_id).float()
        ib_store.clear()
        captured.clear()
        model(input_ids, visual, acoustic, labels=None, stage=2)
        if not {"t", "a", "v", "conf_p"}.issubset(ib_store.keys()):
            print(f"ERROR: ib_store keys {sorted(ib_store.keys())}", file=sys.stderr)
            return 1
        ids_row = input_ids[inner].detach().cpu().tolist()
        m_row = attn[inner].detach().cpu()
        toks = tok.convert_ids_to_tokens(ids_row)
        ct = per_token_ib_conf_mean_over_bottleneck(ib_store["t"], attn)[inner].detach().cpu()
        ca = per_token_ib_conf_mean_over_bottleneck(ib_store["a"], attn)[inner].detach().cpu()
        cv = per_token_ib_conf_mean_over_bottleneck(ib_store["v"], attn)[inner].detach().cpu()
        cf = per_token_ib_conf_mean_over_bottleneck(ib_store["conf_p"], attn)[inner].detach().cpu()
    finally:
        h.remove()
        cleanup_ib()

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(
            ["global_index", "pos", "token_id", "token", "valid", "ib_conf_t", "ib_conf_a", "ib_conf_v", "ib_conf_fused"]
        )
        for pos in range(len(ids_row)):
            w.writerow(
                [
                    global_index,
                    pos,
                    ids_row[pos],
                    toks[pos].replace("\t", " "),
                    int(m_row[pos].item() > 0.5),
                    f"{ct[pos].item():.8f}",
                    f"{ca[pos].item():.8f}",
                    f"{cv[pos].item():.8f}",
                    f"{cf[pos].item():.8f}",
                ]
            )
    print(f"Wrote {len(ids_row)} token rows to {out_path}")
    return 0


def _run_albert(cli, ckpt_path: str, split: str, global_index: int, batch_size: int, out_path: str) -> int:
    from albert_infogate import InfoGate_AlbertForSequenceClassification  # noqa: E402
    from data_humor import build_humor_loaders  # noqa: E402
    from transformers import AlbertTokenizer  # noqa: E402

    model = InfoGate_AlbertForSequenceClassification.from_pretrained(
        cli.model, multimodal_config=cli, num_labels=1
    )
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(DEVICE)
    model.eval()

    captured: list[tuple[torch.Tensor | None, torch.Tensor]] = []

    def _hook(_mod, _inp, out):
        w, p = _weights_and_primary_from_mselector_output(out)
        captured.append((w.detach().cpu() if w is not None else None, p.detach().cpu()))

    msel = model.albert.infogate.mselector
    n_mod = int(msel.num_modalities)
    hook_handle = None
    _orig_forward4 = None
    if n_mod == 3:
        hook_handle = msel.register_forward_hook(_hook)
    elif n_mod == 4:
        _orig_forward4 = msel.forward4

        def _forward4_capture(H_a, H_l, H_v, H_hcf, mask=None):
            out = _orig_forward4(H_a, H_l, H_v, H_hcf, mask)
            captured.append((out[4].detach().cpu(), out[6].detach().cpu()))
            return out

        msel.forward4 = _forward4_capture
    else:
        print(f"ERROR: unsupported MSelector.num_modalities={n_mod}", file=sys.stderr)
        return 1

    ig = model.albert.infogate
    cleanup_ib, ib_store = install_infogate_ib_trace(ig)

    hcf_dim = global_configs.HCF_DIM
    tok = AlbertTokenizer.from_pretrained(cli.model)
    _tr, dev_dl, test_dl, _nopt = build_humor_loaders(
        dataset=cli.dataset,
        tokenizer=tok,
        max_seq_length=cli.max_seq_length,
        acoustic_dim=global_configs.ACOUSTIC_DIM,
        visual_dim=global_configs.VISUAL_DIM,
        train_batch_size=batch_size,
        dev_batch_size=batch_size,
        test_batch_size=batch_size,
        gradient_accumulation_step=1,
        n_epochs=1,
        hcf_dim=hcf_dim,
        slice_hkt=True,
    )
    dl = dev_dl if split == "dev" else test_dl
    n = len(dl.dataset)
    if global_index < 0 or global_index >= n:
        print(f"ERROR: global_index {global_index} out of range [0,{n-1}]", file=sys.stderr)
        return 1

    batch_idx = global_index // batch_size
    inner = global_index % batch_size
    batch = None
    for i, b in enumerate(dl):
        if i == batch_idx:
            batch = b
            break
    if batch is None:
        print("ERROR: batch not found", file=sys.stderr)
        return 1

    use_hcf = hcf_dim > 0
    pad_id = model.config.pad_token_id if model.config.pad_token_id is not None else 0
    try:
        batch = tuple(t.to(DEVICE) for t in batch)
        if use_hcf:
            input_ids, visual, acoustic, hcf, _labels = batch
            hcf = hcf.squeeze(1)
        else:
            input_ids, visual, acoustic, _labels = batch
            hcf = None
        visual = visual.squeeze(1)
        acoustic = acoustic.squeeze(1)
        attn = input_ids.ne(pad_id).float()
        ib_store.clear()
        captured.clear()
        model(input_ids, visual, acoustic, hcf=hcf, stage=2)
        need = {"t", "a", "v", "conf_p"}
        if use_hcf:
            need.add("h")
        if not need.issubset(ib_store.keys()):
            print(f"ERROR: ib_store keys {sorted(ib_store.keys())}", file=sys.stderr)
            return 1
        ids_row = input_ids[inner].detach().cpu().tolist()
        m_row = attn[inner].detach().cpu()
        toks = tok.convert_ids_to_tokens(ids_row)
        ct = per_token_ib_conf_mean_over_bottleneck(ib_store["t"], attn)[inner].detach().cpu()
        ca = per_token_ib_conf_mean_over_bottleneck(ib_store["a"], attn)[inner].detach().cpu()
        cv = per_token_ib_conf_mean_over_bottleneck(ib_store["v"], attn)[inner].detach().cpu()
        cf = per_token_ib_conf_mean_over_bottleneck(ib_store["conf_p"], attn)[inner].detach().cpu()
        ch = (
            per_token_ib_conf_mean_over_bottleneck(ib_store["h"], attn)[inner].detach().cpu()
            if use_hcf
            else None
        )
    finally:
        if hook_handle is not None:
            hook_handle.remove()
        if _orig_forward4 is not None:
            msel.forward4 = _orig_forward4
        cleanup_ib()

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    header = ["global_index", "pos", "token_id", "token", "valid", "ib_conf_t", "ib_conf_a", "ib_conf_v"]
    if ch is not None:
        header.append("ib_conf_h")
    header.append("ib_conf_fused")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(header)
        for pos in range(len(ids_row)):
            row = [
                global_index,
                pos,
                ids_row[pos],
                toks[pos].replace("\t", " "),
                int(m_row[pos].item() > 0.5),
                f"{ct[pos].item():.8f}",
                f"{ca[pos].item():.8f}",
                f"{cv[pos].item():.8f}",
            ]
            if ch is not None:
                row.append(f"{ch[pos].item():.8f}")
            row.append(f"{cf[pos].item():.8f}")
            w.writerow(row)
    print(f"Wrote {len(ids_row)} token rows to {out_path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backbone", choices=("deberta", "albert"), required=True)
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--dataset", type=str, required=True)
    ap.add_argument("--split", choices=("dev", "test"), default="test")
    ap.add_argument("--global-index", type=int, required=True)
    ap.add_argument("--output", type=str, required=True)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=128)
    args = ap.parse_args()

    ckpt_path = os.path.abspath(args.checkpoint)
    if not os.path.isfile(ckpt_path):
        print(f"ERROR: checkpoint not found: {ckpt_path}", file=sys.stderr)
        return 1

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.backbone == "deberta":
        if args.dataset not in ("mosi", "mosei"):
            print("ERROR: deberta backbone expects --dataset mosi|mosei", file=sys.stderr)
            return 1
        global_configs.set_dataset_config(args.dataset)
        cli = SimpleNamespace(
            model=os.path.join(_MY, "deberta-v3-base"),
            dataset=args.dataset,
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
        _apply_ckpt_arch_deberta(cli, ckpt_path)
        if not getattr(cli, "ablation", None):
            cli.ablation = "none"
        return _run_deberta(cli, ckpt_path, args.split, args.global_index, args.batch_size, args.output)

    if args.dataset not in ("mustard", "ur_funny"):
        print("ERROR: albert backbone expects --dataset mustard|ur_funny", file=sys.stderr)
        return 1
    global_configs.set_dataset_config(args.dataset)
    cli = SimpleNamespace(
        model=os.path.join(_MY, "albert-base-v2"),
        dataset=args.dataset,
        max_seq_length=64,
        unified_dim=256,
        ib_hidden_dim=256,
        bottleneck_dim=128,
        num_heads=4,
        num_infogate_layers=3,
        dropout_prob=0.25,
        beta_ib=16.0,
        alpha_ib=0.005,
        mse_weight=0.0,
        selector_target_temp=0.6,
        selector_balance_weight=0.0,
        selector_rib_weight=0.05,
        gumbel_tau_start=1.0,
        gumbel_tau_end=0.5,
        task_type="binary",
        use_l_lib=True,
        use_l_rib=True,
    )
    _apply_ckpt_arch_albert(cli, ckpt_path)
    if not getattr(cli, "task_type", None):
        cli.task_type = "binary"
    if not hasattr(cli, "use_l_lib"):
        cli.use_l_lib = True
    if not hasattr(cli, "use_l_rib"):
        cli.use_l_rib = True
    if not hasattr(cli, "mse_weight"):
        cli.mse_weight = 0.0
    return _run_albert(cli, ckpt_path, args.split, args.global_index, args.batch_size, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
