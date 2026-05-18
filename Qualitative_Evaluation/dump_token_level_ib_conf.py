#!/usr/bin/env python3
"""Export **token-level** VTB confidences for one sample (mask-aligned).

Hooks capture ``conf`` tensors ``[B, T, D]`` (see ``install_infogate_ib_trace``).
This script reduces **D** with a mean so each position gets one scalar per slot
(same per-D reduction as ``per_sample_ib_conf_mean`` before the temporal mean).

Outputs CSV columns::

    pos, input_id, subword, is_valid,
    conf_t, conf_a, conf_v[, conf_h], conf_fused

``conf_fused`` is mean over D of ``conf_p`` (primary-routed mixed conf).
Padded positions have ``is_valid=0`` and conf columns set to 0.

Usage (from ``My_creation/``)::

    conda run -n ITHP5090 python Qualitative_Evaluation/dump_token_level_ib_conf.py \\
      --backend deberta \\
      --checkpoint logs/optuna/4090D_restart/phase1/checkpoints/optuna_mosei_phase1/trial_70/infogate_mosei_best.pt \\
      --dataset mosei --split test --sample_index 2856 \\
      --output Qualitative_Evaluation/results/mosei_trial70_idx2856_token_ib_conf.csv

    conda run -n ITHP5090 python Qualitative_Evaluation/dump_token_level_ib_conf.py \\
      --backend albert \\
      --checkpoint logs/optuna/4090D_restart/classification/.../infogate_mustard_best.pt \\
      --dataset mustard --split test --sample_index 7 \\
      --output Qualitative_Evaluation/results/mustard_trial113_idx7_token_ib_conf.csv
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
from transformers import AlbertTokenizer, DebertaV2Tokenizer

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


def _load_sidecar_module(filename: str):
    path = os.path.join(_QE_DIR, filename)
    spec = importlib.util.spec_from_file_location("_qe_sidecar", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _apply_ckpt_deberta(cli: SimpleNamespace, ckpt_path: str) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    saved = ckpt.get("args")
    if saved is None:
        return
    keys = (
        "model", "dataset", "max_seq_length", "unified_dim", "ib_hidden_dim",
        "bottleneck_dim", "num_heads", "num_infogate_layers", "dropout_prob",
        "beta_ib", "alpha_ib", "mse_weight", "selector_target_temp",
        "selector_balance_weight", "selector_rib_weight", "gumbel_tau_start",
        "gumbel_tau_end", "ablation",
    )
    for k in keys:
        if hasattr(saved, k):
            setattr(cli, k, getattr(saved, k))
    for flag, attr in (("disable_l_lib", "use_l_lib"), ("disable_l_rib", "use_l_rib")):
        if hasattr(saved, flag):
            setattr(cli, attr, not bool(getattr(saved, flag)))


def _apply_ckpt_albert(cli: SimpleNamespace, ckpt_path: str) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    saved = ckpt.get("args")
    if saved is None:
        return
    keys = (
        "model", "dataset", "max_seq_length", "unified_dim", "ib_hidden_dim",
        "bottleneck_dim", "num_heads", "num_infogate_layers", "dropout_prob",
        "beta_ib", "alpha_ib", "mse_weight", "selector_target_temp",
        "selector_balance_weight", "selector_rib_weight", "gumbel_tau_start",
        "gumbel_tau_end", "task_type",
    )
    for k in keys:
        if hasattr(saved, k):
            setattr(cli, k, getattr(saved, k))
    for flag, attr in (("disable_l_lib", "use_l_lib"), ("disable_l_rib", "use_l_rib")):
        if hasattr(saved, flag):
            setattr(cli, attr, not bool(getattr(saved, flag)))


def run_deberta(cli: SimpleNamespace, ckpt_path: str, split: str, sample_index: int, output: str) -> int:
    from deberta_infogate import InfoGate_DeBertaForSequenceClassification  # noqa: E402

    dumper = _load_sidecar_module("dump_dpr_primary_on_split.py")
    convert_to_features = dumper.convert_to_features
    build_tensor_dataset = dumper.build_tensor_dataset

    global_configs.set_dataset_config(cli.dataset)
    random.seed(128)
    np.random.seed(128)
    torch.manual_seed(128)

    model = InfoGate_DeBertaForSequenceClassification.from_pretrained(
        cli.model, multimodal_config=cli, num_labels=1
    )
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(DEVICE)
    model.eval()

    ig = model.dberta.infogate
    cleanup_ib, ib_store = install_infogate_ib_trace(ig)

    ds_path = os.path.join(_MY, "datasets", f"{cli.dataset}.pkl")
    with open(ds_path, "rb") as fh:
        data = pickle.load(fh)
    key = "dev" if split == "dev" and "dev" in data else (
        "valid" if split == "dev" and "valid" in data else "test"
    )
    if split == "dev" and key not in ("dev", "valid"):
        print("ERROR: no dev/valid in pickle", file=sys.stderr)
        return 1
    raw_list = data["test"] if key == "test" else data[key]
    if sample_index < 0 or sample_index >= len(raw_list):
        print(f"ERROR: sample_index {sample_index} out of range [0,{len(raw_list)})", file=sys.stderr)
        return 1

    tok = DebertaV2Tokenizer.from_pretrained(cli.model)
    feats = convert_to_features([raw_list[sample_index]], cli.max_seq_length, tok, cli.dataset)
    ds = build_tensor_dataset(feats)
    dl = DataLoader(ds, batch_size=1, shuffle=False)

    pad_id = model.config.pad_token_id if model.config.pad_token_id is not None else 0
    try:
        with torch.no_grad():
            batch = next(iter(dl))
            batch = tuple(t.to(DEVICE) for t in batch)
            input_ids, visual, acoustic, labels = batch
            visual = visual.squeeze(1)
            acoustic = acoustic.squeeze(1)
            attn = input_ids.ne(pad_id).float()
            ib_store.clear()
            model(input_ids, visual, acoustic, labels=None, stage=2)
            need = {"t", "a", "v", "conf_p"}
            if not need.issubset(ib_store.keys()):
                print("ERROR: IB trace missing", sorted(ib_store.keys()), file=sys.stderr)
                return 1
            ct = per_token_ib_conf_mean_over_bottleneck(ib_store["t"], attn)[0].detach().cpu().numpy()
            ca = per_token_ib_conf_mean_over_bottleneck(ib_store["a"], attn)[0].detach().cpu().numpy()
            cv = per_token_ib_conf_mean_over_bottleneck(ib_store["v"], attn)[0].detach().cpu().numpy()
            cf = per_token_ib_conf_mean_over_bottleneck(ib_store["conf_p"], attn)[0].detach().cpu().numpy()
    finally:
        cleanup_ib()

    ids = input_ids[0].detach().cpu().tolist()
    toks = tok.convert_ids_to_tokens(ids)
    mask = attn[0].detach().cpu().numpy()
    T = len(ids)
    out_dir = os.path.dirname(os.path.abspath(output)) or "."
    os.makedirs(out_dir, exist_ok=True)
    fieldnames = [
        "pos", "input_id", "subword", "is_valid",
        "conf_t", "conf_a", "conf_v", "conf_fused",
    ]
    with open(output, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for pos in range(T):
            valid = int(mask[pos] >= 0.5)
            w.writerow(
                {
                    "pos": pos,
                    "input_id": ids[pos],
                    "subword": toks[pos] if pos < len(toks) else "",
                    "is_valid": valid,
                    "conf_t": float(ct[pos]) if valid else 0.0,
                    "conf_a": float(ca[pos]) if valid else 0.0,
                    "conf_v": float(cv[pos]) if valid else 0.0,
                    "conf_fused": float(cf[pos]) if valid else 0.0,
                }
            )
    print(f"Wrote {T} token rows to {output}")
    return 0


def run_albert(cli: SimpleNamespace, ckpt_path: str, split: str, sample_index: int, output: str) -> int:
    from albert_infogate import InfoGate_AlbertForSequenceClassification  # noqa: E402
    from data_humor import convert_to_features, features_to_dataset  # noqa: E402

    classify = _load_sidecar_module("dump_dpr_primary_classify.py")

    global_configs.set_dataset_config(cli.dataset)
    hcf_dim = global_configs.HCF_DIM
    random.seed(128)
    np.random.seed(128)
    torch.manual_seed(128)

    model = InfoGate_AlbertForSequenceClassification.from_pretrained(
        cli.model, multimodal_config=cli, num_labels=1
    )
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(DEVICE)
    model.eval()

    captured: list = []

    def _hook(_mod, _inp, out):
        w, p = classify._weights_and_primary_from_mselector_output(out)
        captured.append((w, p))

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
            captured.append((out[4], out[6]))
            return out

        msel.forward4 = _forward4_capture
    else:
        print(f"ERROR: num_modalities={n_mod}", file=sys.stderr)
        return 1

    ig = model.albert.infogate
    cleanup_ib, ib_store = install_infogate_ib_trace(ig)

    pkl_path = os.path.join(_MY, "datasets", "mustard.pkl" if cli.dataset == "mustard" else "ur_funny.pkl")
    if cli.dataset == "ur_funny":
        for name in ("ur_funny.pkl", "urfunnyv2.pkl"):
            cand = os.path.join(_MY, "datasets", name)
            if os.path.isfile(cand):
                pkl_path = cand
                break
    with open(pkl_path, "rb") as fh:
        data = pickle.load(fh)
    key = "dev" if split == "dev" and "dev" in data else (
        "valid" if split == "dev" and "valid" in data else "test"
    )
    raw_list = data["test"] if key == "test" else data[key]
    if sample_index < 0 or sample_index >= len(raw_list):
        print(f"ERROR: sample_index out of range [0,{len(raw_list)})", file=sys.stderr)
        return 1

    tok = AlbertTokenizer.from_pretrained(cli.model)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    feats = convert_to_features(
        [raw_list[sample_index]],
        tok,
        cli.max_seq_length,
        global_configs.ACOUSTIC_DIM,
        global_configs.VISUAL_DIM,
        pad_id,
        hcf_dim=hcf_dim,
        slice_hkt=True,
    )
    ds = features_to_dataset(feats, include_hcf=hcf_dim > 0)
    dl = DataLoader(ds, batch_size=1, shuffle=False)

    try:
        with torch.no_grad():
            batch = next(iter(dl))
            batch = tuple(t.to(DEVICE) for t in batch)
            if hcf_dim > 0:
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
            if hcf_dim > 0:
                need.add("h")
            if not need.issubset(ib_store.keys()):
                print("ERROR: IB trace missing", sorted(ib_store.keys()), file=sys.stderr)
                return 1
            ct = per_token_ib_conf_mean_over_bottleneck(ib_store["t"], attn)[0].detach().cpu().numpy()
            ca = per_token_ib_conf_mean_over_bottleneck(ib_store["a"], attn)[0].detach().cpu().numpy()
            cv = per_token_ib_conf_mean_over_bottleneck(ib_store["v"], attn)[0].detach().cpu().numpy()
            cf = per_token_ib_conf_mean_over_bottleneck(ib_store["conf_p"], attn)[0].detach().cpu().numpy()
            ch = None
            if hcf_dim > 0:
                ch = per_token_ib_conf_mean_over_bottleneck(ib_store["h"], attn)[0].detach().cpu().numpy()
    finally:
        cleanup_ib()
        if hook_handle is not None:
            hook_handle.remove()
        if _orig_forward4 is not None:
            msel.forward4 = _orig_forward4

    ids = input_ids[0].detach().cpu().tolist()
    toks = tok.convert_ids_to_tokens(ids)
    mask = attn[0].detach().cpu().numpy()
    T = len(ids)
    out_dir = os.path.dirname(os.path.abspath(output)) or "."
    os.makedirs(out_dir, exist_ok=True)
    fieldnames = ["pos", "input_id", "subword", "is_valid", "conf_t", "conf_a", "conf_v"]
    if ch is not None:
        fieldnames.append("conf_h")
    fieldnames.append("conf_fused")
    with open(output, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for pos in range(T):
            valid = int(mask[pos] >= 0.5)
            row = {
                "pos": pos,
                "input_id": ids[pos],
                "subword": toks[pos] if pos < len(toks) else "",
                "is_valid": valid,
                "conf_t": float(ct[pos]) if valid else 0.0,
                "conf_a": float(ca[pos]) if valid else 0.0,
                "conf_v": float(cv[pos]) if valid else 0.0,
                "conf_fused": float(cf[pos]) if valid else 0.0,
            }
            if ch is not None:
                row["conf_h"] = float(ch[pos]) if valid else 0.0
            w.writerow(row)
    print(f"Wrote {T} token rows to {output}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", choices=("deberta", "albert"), required=True)
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--split", type=str, choices=("dev", "test"), default="test")
    ap.add_argument("--sample_index", type=int, required=True)
    ap.add_argument("--output", type=str, required=True)
    ap.add_argument(
        "--model",
        type=str,
        default="",
        help="Override backbone dir (default: deberta under My_creation/ or ckpt args).",
    )
    ap.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=("mosi", "mosei", "mustard", "ur_funny"),
        help="With --backend deberta: mosi|mosei. With albert: mustard|ur_funny.",
    )
    args = ap.parse_args()

    ckpt_path = os.path.abspath(args.checkpoint)
    if not os.path.isfile(ckpt_path):
        print(f"ERROR: checkpoint not found: {ckpt_path}", file=sys.stderr)
        return 1

    if args.backend == "deberta":
        if args.dataset not in ("mosi", "mosei"):
            print("ERROR: deberta backend requires --dataset mosi or mosei", file=sys.stderr)
            return 1

        model_dir = args.model or os.path.join(_MY, "deberta-v3-base")
        cli = SimpleNamespace(
            model=model_dir,
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
        _apply_ckpt_deberta(cli, ckpt_path)
        if not getattr(cli, "ablation", None):
            cli.ablation = "none"
        return run_deberta(cli, ckpt_path, args.split, args.sample_index, args.output)

    if args.dataset not in ("mustard", "ur_funny"):
        print("ERROR: albert backend requires --dataset mustard or ur_funny", file=sys.stderr)
        return 1

    model_dir = args.model or os.path.join(_MY, "albert-base-v2")
    cli = SimpleNamespace(
        model=model_dir,
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
    _apply_ckpt_albert(cli, ckpt_path)
    if not getattr(cli, "task_type", None):
        cli.task_type = "binary"
    return run_albert(cli, ckpt_path, args.split, args.sample_index, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
