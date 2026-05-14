#!/usr/bin/env python3
"""Export per-sample DPR primary modality indices (eval, deterministic routing).

``logs/optuna/4090D_restart/*.log`` only contain epoch-level ``Diag`` batch means;
they cannot identify which sample is A/V primary.  This script loads a checkpoint,
runs ``model.eval()`` on dev or test, and writes a CSV:
``global_index, split_index, primary_idx, primary_name``.

Slot order matches ``MSelector`` / ``train.py`` diagnostics: 0=acoustic, 1=language,
2=visual.

Usage (from ``My_creation/``)::

    python Qualitative_Evaluation/dump_dpr_primary_on_split.py \\
      --checkpoint ablation_study/runs/mosei_phase1_trial70_no_ib/checkpoints/infogate_mosei_best.pt \\
      --dataset mosei --split test --output /tmp/mosei70_noib_dpr_test.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import pickle
import random
import sys
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import DebertaV2Tokenizer

_QE_DIR = os.path.dirname(os.path.abspath(__file__))
_MY = os.path.dirname(_QE_DIR)
if _MY not in sys.path:
    sys.path.insert(0, _MY)

import global_configs  # noqa: E402
from deberta_infogate import InfoGate_DeBertaForSequenceClassification  # noqa: E402
from global_configs import DEVICE  # noqa: E402


def _apply_ckpt_arch(cli: SimpleNamespace, ckpt_path: str) -> None:
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


class InputFeatures:
    __slots__ = ("input_ids", "visual", "acoustic", "input_mask", "segment_ids", "label_id")

    def __init__(self, input_ids, visual, acoustic, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.visual = visual
        self.acoustic = acoustic
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


def prepare_deberta_input(tokens, visual, acoustic, tokenizer, max_seq_length, acoustic_dim, visual_dim):
    CLS = tokenizer.cls_token
    SEP = tokenizer.sep_token
    tokens = [CLS] + tokens + [SEP]

    az = np.zeros((1, acoustic_dim))
    acoustic = np.concatenate((az, acoustic, az))
    vz = np.zeros((1, visual_dim))
    visual = np.concatenate((vz, visual, vz))

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    segment_ids = [0] * len(input_ids)
    input_mask = [1] * len(input_ids)

    pad = max_seq_length - len(input_ids)
    acoustic = np.concatenate((acoustic, np.zeros((pad, acoustic_dim))))
    visual = np.concatenate((visual, np.zeros((pad, visual_dim))))
    input_ids += [0] * pad
    input_mask += [0] * pad
    segment_ids += [0] * pad

    return input_ids, visual, acoustic, input_mask, segment_ids


def convert_to_features(examples, max_seq_length, tokenizer, dataset: str):
    global_configs.set_dataset_config(dataset)
    adim = global_configs.ACOUSTIC_DIM
    vdim = global_configs.VISUAL_DIM
    feats = []
    for example in examples:
        (words, visual, acoustic), label_id, _segment = example

        tokens, inversions = [], []
        for idx, word in enumerate(words):
            toks = tokenizer.tokenize(word)
            tokens.extend(toks)
            inversions.extend([idx] * len(toks))

        aligned_v = np.array([visual[i] for i in inversions])
        aligned_a = np.array([acoustic[i] for i in inversions])

        if len(tokens) > max_seq_length - 2:
            tokens = tokens[: max_seq_length - 2]
            aligned_a = aligned_a[: max_seq_length - 2]
            aligned_v = aligned_v[: max_seq_length - 2]

        ids, vis, aud, mask, seg = prepare_deberta_input(
            tokens, aligned_v, aligned_a, tokenizer, max_seq_length, adim, vdim
        )
        feats.append(InputFeatures(ids, vis, aud, mask, seg, label_id))
    return feats


def build_tensor_dataset(feats):
    return TensorDataset(
        torch.tensor(np.array([f.input_ids for f in feats]), dtype=torch.long),
        torch.tensor(np.array([f.visual for f in feats]), dtype=torch.float),
        torch.tensor(np.array([f.acoustic for f in feats]), dtype=torch.float),
        torch.tensor(np.array([f.label_id for f in feats]), dtype=torch.float),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--dataset", type=str, choices=("mosi", "mosei"), required=True)
    ap.add_argument("--split", type=str, choices=("dev", "test"), default="test")
    ap.add_argument("--output", type=str, required=True)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=128)
    args = ap.parse_args()

    ckpt_path = os.path.abspath(args.checkpoint)
    if not os.path.isfile(ckpt_path):
        print(f"ERROR: checkpoint not found: {ckpt_path}", file=sys.stderr)
        return 1

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
    _apply_ckpt_arch(cli, ckpt_path)
    if not getattr(cli, "ablation", None):
        cli.ablation = "none"

    global_configs.set_dataset_config(cli.dataset)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model = InfoGate_DeBertaForSequenceClassification.from_pretrained(
        cli.model, multimodal_config=cli, num_labels=1
    )
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(DEVICE)
    model.eval()

    captured: list[torch.Tensor] = []

    def _hook(_mod, _inp, out):
        captured.append(out[-1].detach().cpu())

    h = model.dberta.infogate.mselector.register_forward_hook(_hook)

    ds_path = os.path.join(_MY, "datasets", f"{cli.dataset}.pkl")
    with open(ds_path, "rb") as fh:
        data = pickle.load(fh)
    key = "dev" if args.split == "dev" and "dev" in data else (
        "valid" if args.split == "dev" and "valid" in data else None
    )
    if args.split == "dev":
        if key is None:
            print("ERROR: no dev/valid split in pickle", file=sys.stderr)
            return 1
        raw = data[key]
    else:
        raw = data["test"]

    tok = DebertaV2Tokenizer.from_pretrained(cli.model)
    feats = convert_to_features(raw, cli.max_seq_length, tok, cli.dataset)
    ds = build_tensor_dataset(feats)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    names = {0: "acoustic", 1: "language", 2: "visual"}
    rows = []
    offset = 0
    with torch.no_grad():
        for batch in tqdm(dl, desc=f"DPR {args.split}"):
            batch = tuple(t.to(DEVICE) for t in batch)
            input_ids, visual, acoustic, _labels = batch
            visual = visual.squeeze(1)
            acoustic = acoustic.squeeze(1)
            captured.clear()
            _ = model(input_ids, visual, acoustic, labels=None, stage=2)
            if not captured:
                print("ERROR: mselector hook did not fire", file=sys.stderr)
                h.remove()
                return 1
            pidx = captured[-1]
            for j in range(pidx.size(0)):
                pi = int(pidx[j].item())
                rows.append(
                    {
                        "global_index": offset + j,
                        "split_index": offset + j,
                        "primary_idx": pi,
                        "primary_name": names.get(pi, str(pi)),
                    }
                )
            offset += pidx.size(0)
    h.remove()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["global_index", "split_index", "primary_idx", "primary_name"])
        w.writeheader()
        w.writerows(rows)

    n = len(rows)
    na = sum(1 for r in rows if r["primary_idx"] == 0)
    nl = sum(1 for r in rows if r["primary_idx"] == 1)
    nv = sum(1 for r in rows if r["primary_idx"] == 2)
    nnl = na + nv
    print(f"Wrote {n} rows to {args.output}")
    print(f"Counts: acoustic={na}  language={nl}  visual={nv}  non_language={nnl} ({100.0 * nnl / max(n,1):.2f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
