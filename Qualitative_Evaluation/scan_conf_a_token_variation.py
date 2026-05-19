#!/usr/bin/env python3
"""Scan test split for samples where token-level ``ib_conf_a`` has high variation.

For each row, ``conf_a`` is ``[B,T,D]`` from ``IBEncoder``; we take ``mean_D`` per token
(``per_token_ib_conf_mean_over_bottleneck``), then on **valid** positions only compute:

- ``std_tok``: std dev across time
- ``rng_tok``: max - min across time

Ranks rows by ``std_tok`` (descending).  Use to pick qualitative cases where ``conf_a``
is not flat.

Optional ``--acoustic_primary_only``: keep only samples where DPR primary is acoustic
(``primary_idx == 0``; same slot order as ``dump_dpr_primary_on_split``).

Usage (from ``My_creation/``)::

    conda run -n ITHP5090 python Qualitative_Evaluation/scan_conf_a_token_variation.py \\
      --backbone deberta --dataset mosei --split test \\
      --checkpoint logs/optuna/4090D_restart/phase1/checkpoints/optuna_mosei_phase1/trial_70/infogate_mosei_best.pt \\
      --top 20

    conda run -n ITHP5090 python Qualitative_Evaluation/scan_conf_a_token_variation.py \\
      --backbone albert --dataset mustard --split test \\
      --checkpoint logs/optuna/4090D_restart/classification/optuna_classify_mustard_albert_hcf_b3_20260427_235533/checkpoints/optuna_mustard/trial_113/infogate_mustard_best.pt \\
      --top 20

    Acoustic-primary only::

    conda run -n ITHP5090 python Qualitative_Evaluation/scan_conf_a_token_variation.py \\
      --backbone deberta --dataset mosei --split test --acoustic_primary_only \\
      --checkpoint logs/optuna/4090D_restart/phase1/checkpoints/optuna_mosei_phase1/trial_70/infogate_mosei_best.pt \\
      --top 20
"""

from __future__ import annotations

import argparse
import heapq
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
    per_sample_ib_conf_mean,
    per_token_ib_conf_mean_over_bottleneck,
)


def _apply_ckpt_arch_deberta(cli: SimpleNamespace, ckpt_path: str) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    saved = ckpt.get("args")
    if saved is None:
        return
    keys = (
        "model", "dataset", "max_seq_length", "unified_dim", "ib_hidden_dim", "bottleneck_dim",
        "num_heads", "num_infogate_layers", "dropout_prob", "beta_ib", "alpha_ib", "mse_weight",
        "selector_target_temp", "selector_balance_weight", "selector_rib_weight",
        "align_mix_floor",
        "gumbel_tau_start", "gumbel_tau_end", "ablation",
    )
    for k in keys:
        if hasattr(saved, k):
            setattr(cli, k, getattr(saved, k))
    for flag, attr in (("disable_l_lib", "use_l_lib"), ("disable_l_rib", "use_l_rib")):
        if hasattr(saved, flag):
            setattr(cli, attr, not bool(getattr(saved, flag)))


def _apply_ckpt_arch_albert(cli: SimpleNamespace, ckpt_path: str) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    saved = ckpt.get("args")
    if saved is None:
        return
    keys = (
        "model", "dataset", "max_seq_length", "unified_dim", "ib_hidden_dim", "bottleneck_dim",
        "num_heads", "num_infogate_layers", "dropout_prob", "beta_ib", "alpha_ib", "mse_weight",
        "selector_target_temp", "selector_balance_weight", "selector_rib_weight",
        "align_mix_floor",
        "gumbel_tau_start", "gumbel_tau_end", "task_type",
    )
    for k in keys:
        if hasattr(saved, k):
            setattr(cli, k, getattr(saved, k))
    for flag, attr in (("disable_l_lib", "use_l_lib"), ("disable_l_rib", "use_l_rib")):
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


def _masked_std_rng(per_tok: torch.Tensor, attn: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """per_tok, attn: [B, T]. Return std and max-min over valid positions per row."""
    b, t = per_tok.shape
    stds = []
    rngs = []
    for j in range(b):
        m = attn[j] > 0.5
        vals = per_tok[j, m]
        if vals.numel() < 2:
            stds.append(torch.tensor(0.0, device=per_tok.device))
            rngs.append(torch.tensor(0.0, device=per_tok.device))
        else:
            stds.append(vals.std(unbiased=False))
            rngs.append(vals.max() - vals.min())
    return torch.stack(stds), torch.stack(rngs)


def _run_deberta(cli, ckpt_path: str, split: str, batch_size: int, top: int, acoustic_primary_only: bool) -> None:
    import importlib.util
    from deberta_infogate import InfoGate_DeBertaForSequenceClassification  # noqa: E402
    from transformers import DebertaV2Tokenizer  # noqa: E402

    path = os.path.join(_QE_DIR, "dump_dpr_primary_on_split.py")
    spec = importlib.util.spec_from_file_location("_dpr_os_split", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

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
    raw = data["test"] if split == "test" else (data["dev"] if "dev" in data else data["valid"])
    tok = DebertaV2Tokenizer.from_pretrained(cli.model)
    feats = mod.convert_to_features(raw, cli.max_seq_length, tok, cli.dataset)
    ds = mod.build_tensor_dataset(feats)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)

    pad_id = model.config.pad_token_id if model.config.pad_token_id is not None else 0
    heap: list[tuple[float, int, float, float]] = []
    offset = 0
    n_a_primary = 0
    try:
        with torch.no_grad():
            for batch in dl:
                batch = tuple(t.to(DEVICE) for t in batch)
                input_ids, visual, acoustic, _labels = batch
                visual = visual.squeeze(1)
                acoustic = acoustic.squeeze(1)
                attn = input_ids.ne(pad_id).float()
                ib_store.clear()
                captured.clear()
                model(input_ids, visual, acoustic, labels=None, stage=2)
                if "a" not in ib_store:
                    print("ERROR: no ib_store['a']", file=sys.stderr)
                    return
                if not captured:
                    print("ERROR: mselector hook did not fire", file=sys.stderr)
                    return
                wcpu, pidx = captured[-1]
                if wcpu is None:
                    print("ERROR: could not parse DPR weights from mselector output", file=sys.stderr)
                    return
                per_tok = per_token_ib_conf_mean_over_bottleneck(ib_store["a"], attn)
                mean_a = per_sample_ib_conf_mean(ib_store["a"], attn)
                std_b, rng_b = _masked_std_rng(per_tok, attn)
                bsz = input_ids.size(0)
                for j in range(bsz):
                    pi = int(pidx[j].item())
                    if acoustic_primary_only and pi != 0:
                        continue
                    if pi == 0:
                        n_a_primary += 1
                    gi = offset + j
                    st = float(std_b[j].item())
                    rg = float(rng_b[j].item())
                    ma = float(mean_a[j].item())
                    if len(heap) < top:
                        heapq.heappush(heap, (st, gi, rg, ma))
                    else:
                        if st > heap[0][0]:
                            heapq.heapreplace(heap, (st, gi, rg, ma))
                offset += bsz
    finally:
        h.remove()
        cleanup_ib()

    rows = sorted(heap, key=lambda x: -x[0])
    filt = " primary_idx==0 (acoustic) only" if acoustic_primary_only else ""
    print(
        f"# {cli.dataset} {split}: top-{len(rows)} by token-level std(mean_D conf_a){filt}; "
        f"acoustic-primary count in split={n_a_primary}"
    )
    print("global_index\tstd_tok\tmax-min\tmean_conf_a(sample)")
    for st, gi, rg, ma in rows:
        print(f"{gi}\t{st:.6f}\t{rg:.6f}\t{ma:.6f}")


def _run_albert(cli, ckpt_path: str, split: str, batch_size: int, top: int, acoustic_primary_only: bool) -> None:
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
        return

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
    pad_id = model.config.pad_token_id if model.config.pad_token_id is not None else 0
    heap: list[tuple[float, int, float, float]] = []
    offset = 0
    n_a_primary = 0
    try:
        with torch.no_grad():
            for batch in dl:
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
                if not captured:
                    print("ERROR: mselector did not emit routing (hook/wrapper)", file=sys.stderr)
                    return
                wcpu, pidx = captured[-1]
                if wcpu is None:
                    print("ERROR: could not parse DPR weights from mselector output", file=sys.stderr)
                    return
                per_tok = per_token_ib_conf_mean_over_bottleneck(ib_store["a"], attn)
                mean_a = per_sample_ib_conf_mean(ib_store["a"], attn)
                std_b, rng_b = _masked_std_rng(per_tok, attn)
                bsz = input_ids.size(0)
                for j in range(bsz):
                    pi = int(pidx[j].item())
                    if acoustic_primary_only and pi != 0:
                        continue
                    if pi == 0:
                        n_a_primary += 1
                    gi = offset + j
                    st = float(std_b[j].item())
                    rg = float(rng_b[j].item())
                    ma = float(mean_a[j].item())
                    if len(heap) < top:
                        heapq.heappush(heap, (st, gi, rg, ma))
                    else:
                        if st > heap[0][0]:
                            heapq.heapreplace(heap, (st, gi, rg, ma))
                offset += bsz
    finally:
        if hook_handle is not None:
            hook_handle.remove()
        if _orig_forward4 is not None:
            msel.forward4 = _orig_forward4
        cleanup_ib()

    rows = sorted(heap, key=lambda x: -x[0])
    filt = " primary_idx==0 (acoustic) only" if acoustic_primary_only else ""
    print(
        f"# {cli.dataset} {split}: top-{len(rows)} by token-level std(mean_D conf_a){filt}; "
        f"acoustic-primary count in split={n_a_primary}"
    )
    print("global_index\tstd_tok\tmax-min\tmean_conf_a(sample)")
    for st, gi, rg, ma in rows:
        print(f"{gi}\t{st:.6f}\t{rg:.6f}\t{ma:.6f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backbone", choices=("deberta", "albert"), required=True)
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--dataset", type=str, required=True)
    ap.add_argument("--split", choices=("dev", "test"), default="test")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--seed", type=int, default=128)
    ap.add_argument(
        "--acoustic_primary_only",
        action="store_true",
        help="Only rank samples with DPR primary acoustic (primary_idx==0).",
    )
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
            print("ERROR: deberta expects mosi|mosei", file=sys.stderr)
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
            align_mix_floor=0.3,
            gumbel_tau_start=1.0,
            gumbel_tau_end=0.5,
            ablation="none",
            use_l_lib=True,
            use_l_rib=True,
        )
        _apply_ckpt_arch_deberta(cli, ckpt_path)
        if not getattr(cli, "ablation", None):
            cli.ablation = "none"
        _run_deberta(
            cli, ckpt_path, args.split, args.batch_size, args.top, args.acoustic_primary_only
        )
        return 0

    if args.dataset not in ("mustard", "ur_funny"):
        print("ERROR: albert expects mustard|ur_funny", file=sys.stderr)
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
        align_mix_floor=0.3,
        gumbel_tau_start=1.0,
        gumbel_tau_end=0.5,
        task_type="binary",
        use_l_lib=True,
        use_l_rib=True,
    )
    _apply_ckpt_arch_albert(cli, ckpt_path)
    if not getattr(cli, "task_type", None):
        cli.task_type = "binary"
    _run_albert(
        cli, ckpt_path, args.split, args.batch_size, args.top, args.acoustic_primary_only
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
