#!/usr/bin/env python3
"""Extract MOSEI trial70 features for the introduction 3D landscape figure.

This script reuses the frozen fixed_experiment MOSEI trial70 training stack and
caches two test-set representations:
  - concat: masked-pooled DeBERTa text + acoustic + visual (MOSEI test, real forward)
  - prism:  PRISM/InfoGate fused pooled representation h_p (same samples)
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
MY_CREATION = HERE.parent
FIXED = MY_CREATION / "fixed_experiment"
DEFAULT_RUN = FIXED / "runs" / "mosei_phase1_trial70"
DEFAULT_CKPT = DEFAULT_RUN / "checkpoints" / "infogate_mosei_best.pt"
DEFAULT_OUTPUT = HERE / "outputs" / "mosei_trial70_intro_features.npz"


def _prepare_train_module(checkpoint_dir: Path):
    if str(FIXED) not in sys.path:
        sys.path.insert(0, str(FIXED))

    from mosei_phase1_trial70_hparams import build_train_argv

    sys.argv = ["train.py", *build_train_argv(checkpoint_dir=str(checkpoint_dir))]
    os.chdir(MY_CREATION)
    if "train" in sys.modules:
        return importlib.reload(sys.modules["train"])
    return importlib.import_module("train")


def _load_model(train_mod, ckpt_path: Path):
    train_mod.set_seed(train_mod.args.seed)
    _, _, test_dl, n_opt = train_mod.setup_data()
    model, _, _ = train_mod.build_model(n_opt)

    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"Missing checkpoint: {ckpt_path}\n"
            "Regenerate it with:\n"
            f"  cd {MY_CREATION} && bash fixed_experiment/run_mosei_phase1_trial70.sh"
        )

    try:
        blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        blob = torch.load(ckpt_path, map_location="cpu")
    state = blob.get("model_state_dict", blob)
    model.load_state_dict(state, strict=True)
    model.to(train_mod.DEVICE)
    model.eval()
    return model, test_dl, blob


def _masked_mean(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if x.dim() == 2:
        return x
    if mask is None or mask.dim() != 2 or mask.shape[:2] != x.shape[:2]:
        return x.mean(dim=1)
    m = mask.to(dtype=x.dtype, device=x.device).unsqueeze(-1)
    return (x * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)


def collect_features(train_mod, model, test_dl, *, stage: int, max_samples: int | None):
    concat_rows: list[np.ndarray] = []
    prism_rows: list[np.ndarray] = []
    label_rows: list[np.ndarray] = []
    pred_rows: list[np.ndarray] = []
    n_seen = 0

    with torch.no_grad():
        for batch in test_dl:
            batch = tuple(t.to(train_mod.DEVICE) for t in batch)
            input_ids, visual, acoustic, labels = batch
            if visual.dim() == 4:
                visual = visual.squeeze(1)
            if acoustic.dim() == 4:
                acoustic = acoustic.squeeze(1)

            pad_id = model.dberta.config.pad_token_id
            pad_id = 0 if pad_id is None else int(pad_id)
            attn = input_ids.ne(pad_id).long()

            text_features = model.dberta.model(
                input_ids=input_ids,
                attention_mask=attn,
            )[0]
            logits, _, _, h_p = model(
                input_ids,
                visual,
                acoustic,
                labels=labels,
                stage=stage,
            )

            text_pool = _masked_mean(text_features, attn)
            acoustic_pool = _masked_mean(acoustic, attn)
            visual_pool = _masked_mean(visual, attn)
            concat = torch.cat([text_pool, acoustic_pool, visual_pool], dim=-1)

            concat_rows.append(concat.detach().float().cpu().numpy())
            prism_rows.append(h_p.detach().float().cpu().numpy())
            label_rows.append(labels.detach().float().cpu().numpy().reshape(-1))
            pred_rows.append(logits.detach().float().cpu().numpy().reshape(-1))

            n_seen += int(labels.shape[0])
            if max_samples is not None and n_seen >= max_samples:
                break

    concat_np = np.concatenate(concat_rows, axis=0)
    prism_np = np.concatenate(prism_rows, axis=0)
    labels_np = np.concatenate(label_rows, axis=0)
    preds_np = np.concatenate(pred_rows, axis=0)
    if max_samples is not None:
        concat_np = concat_np[:max_samples]
        prism_np = prism_np[:max_samples]
        labels_np = labels_np[:max_samples]
        preds_np = preds_np[:max_samples]
    return concat_np, prism_np, labels_np, preds_np


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--stage", type=int, default=2, choices=(1, 2))
    ap.add_argument("--max-samples", type=int, default=None)
    args = ap.parse_args()

    ckpt_path = args.checkpoint.expanduser().resolve()
    train_mod = _prepare_train_module(ckpt_path.parent)
    model, test_dl, blob = _load_model(train_mod, ckpt_path)
    concat, prism, labels, preds = collect_features(
        train_mod,
        model,
        test_dl,
        stage=int(args.stage),
        max_samples=args.max_samples,
    )

    out = args.output.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "checkpoint": str(ckpt_path),
        "dataset": "mosei",
        "trial": "phase1_trial70",
        "stage": int(args.stage),
        "n_samples": int(labels.shape[0]),
        "concat_dim": int(concat.shape[1]),
        "prism_dim": int(prism.shape[1]),
        "checkpoint_epoch": int(blob.get("epoch", -1)) if isinstance(blob, dict) else -1,
        "test_results": blob.get("test_results", {}) if isinstance(blob, dict) else {},
    }
    np.savez_compressed(
        out,
        concat=concat,
        prism=prism,
        labels=labels,
        preds=preds,
        meta=json.dumps(meta, sort_keys=True),
    )
    print(json.dumps(meta, indent=2, sort_keys=True))
    print(f"saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
