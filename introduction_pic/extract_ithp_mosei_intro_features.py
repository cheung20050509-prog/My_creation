#!/usr/bin/env python3
"""Extract ITHP MOSEI features for introduction 3D landscape (concat vs ITHP pooled fusion)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
ITHP_ROOT = REPO / "ITHP" / "ITHP"
DEFAULT_CKPT = ITHP_ROOT / "checkpoints" / "mosei_seed128" / "best.pt"
DEFAULT_OUTPUT = HERE / "outputs" / "ithp_mosei_intro_features.npz"


def _configure_ithp_argv(dataset: str = "mosei", seed: int = 128) -> None:
    sys.argv = [
        "train.py",
        "--dataset",
        dataset,
        "--seed",
        str(seed),
    ]


def _masked_mean(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 2:
        return x
    return x.mean(dim=1)


def _norm_av(visual: torch.Tensor, acoustic: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    visual_norm = (visual - visual.min()) / (visual.max() - visual.min())
    acoustic_norm = (acoustic - acoustic.min()) / (acoustic.max() - acoustic.min())
    return visual_norm, acoustic_norm


def _ithp_fused_and_concat(model, input_ids, visual_norm, acoustic_norm) -> tuple[torch.Tensor, torch.Tensor]:
    dberta = model.dberta
    x = dberta.model(input_ids)[0]
    b1, _, _, _, _, _ = dberta.ITHP(x, visual_norm, acoustic_norm)
    h_m = dberta.expand(b1)
    sequence_output = dberta.dropout(
        dberta.LayerNorm(dberta.beta_shift * h_m + x)
    )
    pooled = dberta.pooler(sequence_output)

    text_pool = _masked_mean(x)
    acoustic_pool = _masked_mean(acoustic_norm)
    visual_pool = _masked_mean(visual_norm)
    concat = torch.cat([text_pool, acoustic_pool, visual_pool], dim=-1)
    return concat, pooled


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument(
        "--shared-concat-npz",
        type=Path,
        default=HERE / "outputs" / "mosei_trial70_intro_features.npz",
        help="Reuse concat/labels from PRISM extract so both figures share the same baseline.",
    )
    args = ap.parse_args()

    ckpt_path = args.checkpoint.expanduser().resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Missing ITHP checkpoint: {ckpt_path}")

    os.chdir(ITHP_ROOT)
    if str(ITHP_ROOT) not in sys.path:
        sys.path.insert(0, str(ITHP_ROOT))

    _configure_ithp_argv()
    import global_configs as gc  # noqa: E402
    import train as train_mod  # noqa: E402

    try:
        blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        blob = torch.load(ckpt_path, map_location="cpu")

    saved = blob.get("args")
    if isinstance(saved, dict):
        for k, v in saved.items():
            setattr(train_mod.args, k, v)
    gc.set_dataset_config(train_mod.args.dataset)
    train_mod.set_random_seed(train_mod.args.seed)

    _, _, test_dl, n_opt = train_mod.set_up_data_loader()
    # ITHP train.py uses shuffle=True on test; force deterministic order for alignment
    # with PRISM fixed_experiment extract (shuffle=False, same MOSEI test split).
    test_dl = DataLoader(
        test_dl.dataset,
        batch_size=train_mod.args.test_batch_size,
        shuffle=False,
    )
    model, _, _ = train_mod.prep_for_training(n_opt)
    model.load_state_dict(blob["model_state_dict"], strict=True)
    model.to(gc.DEVICE)
    model.eval()

    ithp_rows: list[np.ndarray] = []
    label_rows: list[np.ndarray] = []
    pred_rows: list[np.ndarray] = []
    n_seen = 0

    with torch.no_grad():
        for batch in tqdm(test_dl, desc="ITHP MOSEI test"):
            batch = tuple(t.to(gc.DEVICE) for t in batch)
            input_ids, visual, acoustic, labels = batch
            visual = visual.squeeze(1)
            acoustic = acoustic.squeeze(1)
            visual_norm, acoustic_norm = _norm_av(visual, acoustic)

            _, pooled = _ithp_fused_and_concat(
                model, input_ids, visual_norm, acoustic_norm
            )
            logits, _, _, _, _, _ = model(input_ids, visual_norm, acoustic_norm)

            ithp_rows.append(pooled.detach().float().cpu().numpy())
            label_rows.append(labels.detach().float().cpu().numpy().reshape(-1))
            pred_rows.append(logits.detach().float().cpu().numpy().reshape(-1))
            n_seen += int(labels.shape[0])
            if args.max_samples is not None and n_seen >= args.max_samples:
                break

    ithp_np = np.concatenate(ithp_rows, axis=0)
    labels_np = np.concatenate(label_rows, axis=0)
    preds_np = np.concatenate(pred_rows, axis=0)
    if args.max_samples is not None:
        ithp_np = ithp_np[: args.max_samples]
        labels_np = labels_np[: args.max_samples]
        preds_np = preds_np[: args.max_samples]

    shared_path = args.shared_concat_npz.expanduser().resolve()
    if not shared_path.is_file():
        raise FileNotFoundError(
            f"Missing shared concat npz: {shared_path}\n"
            "Run extract_mosei_trial70_intro_features.py first."
        )
    shared = np.load(shared_path, allow_pickle=False)
    concat_np = shared["concat"]
    shared_labels = shared["labels"]
    if args.max_samples is not None:
        concat_np = concat_np[: args.max_samples]
        shared_labels = shared_labels[: args.max_samples]
    if not np.allclose(labels_np, shared_labels, atol=1e-5, rtol=1e-5):
        raise RuntimeError("ITHP test order does not match PRISM shared concat labels.")

    meta = {
        "checkpoint": str(ckpt_path),
        "dataset": "mosei",
        "model": "ITHP",
        "n_samples": int(labels_np.shape[0]),
        "concat_dim": int(concat_np.shape[1]),
        "concat_source": str(shared_path),
        "ithp_dim": int(ithp_np.shape[1]),
        "checkpoint_epoch": int(blob.get("epoch", -1)) if isinstance(blob, dict) else -1,
    }
    out = args.output.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        concat=concat_np,
        ithp=ithp_np,
        labels=labels_np,
        preds=preds_np,
        meta=json.dumps(meta, sort_keys=True),
    )
    print(json.dumps(meta, indent=2, sort_keys=True))
    print(f"saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
