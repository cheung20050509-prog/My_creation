"""SIMSv2 loader aligned with MMSA / KuDA shared data fields only.

Uses: text_bert, audio, vision, audio_lengths, vision_lengths, regression_labels.
Does not import MMSA or KuDA training frameworks, knowledge injection, or model code.

InfoGate expects text/audio/visual at a shared time axis [B, T, D]; valid A/V frames are
resampled onto MMSA text token positions (content tokens only).
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import TensorDataset

REQUIRED_KEYS = (
    "text_bert",
    "audio",
    "vision",
    "audio_lengths",
    "vision_lengths",
    "regression_labels",
)

# BERT special tokens in MMSA CH-SIMS packs
_CLS_ID = 101
_SEP_ID = 102
_PAD_ID = 0


def validate_mmsa_split(split: dict, split_name: str = "split") -> None:
    missing = [k for k in REQUIRED_KEYS if k not in split]
    if missing:
        raise KeyError(f"{split_name} missing MMSA keys: {missing}")


def infer_seq_lens_from_split(split: dict, max_seq_length: int) -> tuple[int, int, int]:
    """Return (T, V, A) bucket caps: text cap, vision frames, audio frames."""
    validate_mmsa_split(split)
    aud = np.asarray(split["audio"])
    vis = np.asarray(split["vision"])
    t_cap = int(max_seq_length)
    v_cap = int(vis.shape[1]) if aud.ndim >= 2 else 0
    a_cap = int(aud.shape[1]) if aud.ndim >= 2 else 0
    return t_cap, v_cap, a_cap


def _sanitize_modal(x: np.ndarray) -> np.ndarray:
    out = np.asarray(x, dtype=np.float32)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out


def _parse_text_bert(text_bert: np.ndarray, max_seq_length: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse MMSA text_bert (3, L) -> input_ids, input_mask, segment_ids."""
    tb = np.asarray(text_bert, dtype=np.float64)
    if tb.ndim != 2 or tb.shape[0] < 3:
        raise ValueError(f"text_bert must be (3, L), got {tb.shape}")

    row0 = tb[0].astype(np.int64)
    row1 = tb[1].astype(np.int64)
    row2 = tb[2].astype(np.int64)

    # MMSA order: [input_ids, input_mask, segment_ids]
    if row0[0] == _CLS_ID and np.any(row1 > 0):
        input_ids, input_mask, segment_ids = row0, row1, row2
    elif row1[0] == _CLS_ID and np.any(row0 > 0):
        # rare alternate storage
        input_ids, input_mask, segment_ids = row1, row0, row2
    else:
        input_ids, input_mask, segment_ids = row0, row1, row2

    L = min(int(max_seq_length), int(input_ids.shape[0]))
    input_ids = input_ids[:L].astype(np.int64)
    input_mask = input_mask[:L].astype(np.int64)
    segment_ids = segment_ids[:L].astype(np.int64)

    if input_mask.sum() == 0:
        input_mask = (input_ids != _PAD_ID).astype(np.int64)

    return input_ids, input_mask, segment_ids


def _content_token_indices(input_ids: np.ndarray, input_mask: np.ndarray) -> np.ndarray:
    """Positions used for A/V resampling (valid, non-special tokens)."""
    ids = np.asarray(input_ids).reshape(-1)
    mask = np.asarray(input_mask).reshape(-1)
    valid = mask > 0
    special = np.isin(ids, (_PAD_ID, _CLS_ID, _SEP_ID))
    idx = np.where(valid & ~special)[0]
    if idx.size == 0:
        idx = np.where(valid)[0]
    return idx.astype(np.int64)


def _resample_modal_to_text_axis(
    modal: np.ndarray,
    valid_len: int,
    input_ids: np.ndarray,
    input_mask: np.ndarray,
    out_len: int,
) -> np.ndarray:
    """Map valid_len frames of modal [valid_len, D] -> [out_len, D] on text positions."""
    modal = _sanitize_modal(modal)
    feat_dim = modal.shape[-1]
    out = np.zeros((out_len, feat_dim), dtype=np.float32)

    vlen = max(0, min(int(valid_len), int(modal.shape[0])))
    if vlen == 0:
        return out

    seq = modal[:vlen]
    targets = _content_token_indices(input_ids, input_mask)
    n_buckets = int(targets.size)
    if n_buckets == 0:
        return out

    # Uniform temporal bins -> one vector per content text position
    edges = np.linspace(0, vlen, n_buckets + 1).astype(np.int64)
    for b, pos in enumerate(targets):
        start, end = edges[b], edges[b + 1]
        if end <= start:
            end = min(start + 1, vlen)
        out[int(pos)] = seq[start:end].mean(axis=0)

    return out


def build_mmsa_sample(
    text_bert,
    audio,
    vision,
    audio_length,
    vision_length,
    label,
    max_seq_length: int,
) -> dict:
    input_ids, input_mask, segment_ids = _parse_text_bert(text_bert, max_seq_length)

    aud_len = int(audio_length)
    vis_len = int(vision_length)
    acoustic = _resample_modal_to_text_axis(
        np.asarray(audio), aud_len, input_ids, input_mask, max_seq_length
    )
    visual = _resample_modal_to_text_axis(
        np.asarray(vision), vis_len, input_ids, input_mask, max_seq_length
    )

    label_v = float(np.asarray(label).reshape(-1)[0])

    return {
        "input_ids": input_ids,
        "input_mask": input_mask,
        "segment_ids": segment_ids,
        "visual": visual,
        "acoustic": acoustic,
        "label_id": label_v,
    }


def build_mmsa_features(split: dict, max_seq_length: int) -> list[dict]:
    validate_mmsa_split(split)
    n = len(split["regression_labels"])
    feats = []
    for i in range(n):
        feats.append(
            build_mmsa_sample(
                split["text_bert"][i],
                split["audio"][i],
                split["vision"][i],
                split["audio_lengths"][i],
                split["vision_lengths"][i],
                split["regression_labels"][i],
                max_seq_length,
            )
        )
    return feats


def build_tensor_dataset(split: dict, max_seq_length: int) -> TensorDataset:
    """TensorDataset: input_ids, visual, acoustic, input_mask, segment_ids, label."""
    feats = build_mmsa_features(split, max_seq_length)
    return TensorDataset(
        torch.tensor(np.stack([f["input_ids"] for f in feats]), dtype=torch.long),
        torch.tensor(np.stack([f["visual"] for f in feats]), dtype=torch.float),
        torch.tensor(np.stack([f["acoustic"] for f in feats]), dtype=torch.float),
        torch.tensor(np.stack([f["input_mask"] for f in feats]), dtype=torch.long),
        torch.tensor(np.stack([f["segment_ids"] for f in feats]), dtype=torch.long),
        torch.tensor(np.array([f["label_id"] for f in feats]), dtype=torch.float),
    )


def uses_simsv2_mmsa(dataset: str, feature_mode: str) -> bool:
    return dataset == "simsv2" and feature_mode == "mmsa"


def unpack_batch(batch, use_mmsa: bool):
    """Unpack DataLoader batch -> ids, visual, acoustic, labels, mask, segment_ids."""
    if use_mmsa and len(batch) >= 6:
        input_ids, visual, acoustic, input_mask, segment_ids, label_ids = batch[:6]
        return input_ids, visual, acoustic, label_ids, input_mask, segment_ids
    input_ids, visual, acoustic, label_ids = batch[:4]
    return input_ids, visual, acoustic, label_ids, None, None


def format_seq_lens_report(split: dict, max_seq_length: int) -> str:
    t_cap, v_cap, a_cap = infer_seq_lens_from_split(split, max_seq_length)
    aud_lens = np.asarray(split["audio_lengths"], dtype=np.int64)
    vis_lens = np.asarray(split["vision_lengths"], dtype=np.int64)
    masks = []
    tb0 = np.asarray(split["text_bert"][0])
    for row in tb0:
        if row[0] == _CLS_ID:
            masks.append(np.asarray(row))
            break
    else:
        masks.append(np.asarray(tb0[1]))
    text_valid = int((masks[0] > 0).sum()) if masks else max_seq_length
    return (
        f"seq_lens=(T={t_cap}, V={v_cap}, A={a_cap}) "
        f"mean_valid=(text~{text_valid}, V={vis_lens.mean():.1f}, A={aud_lens.mean():.1f})"
    )
