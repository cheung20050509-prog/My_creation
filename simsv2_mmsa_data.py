"""SIMSv2 feature loading aligned with MMSA / KuDA pickle layout.

Uses offline ``text_bert`` (input_ids, attention_mask, token_type_ids),
``audio_lengths`` / ``vision_lengths``, per-modality ``seq_lens`` truncation,
then linearly resamples audio/vision along time to match the text token count
so InfoGate's shared time dimension T is satisfied.
"""

from __future__ import annotations

import re
from typing import Sequence, Tuple, Union

import numpy as np

# Presets: MMSA config_regression.json uses (T, A, V) = (39, 400, 55).
# KuDA opts.py SIMS default is seq_lens [39, 55, 400] meaning (T, V, A), i.e. same caps.
SEQ_LENS_MMSA = (39, 400, 55)  # (T, A, V)
SEQ_LENS_KUDA_SIMS = (39, 55, 400)  # (T, V, A) -> maps to same (T, A, V)


def clean_modal(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x[np.isneginf(x)] = 0.0
    return x


def resolve_seq_lens(
    order: str,
    explicit: Union[str, None],
) -> Tuple[int, int, int]:
    """Return (T_cap, A_cap, V_cap) for truncation."""
    order = (order or "mmsa").lower()
    if explicit and str(explicit).strip():
        parts = [int(p.strip()) for p in re.split(r"[,\s]+", str(explicit).strip()) if p.strip()]
        if len(parts) != 3:
            raise ValueError("--simsv2_seq_lens expects three integers, e.g. 39,400,55")
        if order == "kuda":
            t_cap, v_cap, a_cap = parts
            return int(t_cap), int(a_cap), int(v_cap)
        t_cap, a_cap, v_cap = parts
        return int(t_cap), int(a_cap), int(v_cap)
    if order == "kuda":
        t_cap, v_cap, a_cap = SEQ_LENS_KUDA_SIMS
        return int(t_cap), int(a_cap), int(v_cap)
    return tuple(int(x) for x in SEQ_LENS_MMSA)  # type: ignore[return-value]


def truncate_mmsa_time_prefix(modal: np.ndarray, max_len: int) -> np.ndarray:
    """Keep the non-padding prefix of ``modal`` (T, D), then at most ``max_len`` frames.

    Padding rows are all-zero (MMSA convention). If no zero row is found, the full
    length ``T`` is treated as valid. Matches the intent of MMSA/KuDA truncation:
    clip to the first ``max_len`` valid timesteps before the all-zero tail.
    """
    modal = clean_modal(np.asarray(modal, dtype=np.float32))
    if modal.ndim != 2:
        raise ValueError(f"Expected modal (T, D), got shape {modal.shape}")
    t_total, _d = modal.shape
    if t_total == 0 or max_len <= 0:
        return modal[:0]

    zero_row = np.zeros(modal.shape[1], dtype=modal.dtype)
    end = t_total
    for i in range(t_total):
        if np.array_equal(modal[i], zero_row):
            end = i
            break
    if end <= 0:
        # Degenerate: first row already padding — fall back to raw prefix cap.
        return modal[: min(t_total, max_len)]

    return modal[: min(end, max_len)]


def resample_time_axis(x: np.ndarray, src_len: int, tgt_len: int) -> np.ndarray:
    """Linearly resample ``x[:src_len]`` along time to length ``tgt_len``."""
    if tgt_len <= 0:
        raise ValueError("tgt_len must be positive")
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"Expected (T, D), got {x.shape}")
    src_len = int(max(0, min(src_len, x.shape[0])))
    if src_len <= 0:
        d = x.shape[1]
        return np.zeros((tgt_len, d), dtype=np.float32)
    if src_len == tgt_len:
        return np.array(x[:src_len], copy=True)
    old_idx = np.linspace(0.0, float(src_len - 1), num=src_len)
    new_idx = np.linspace(0.0, float(src_len - 1), num=tgt_len)
    out = np.empty((tgt_len, x.shape[1]), dtype=np.float32)
    for d in range(x.shape[1]):
        out[:, d] = np.interp(new_idx, old_idx, x[:src_len, d].astype(np.float64)).astype(np.float32)
    return out


def _inner_len_from_text_bert(
    text_bert: np.ndarray,
    t_cap: int,
    max_inner: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Truncate ``text_bert`` to ``t_cap`` positions; length = leading mask==1 run (MMSA-style)."""
    tb = np.asarray(text_bert, dtype=np.float64)
    if tb.shape[0] != 3:
        raise ValueError(f"text_bert must be (3, L), got {tb.shape}")
    L = min(tb.shape[1], t_cap)
    ids = np.asarray(tb[0, :L], dtype=np.int64)
    mask = np.asarray(tb[1, :L], dtype=np.float32)
    seg = np.asarray(tb[2, :L], dtype=np.int64)
    n_lead = 0
    for j in range(L):
        if mask[j] > 0.5:
            n_lead += 1
        else:
            break
    if n_lead <= 0:
        n_lead = min(L, max_inner)
    n_valid = min(n_lead, max_inner)
    ids = ids[:n_valid]
    mask = mask[:n_valid]
    seg = seg[:n_valid]
    return ids, mask, seg, n_valid


def build_simsv2_mmsa_feature_vectors(
    *,
    text_bert: np.ndarray,
    vision: np.ndarray,
    audio: np.ndarray,
    audio_length: int,
    vision_length: int,
    label_id: float,
    seq_lens: Tuple[int, int, int],
    max_seq_length: int,
    acoustic_dim: int,
    visual_dim: int,
    cls_id: int,
    sep_id: int,
    pad_id: int,
) -> Tuple[list, np.ndarray, np.ndarray, list, list, float]:
    """Build one sample: same contract as ``prepare_deberta_input`` output lists/arrays."""
    t_cap, a_cap, v_cap = seq_lens
    max_inner = max(1, max_seq_length - 2)

    ids_inner, _mask_inner, _seg_inner, n_text = _inner_len_from_text_bert(
        text_bert, t_cap, max_inner
    )

    a_full = truncate_mmsa_time_prefix(audio, a_cap)
    v_full = truncate_mmsa_time_prefix(vision, v_cap)
    a_eff = int(max(0, min(int(audio_length), a_full.shape[0])))
    v_eff = int(max(0, min(int(vision_length), v_full.shape[0])))
    if a_eff <= 0 and a_full.shape[0] > 0:
        a_eff = min(a_full.shape[0], a_cap)
    if v_eff <= 0 and v_full.shape[0] > 0:
        v_eff = min(v_full.shape[0], v_cap)

    aud_rs = resample_time_axis(a_full, a_eff, n_text)
    vis_rs = resample_time_axis(v_full, v_eff, n_text)

    if aud_rs.shape[1] != acoustic_dim or vis_rs.shape[1] != visual_dim:
        raise ValueError(
            f"Dim mismatch: audio {aud_rs.shape[1]} vs {acoustic_dim}, "
            f"vision {vis_rs.shape[1]} vs {visual_dim}"
        )

    az = np.zeros((1, acoustic_dim), dtype=np.float32)
    vz = np.zeros((1, visual_dim), dtype=np.float32)
    acoustic = np.concatenate((az, aud_rs, az), axis=0)
    visual = np.concatenate((vz, vis_rs, vz), axis=0)

    input_ids = [int(cls_id)] + [int(x) for x in ids_inner] + [int(sep_id)]
    segment_ids = [0] * len(input_ids)
    input_mask = [1] * len(input_ids)

    pad = max_seq_length - len(input_ids)
    if pad < 0:
        raise ValueError("max_seq_length too small for CLS + tokens + SEP")
    acoustic = np.concatenate((acoustic, np.zeros((pad, acoustic_dim), dtype=np.float32)), axis=0)
    visual = np.concatenate((visual, np.zeros((pad, visual_dim), dtype=np.float32)), axis=0)
    input_ids = input_ids + [int(pad_id)] * pad
    input_mask = input_mask + [0] * pad
    segment_ids = segment_ids + [0] * pad

    return input_ids, visual, acoustic, input_mask, segment_ids, float(label_id)


def simsv2_mmsa_loader_banner(seq_lens: Sequence[int], order: str) -> str:
    t, a, v = seq_lens
    return (
        f"SIMSv2 loader=mmsa seq_lens=({t},{a},{v}) (T,A,V) order={order} "
        f"resample=on"
    )
