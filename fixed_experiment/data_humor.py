"""HKT-style (context + punchline) pkl loader for binary classification tasks.

Used by `train_classify.py` / `test_classify.py` for UR-FUNNY (humor) and
MUSTARD (sarcasm). Mirrors HKT's `convert_humor_to_features` logic and emits
a 5-tensor TensorDataset compatible with InfoGate's extended forward signature
(input_ids, visual, acoustic, hcf, label).

Sample layout (verified live):
    ((p_words, p_vis, p_aco, p_hcf),     # punchline segment (single sentence str)
     (c_words, c_vis, c_aco, c_hcf),     # context segment (list of sentence str)
     hid,
     label)                              # int 0/1
Visual/acoustic/HCF rows are per-word.

HKT feature slicing (enabled by default; pass ``slice_hkt=False`` to disable):
    acoustic[:, 0:60]      (81 -> 60)
    visual  [:, 55:91]     (91 -> 36)
    hcf stays 4-dim
Matches https://github.com/matalvepu/HKT/blob/main/global_config.py .
"""

import os
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


SPIECE_MARKER = "\u2581"  # SentencePiece word-start marker (DeBERTa-v3 / ALBERT)


PKL_FILENAMES = {
    "ur_funny": ["ur_funny.pkl", "urfunnyv2.pkl"],
    "mustard":  ["mustard.pkl"],
}


# HKT-aligned feature slicing indices.
# Reproduces `visual_features_list=list(range(55, 91))` and
# `acoustic_features_list=list(range(0, 60))` from HKT's global_config.py.
HKT_ACOUSTIC_SLICE = slice(0, 60)
HKT_VISUAL_SLICE = slice(55, 91)

HKT_ACOUSTIC_DIM = 60
HKT_VISUAL_DIM = 36
HKT_HCF_DIM = 4


def get_inversion(tokens, marker=SPIECE_MARKER):
    """For each subword token, return the index of the original word it belongs to."""
    inv_idx = -1
    inversions = []
    for tok in tokens:
        if marker in tok:
            inv_idx += 1
        inversions.append(inv_idx)
    return inversions


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """HKT-style: keep the punchline (b) intact; drop context (a) from the front first.

    Returns the number of head pops applied to `tokens_a` so callers can slice
    `inversions_a` correspondingly.
    """
    pop_count = 0
    while True:
        total = len(tokens_a) + len(tokens_b)
        if total <= max_length:
            break
        if not tokens_a:
            tokens_b.pop()
        else:
            tokens_a.pop(0)
            pop_count += 1
    return pop_count


def _gather_features(inversions, feat):
    """Pick per-token features from per-word `feat` using inversion ids; clamp out-of-range."""
    if not inversions:
        return np.zeros((0, feat.shape[1]), dtype=np.float32)
    n_rows = feat.shape[0]
    # Clamp to be defensive in case tokenization produced more "words" than features.
    safe = [min(max(int(i), 0), n_rows - 1) if n_rows > 0 else 0 for i in inversions]
    if n_rows == 0:
        return np.zeros((len(inversions), feat.shape[1]), dtype=np.float32)
    return feat[safe]


def _to_text(words_field, append_period=False):
    """Normalize a HKT `words` field (str or list[str]) into a single text string."""
    if isinstance(words_field, list):
        text = ". ".join(s for s in words_field if isinstance(s, str))
    elif isinstance(words_field, str):
        text = words_field
    else:
        text = str(words_field)
    if append_period and text and not text.rstrip().endswith("."):
        text = text.rstrip() + "."
    return text


def _ensure_2d(arr, default_dim):
    """Coerce possibly-empty HKT feature rows into a (0, default_dim) float32 array."""
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1 and arr.size == 0:
        return np.zeros((0, default_dim), dtype=np.float32)
    return arr


def convert_to_features(samples, tokenizer, max_seq_length, acoustic_dim, visual_dim,
                        pad_token_id, hcf_dim=0, slice_hkt=True):
    """Vectorize HKT samples into InfoGate-friendly tensors.

    Args:
        slice_hkt: when True, apply HKT's feature slicing
            (acoustic[:, 0:60], visual[:, 55:91]); ``acoustic_dim`` / ``visual_dim``
            should be the **post-slice** dims (60 / 36).
        hcf_dim: when > 0, also emit HCF aligned per token (dim ``hcf_dim``).

    Returns a list of dicts with keys: input_ids, input_mask, visual, acoustic, hcf, label.
    """
    out = []
    cls = tokenizer.cls_token
    sep = tokenizer.sep_token

    # Raw pkl column dims (pre-slice) used inside the pipeline. When slice_hkt
    # is on we build features against the full HKT column widths and only
    # project to the narrower dim at the end; this is what HKT does.
    raw_acoustic_dim = 81 if slice_hkt else acoustic_dim
    raw_visual_dim = 91 if slice_hkt else visual_dim
    use_hcf = hcf_dim > 0

    for ex in samples:
        # HKT order: first tuple = punchline; second tuple = context.
        (p_words, p_vis, p_aco, p_hcf), (c_words, c_vis, c_aco, c_hcf), _hid, label = ex

        text_a = _to_text(c_words, append_period=False)               # context
        text_b = _to_text(p_words, append_period=True)                # punchline + "."

        tokens_a = tokenizer.tokenize(text_a) if text_a else []
        tokens_b = tokenizer.tokenize(text_b) if text_b else []

        inversions_a = get_inversion(tokens_a)
        inversions_b = get_inversion(tokens_b)

        pop = _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        inversions_a = inversions_a[pop:][: len(tokens_a)]
        inversions_b = inversions_b[: len(tokens_b)]

        c_vis_arr = _ensure_2d(c_vis, raw_visual_dim)
        c_aco_arr = _ensure_2d(c_aco, raw_acoustic_dim)
        p_vis_arr = _ensure_2d(p_vis, raw_visual_dim)
        p_aco_arr = _ensure_2d(p_aco, raw_acoustic_dim)

        if use_hcf:
            c_hcf_arr = _ensure_2d(c_hcf, hcf_dim)
            p_hcf_arr = _ensure_2d(p_hcf, hcf_dim)

        visual_a = (_gather_features(inversions_a, c_vis_arr)
                    if tokens_a else np.zeros((0, raw_visual_dim), dtype=np.float32))
        acoustic_a = (_gather_features(inversions_a, c_aco_arr)
                      if tokens_a else np.zeros((0, raw_acoustic_dim), dtype=np.float32))
        visual_b = (_gather_features(inversions_b, p_vis_arr)
                    if tokens_b else np.zeros((0, raw_visual_dim), dtype=np.float32))
        acoustic_b = (_gather_features(inversions_b, p_aco_arr)
                      if tokens_b else np.zeros((0, raw_acoustic_dim), dtype=np.float32))

        if use_hcf:
            hcf_a = (_gather_features(inversions_a, c_hcf_arr)
                     if tokens_a else np.zeros((0, hcf_dim), dtype=np.float32))
            hcf_b = (_gather_features(inversions_b, p_hcf_arr)
                     if tokens_b else np.zeros((0, hcf_dim), dtype=np.float32))

        tokens = [cls] + tokens_a + [sep] + tokens_b + [sep]

        zv = np.zeros((1, raw_visual_dim), dtype=np.float32)
        za = np.zeros((1, raw_acoustic_dim), dtype=np.float32)
        if not tokens_a:
            visual = np.concatenate((zv, zv, visual_b, zv), axis=0)
            acoustic = np.concatenate((za, za, acoustic_b, za), axis=0)
        else:
            visual = np.concatenate((zv, visual_a, zv, visual_b, zv), axis=0)
            acoustic = np.concatenate((za, acoustic_a, za, acoustic_b, za), axis=0)

        if use_hcf:
            zh = np.zeros((1, hcf_dim), dtype=np.float32)
            if not tokens_a:
                hcf = np.concatenate((zh, zh, hcf_b, zh), axis=0)
            else:
                hcf = np.concatenate((zh, hcf_a, zh, hcf_b, zh), axis=0)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        # If for any reason features and tokens disagree in length, harmonize to len(tokens)
        if visual.shape[0] < len(input_ids):
            pad_v = np.zeros((len(input_ids) - visual.shape[0], raw_visual_dim), dtype=np.float32)
            visual = np.concatenate((visual, pad_v), axis=0)
        else:
            visual = visual[: len(input_ids)]
        if acoustic.shape[0] < len(input_ids):
            pad_a = np.zeros((len(input_ids) - acoustic.shape[0], raw_acoustic_dim), dtype=np.float32)
            acoustic = np.concatenate((acoustic, pad_a), axis=0)
        else:
            acoustic = acoustic[: len(input_ids)]
        if use_hcf:
            if hcf.shape[0] < len(input_ids):
                pad_h = np.zeros((len(input_ids) - hcf.shape[0], hcf_dim), dtype=np.float32)
                hcf = np.concatenate((hcf, pad_h), axis=0)
            else:
                hcf = hcf[: len(input_ids)]

        pad = max_seq_length - len(input_ids)
        if pad > 0:
            input_ids = input_ids + [pad_token_id] * pad
            input_mask = input_mask + [0] * pad
            visual = np.concatenate((visual, np.zeros((pad, raw_visual_dim), dtype=np.float32)), axis=0)
            acoustic = np.concatenate((acoustic, np.zeros((pad, raw_acoustic_dim), dtype=np.float32)), axis=0)
            if use_hcf:
                hcf = np.concatenate((hcf, np.zeros((pad, hcf_dim), dtype=np.float32)), axis=0)
        else:
            input_ids = input_ids[:max_seq_length]
            input_mask = input_mask[:max_seq_length]
            visual = visual[:max_seq_length]
            acoustic = acoustic[:max_seq_length]
            if use_hcf:
                hcf = hcf[:max_seq_length]

        # HKT-aligned slicing: project full pkl columns down to the modelled subset.
        if slice_hkt:
            acoustic = acoustic[:, HKT_ACOUSTIC_SLICE]
            visual = visual[:, HKT_VISUAL_SLICE]

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert visual.shape == (max_seq_length, visual_dim), \
            f"visual shape {visual.shape} != ({max_seq_length}, {visual_dim})"
        assert acoustic.shape == (max_seq_length, acoustic_dim), \
            f"acoustic shape {acoustic.shape} != ({max_seq_length}, {acoustic_dim})"
        if use_hcf:
            assert hcf.shape == (max_seq_length, hcf_dim), \
                f"hcf shape {hcf.shape} != ({max_seq_length}, {hcf_dim})"

        feat = {
            "input_ids": input_ids,
            "input_mask": input_mask,
            "visual": visual,
            "acoustic": acoustic,
            "label": float(label),
        }
        if use_hcf:
            feat["hcf"] = hcf
        out.append(feat)
    return out


def _resolve_pkl_path(dataset, datasets_dir):
    candidates = PKL_FILENAMES.get(dataset)
    if not candidates:
        raise ValueError(f"Unsupported humor dataset: {dataset}")
    for name in candidates:
        path = os.path.join(datasets_dir, name)
        if os.path.exists(path):
            return path
    searched = [os.path.join(datasets_dir, n) for n in candidates]
    raise FileNotFoundError(
        f"Could not find pkl for dataset='{dataset}'. Tried: {searched}"
    )


def features_to_dataset(feats, include_hcf):
    tensors = [
        torch.tensor(np.array([f["input_ids"] for f in feats]), dtype=torch.long),
        torch.tensor(np.array([f["visual"] for f in feats]), dtype=torch.float),
        torch.tensor(np.array([f["acoustic"] for f in feats]), dtype=torch.float),
    ]
    if include_hcf:
        tensors.append(
            torch.tensor(np.array([f["hcf"] for f in feats]), dtype=torch.float)
        )
    tensors.append(
        torch.tensor(np.array([f["label"] for f in feats]), dtype=torch.float)
    )
    return TensorDataset(*tensors)


def build_humor_loaders(dataset, tokenizer, max_seq_length,
                        acoustic_dim, visual_dim,
                        train_batch_size, dev_batch_size, test_batch_size,
                        gradient_accumulation_step, n_epochs,
                        hcf_dim=0, slice_hkt=True,
                        datasets_dir=None, pad_token_id=None):
    """Return (train_dl, dev_dl, test_dl, n_optimization_steps).

    Args:
        hcf_dim: when > 0, returned batches include an HCF tensor in position 3
            (layout: input_ids, visual, acoustic, hcf, label). When 0, HCF is
            omitted (layout: input_ids, visual, acoustic, label), matching the
            prior 3-modality behaviour.
        slice_hkt: apply HKT's acoustic[:, 0:60] / visual[:, 55:91] slicing; the
            ``acoustic_dim`` / ``visual_dim`` arguments must match the post-slice
            dims (60 / 36 for UR-FUNNY / MUStARD).

    n_optimization_steps mirrors train.py's `n_opt` calculation so the LR
    scheduler computes warmup correctly.
    """
    if datasets_dir is None:
        datasets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
    if pad_token_id is None:
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    pkl_path = _resolve_pkl_path(dataset, datasets_dir)
    with open(pkl_path, "rb") as fh:
        data = pickle.load(fh)

    train_split = data.get("train", [])
    dev_split = data.get("dev", data.get("valid", []))
    test_split = data.get("test", [])

    if not dev_split:
        raise KeyError("No dev/valid split found in pkl")

    common_kwargs = dict(
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        acoustic_dim=acoustic_dim,
        visual_dim=visual_dim,
        pad_token_id=pad_token_id,
        hcf_dim=hcf_dim,
        slice_hkt=slice_hkt,
    )

    train_feats = convert_to_features(train_split, **common_kwargs)
    dev_feats = convert_to_features(dev_split, **common_kwargs)
    test_feats = convert_to_features(test_split, **common_kwargs)

    include_hcf = hcf_dim > 0
    train_ds = features_to_dataset(train_feats, include_hcf)
    dev_ds = features_to_dataset(dev_feats, include_hcf)
    test_ds = features_to_dataset(test_feats, include_hcf)

    n_opt = int(len(train_ds) / train_batch_size
                / max(gradient_accumulation_step, 1)) * n_epochs

    train_dl = DataLoader(train_ds, batch_size=train_batch_size, shuffle=True)
    dev_dl = DataLoader(dev_ds, batch_size=dev_batch_size, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=test_batch_size, shuffle=False)

    return train_dl, dev_dl, test_dl, n_opt
