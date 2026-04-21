"""HKT-style (context + punchline) pkl loader for binary classification tasks.

Used by `train_classify.py` / `test_classify.py` for UR-FUNNY (humor) and
MUSTARD (sarcasm). Mirrors HKT's `convert_humor_to_features` logic but emits
a 4-tensor TensorDataset compatible with InfoGate's existing forward signature
(input_ids, visual, acoustic, label).

Sample layout (verified live):
    ((p_words, p_vis, p_aco, p_hcf),     # punchline segment (single sentence str)
     (c_words, c_vis, c_aco, c_hcf),     # context segment (list of sentence str)
     hid,
     label)                              # int 0/1
Visual/acoustic/HCF rows are per-word (HCF is dropped here).
"""

import os
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


SPIECE_MARKER = "\u2581"  # SentencePiece word-start marker used by DeBERTa-v3 / ALBERT


PKL_FILENAMES = {
    "ur_funny": ["ur_funny.pkl", "urfunnyv2.pkl"],
    "mustard":  ["mustard.pkl"],
}


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


def convert_to_features(samples, tokenizer, max_seq_length, acoustic_dim, visual_dim,
                        pad_token_id):
    """Vectorize HKT samples into InfoGate-friendly tensors.

    Returns a list of dicts with keys: input_ids, input_mask, visual, acoustic, label.
    """
    out = []
    cls = tokenizer.cls_token
    sep = tokenizer.sep_token
    for ex in samples:
        # HKT order: first tuple = punchline; second tuple = context.
        (p_words, p_vis, p_aco, _p_hcf), (c_words, c_vis, c_aco, _c_hcf), _hid, label = ex

        text_a = _to_text(c_words, append_period=False)               # context
        text_b = _to_text(p_words, append_period=True)                # punchline + "."

        tokens_a = tokenizer.tokenize(text_a) if text_a else []
        tokens_b = tokenizer.tokenize(text_b) if text_b else []

        inversions_a = get_inversion(tokens_a)
        inversions_b = get_inversion(tokens_b)

        pop = _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        inversions_a = inversions_a[pop:][: len(tokens_a)]
        inversions_b = inversions_b[: len(tokens_b)]

        c_vis_arr = np.asarray(c_vis, dtype=np.float32)
        c_aco_arr = np.asarray(c_aco, dtype=np.float32)
        p_vis_arr = np.asarray(p_vis, dtype=np.float32)
        p_aco_arr = np.asarray(p_aco, dtype=np.float32)

        if c_vis_arr.ndim == 1 and c_vis_arr.size == 0:
            c_vis_arr = np.zeros((0, visual_dim), dtype=np.float32)
        if c_aco_arr.ndim == 1 and c_aco_arr.size == 0:
            c_aco_arr = np.zeros((0, acoustic_dim), dtype=np.float32)
        if p_vis_arr.ndim == 1 and p_vis_arr.size == 0:
            p_vis_arr = np.zeros((0, visual_dim), dtype=np.float32)
        if p_aco_arr.ndim == 1 and p_aco_arr.size == 0:
            p_aco_arr = np.zeros((0, acoustic_dim), dtype=np.float32)

        visual_a = _gather_features(inversions_a, c_vis_arr) if tokens_a else np.zeros((0, visual_dim), dtype=np.float32)
        acoustic_a = _gather_features(inversions_a, c_aco_arr) if tokens_a else np.zeros((0, acoustic_dim), dtype=np.float32)
        visual_b = _gather_features(inversions_b, p_vis_arr) if tokens_b else np.zeros((0, visual_dim), dtype=np.float32)
        acoustic_b = _gather_features(inversions_b, p_aco_arr) if tokens_b else np.zeros((0, acoustic_dim), dtype=np.float32)

        tokens = [cls] + tokens_a + [sep] + tokens_b + [sep]

        zv = np.zeros((1, visual_dim), dtype=np.float32)
        za = np.zeros((1, acoustic_dim), dtype=np.float32)
        if not tokens_a:
            visual = np.concatenate((zv, zv, visual_b, zv), axis=0)
            acoustic = np.concatenate((za, za, acoustic_b, za), axis=0)
        else:
            visual = np.concatenate((zv, visual_a, zv, visual_b, zv), axis=0)
            acoustic = np.concatenate((za, acoustic_a, za, acoustic_b, za), axis=0)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        # If for any reason features and tokens disagree in length, harmonize to len(tokens)
        if visual.shape[0] < len(input_ids):
            pad_v = np.zeros((len(input_ids) - visual.shape[0], visual_dim), dtype=np.float32)
            visual = np.concatenate((visual, pad_v), axis=0)
        else:
            visual = visual[: len(input_ids)]
        if acoustic.shape[0] < len(input_ids):
            pad_a = np.zeros((len(input_ids) - acoustic.shape[0], acoustic_dim), dtype=np.float32)
            acoustic = np.concatenate((acoustic, pad_a), axis=0)
        else:
            acoustic = acoustic[: len(input_ids)]

        pad = max_seq_length - len(input_ids)
        if pad > 0:
            input_ids = input_ids + [pad_token_id] * pad
            input_mask = input_mask + [0] * pad
            visual = np.concatenate((visual, np.zeros((pad, visual_dim), dtype=np.float32)), axis=0)
            acoustic = np.concatenate((acoustic, np.zeros((pad, acoustic_dim), dtype=np.float32)), axis=0)
        else:
            input_ids = input_ids[:max_seq_length]
            input_mask = input_mask[:max_seq_length]
            visual = visual[:max_seq_length]
            acoustic = acoustic[:max_seq_length]

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert visual.shape == (max_seq_length, visual_dim)
        assert acoustic.shape == (max_seq_length, acoustic_dim)

        out.append({
            "input_ids": input_ids,
            "input_mask": input_mask,
            "visual": visual,
            "acoustic": acoustic,
            "label": float(label),
        })
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


def features_to_dataset(feats):
    return TensorDataset(
        torch.tensor(np.array([f["input_ids"] for f in feats]), dtype=torch.long),
        torch.tensor(np.array([f["visual"] for f in feats]), dtype=torch.float),
        torch.tensor(np.array([f["acoustic"] for f in feats]), dtype=torch.float),
        torch.tensor(np.array([f["label"] for f in feats]), dtype=torch.float),
    )


def build_humor_loaders(dataset, tokenizer, max_seq_length,
                        acoustic_dim, visual_dim,
                        train_batch_size, dev_batch_size, test_batch_size,
                        gradient_accumulation_step, n_epochs,
                        datasets_dir=None, pad_token_id=None):
    """Return (train_dl, dev_dl, test_dl, n_optimization_steps).

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

    train_feats = convert_to_features(
        train_split, tokenizer, max_seq_length, acoustic_dim, visual_dim, pad_token_id)
    dev_feats = convert_to_features(
        dev_split, tokenizer, max_seq_length, acoustic_dim, visual_dim, pad_token_id)
    test_feats = convert_to_features(
        test_split, tokenizer, max_seq_length, acoustic_dim, visual_dim, pad_token_id)

    train_ds = features_to_dataset(train_feats)
    dev_ds = features_to_dataset(dev_feats)
    test_ds = features_to_dataset(test_feats)

    n_opt = int(len(train_ds) / train_batch_size
                / max(gradient_accumulation_step, 1)) * n_epochs

    train_dl = DataLoader(train_ds, batch_size=train_batch_size, shuffle=True)
    dev_dl = DataLoader(dev_ds, batch_size=dev_batch_size, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=test_batch_size, shuffle=False)

    return train_dl, dev_dl, test_dl, n_opt
