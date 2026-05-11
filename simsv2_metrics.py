"""SIMSV2 metric helpers aligned with KuDA's interval-binning protocol.

This module mirrors KuDA's SIMS/SIMSV2 clipping and bucket definitions while
keeping F1 on the standard sklearn argument order:
    f1_score(y_true, y_pred, average="weighted")
"""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


SIMSV2_CLIP_MIN = -1.0
SIMSV2_CLIP_MAX = 1.0

SIMSV2_BINS_2 = (-1.01, 0.0, 1.01)
SIMSV2_BINS_3 = (-1.01, -0.1, 0.1, 1.01)
SIMSV2_BINS_5 = (-1.01, -0.7, -0.1, 0.1, 0.7, 1.01)


def clip_simsv2_pairs(preds, labels):
    """Clip prediction/label pairs to the KuDA SIMSV2 regression range."""
    preds = np.asarray(preds).flatten()
    labels = np.asarray(labels).flatten()
    if preds.shape != labels.shape:
        raise ValueError(f"pred/label shape mismatch: {preds.shape} vs {labels.shape}")
    preds = np.clip(preds, SIMSV2_CLIP_MIN, SIMSV2_CLIP_MAX)
    labels = np.clip(labels, SIMSV2_CLIP_MIN, SIMSV2_CLIP_MAX)
    return preds, labels


def _bucketize(values, boundaries):
    bucketed = np.full(values.shape, -1, dtype=np.int64)
    for idx in range(len(boundaries) - 1):
        lo = boundaries[idx]
        hi = boundaries[idx + 1]
        mask = np.logical_and(values > lo, values <= hi)
        bucketed[mask] = idx
    if np.any(bucketed < 0):
        raise ValueError("Some SIMSV2 values were not assigned to any interval bucket.")
    return bucketed


def _safe_corrcoef(preds, labels):
    if len(preds) < 2:
        return 0.0
    corr = np.corrcoef(preds, labels)[0][1]
    return 0.0 if np.isnan(corr) else float(corr)


def compute_simsv2_kuda_metrics(preds, labels):
    """Compute KuDA-style SIMSV2 metrics with standard F1 argument order."""
    preds, labels = clip_simsv2_pairs(preds, labels)

    preds_a2 = _bucketize(preds, SIMSV2_BINS_2)
    labels_a2 = _bucketize(labels, SIMSV2_BINS_2)
    preds_a3 = _bucketize(preds, SIMSV2_BINS_3)
    labels_a3 = _bucketize(labels, SIMSV2_BINS_3)
    preds_a5 = _bucketize(preds, SIMSV2_BINS_5)
    labels_a5 = _bucketize(labels, SIMSV2_BINS_5)

    mae = float(np.mean(np.abs(preds - labels)))
    corr = _safe_corrcoef(preds, labels)
    acc2 = accuracy_score(labels_a2, preds_a2)
    acc3 = accuracy_score(labels_a3, preds_a3)
    acc5 = accuracy_score(labels_a5, preds_a5)
    f1 = f1_score(labels_a2, preds_a2, average="weighted")

    return {
        "acc2": float(acc2),
        "acc3": float(acc3),
        "acc5": float(acc5),
        "f1": float(f1),
        "mae": mae,
        "corr": corr,
    }
