"""Utilities for checkpoint selection and HPO objectives."""

DEFAULT_SELECTION_METRIC = "acc2_composite"

SELECTION_METRIC_CHOICES = (
    "acc2_composite",
    "acc2",
    "acc7",
    "acc5",
    "f1",
    "corr",
    "mae",
    "legacy_dev_score",
)

_HIGHER_IS_BETTER = {
    "acc2_composite",
    "acc2",
    "acc7",
    "acc5",
    "f1",
    "corr",
}


def selection_higher_is_better(metric):
    if metric not in SELECTION_METRIC_CHOICES:
        raise ValueError(f"Unsupported selection metric: {metric}")
    return metric in _HIGHER_IS_BETTER


def compute_selection_score(metric, acc2, mae, corr, f1,
                            acc7=None, acc5=None, acc3=None):
    """Compute selection score. Pass acc7 for MOSI/MOSEI, acc5/acc3 for SIMSV2."""
    if metric == "acc2_composite":
        acc_multi = acc5 if acc5 is not None else (acc7 or 0)
        return acc2 + 0.05 * acc_multi + 0.03 * f1 + 0.02 * corr - 0.02 * mae
    if metric == "acc2":
        return acc2
    if metric == "acc7":
        return acc7 if acc7 is not None else 0
    if metric == "acc5":
        return acc5 if acc5 is not None else 0
    if metric == "f1":
        return f1
    if metric == "corr":
        return corr
    if metric == "mae":
        return mae
    if metric == "legacy_dev_score":
        return mae - 0.5 * corr
    raise ValueError(f"Unsupported selection metric: {metric}")


def build_selection_tiebreak(acc2, mae, corr, f1,
                             acc7=None, acc5=None, acc3=None):
    acc_multi = acc5 if acc5 is not None else (acc7 or 0)
    return (acc2, f1, acc_multi, corr, -mae)
