"""Optuna v2 — MOSI/MOSEI/SIMSV2 hyperparameter search for InfoGate.
Multi-objective (acc2_composite ↑, MAE ↓): BoTorch Sampler for rapid convergence.
SQLite persistence, 3-tier search space, adaptive pruning.
"""

import argparse
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime

import optuna
from optuna.samplers import RandomSampler, TPESampler
from optuna.integration.botorch import BoTorchSampler

from selection_utils import (
    DEFAULT_SELECTION_METRIC,
    SELECTION_METRIC_CHOICES,
    compute_selection_score,
    selection_higher_is_better,
)

# ── paths ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_SCRIPT = os.path.join(SCRIPT_DIR, "train.py")
PYTHON = sys.executable
KEEP_TOP_K = 5

# ── regex for log parsing ──
DEV_LINE_RE = re.compile(
    r"\s+Dev\s+Acc2=([\d.]+)%\s+Acc7=([\d.]+)%\s+MAE=([\d.]+)\s+Corr=([\d.]+)\s+F1=([\d.]+)"
)
DEV_LINE_SIMSV2_RE = re.compile(
    r"\s+Dev\s+Acc2=([\d.]+)%\s+Acc5=([\d.]+)%\s+Acc3=([\d.]+)%\s+MAE=([\d.]+)\s+Corr=([\d.]+)\s+F1=([\d.]+)"
)
EPOCH_LINE_RE = re.compile(r"Epoch (\d+)/\d+")
RESULT_LINE_RE = re.compile(
    r"\s+(Selection score|Acc-2|Acc-7|Acc-5|Acc-3|MAE|Corr|F1):\s+([\d.]+)%?"
)

# ── fixed defaults for params not in active tier ──
DEFAULTS = {
    "seed": 128, "learning_rate": 2e-5, "ig_learning_rate": 5e-4,
    "beta_ib": 32.0, "num_infogate_layers": 3, "bottleneck_dim": 128,
    "mse_weight": 0.5, "dropout_prob": 0.1,
    "alpha_ib": 0.01,
    "stage1_epochs": 10, "warmup_proportion": 0.1,
    "weight_decay": 1e-3, "ema_decay": 0.999,
    "selector_target_temp": 0.35, "selector_rib_weight": 0.05,
    "gumbel_tau_start": 1.0, "gumbel_tau_end": 0.5,
    "num_heads": 4, "unified_dim": 256,
    "ema_start_epoch": 5,
}

LOG_FLOAT_BOUNDS = {
    "learning_rate": (5e-6, 5e-5),
    "ig_learning_rate": (5e-5, 2e-3),
    "beta_ib": (4.0, 64.0),
    "alpha_ib": (0.001, 0.05),
    "weight_decay": (1e-4, 0.1),
}

LINEAR_FLOAT_BOUNDS = {
    # Upper bounds widened slightly after 4090D_restart/phase1 MOSI best (trial 52)
    # saturated dropout~0.40 and mse_weight~1.92 under the old [0,2] cap.
    "mse_weight": (0.0, 2.25),
    "dropout_prob": (0.05, 0.48),
    "warmup_proportion": (0.02, 0.25),
    "selector_target_temp": (0.1, 1.0),
    "selector_rib_weight": (0.01, 0.2),
    "gumbel_tau_start": (0.5, 2.0),
    "gumbel_tau_end": (0.1, 1.0),
}

INT_BOUNDS = {
    "stage1_epochs": (3, 20),
}

CATEGORICAL_SPACE = {
    "num_infogate_layers": [2, 3, 4, 5],
    "bottleneck_dim": [64, 96, 128, 192],
    "ema_decay": [0.99, 0.995, 0.999, 0.9995],
    "num_heads": [2, 4, 8],
    "unified_dim": [128, 256, 384],
    "ema_start_epoch": [3, 5, 8, 10],
}


# ══════════════════════════════════════════════════════════════
# Search space (3 tiers)
# ══════════════════════════════════════════════════════════════

def suggest_float_param(trial, name, bounds, *, log=False, local_space=None):
    low, high = bounds
    if local_space and name in local_space:
        low, high = local_space[name]
    return trial.suggest_float(name, low, high, log=log)


def suggest_int_param(trial, name, bounds, local_space=None):
    low, high = bounds
    if local_space and name in local_space:
        low, high = local_space[name]
    return trial.suggest_int(name, low, high)


def suggest_categorical_param(trial, name, choices, local_space=None):
    if local_space and name in local_space:
        choices = local_space[name]
    return trial.suggest_categorical(name, list(choices))


def suggest_tier1(trial, dataset="mosi", n_epochs_max=None, n_epochs_min=None,
                  local_space=None, search_tier=1):
    candidates = BATCH_CANDIDATES[dataset]
    batch_idx = suggest_categorical_param(
        trial, "batch_config", list(range(len(candidates))), local_space)
    bs, accum = candidates[batch_idx]
    ep_lo, ep_hi = DATASET_EPOCH_RANGE[dataset]
    # Tier>=2 MOSI: allow a few more epochs (stage-2 + extra IB knobs need runway);
    # floor nudged up slightly since very short runs rarely compete on dev-MAE.
    if dataset == "mosi" and int(search_tier) >= 2:
        ep_lo = max(ep_lo, 88)
        ep_hi = min(135, ep_hi + 5)
    if n_epochs_min is not None:
        ep_lo = n_epochs_min
    if n_epochs_max is not None:
        ep_hi = n_epochs_max
    if local_space and "n_epochs" in local_space:
        local_lo, local_hi = local_space["n_epochs"]
        ep_lo = max(ep_lo, local_lo)
        ep_hi = min(ep_hi, local_hi)
    return {
        "train_batch_size": bs,
        "gradient_accumulation_step": accum,
        "n_epochs": trial.suggest_int("n_epochs", ep_lo, ep_hi),
        "seed": 128,  # Fixed seed
        "learning_rate": suggest_float_param(
            trial, "learning_rate", LOG_FLOAT_BOUNDS["learning_rate"], log=True,
            local_space=local_space),
        "ig_learning_rate": suggest_float_param(
            trial, "ig_learning_rate", LOG_FLOAT_BOUNDS["ig_learning_rate"], log=True,
            local_space=local_space),
        "beta_ib": suggest_float_param(
            trial, "beta_ib", LOG_FLOAT_BOUNDS["beta_ib"], log=True,
            local_space=local_space),
        "num_infogate_layers": suggest_categorical_param(
            trial, "num_infogate_layers", CATEGORICAL_SPACE["num_infogate_layers"],
            local_space),
        "bottleneck_dim": suggest_categorical_param(
            trial, "bottleneck_dim", CATEGORICAL_SPACE["bottleneck_dim"],
            local_space),
        "mse_weight": suggest_float_param(
            trial, "mse_weight", LINEAR_FLOAT_BOUNDS["mse_weight"],
            local_space=local_space),
        "dropout_prob": suggest_float_param(
            trial, "dropout_prob", LINEAR_FLOAT_BOUNDS["dropout_prob"],
            local_space=local_space),
    }


def suggest_tier2(trial, local_space=None):
    return {
        "alpha_ib": suggest_float_param(
            trial, "alpha_ib", LOG_FLOAT_BOUNDS["alpha_ib"], log=True,
            local_space=local_space),
        "stage1_epochs": suggest_int_param(
            trial, "stage1_epochs", INT_BOUNDS["stage1_epochs"], local_space),
        "warmup_proportion": suggest_float_param(
            trial, "warmup_proportion", LINEAR_FLOAT_BOUNDS["warmup_proportion"],
            local_space=local_space),
        "weight_decay": suggest_float_param(
            trial, "weight_decay", LOG_FLOAT_BOUNDS["weight_decay"], log=True,
            local_space=local_space),
        "ema_decay": suggest_categorical_param(
            trial, "ema_decay", CATEGORICAL_SPACE["ema_decay"], local_space),
    }


def suggest_tier3(trial, local_space=None):
    return {
        "selector_target_temp": suggest_float_param(
            trial, "selector_target_temp",
            LINEAR_FLOAT_BOUNDS["selector_target_temp"], local_space=local_space),
        "selector_rib_weight": suggest_float_param(
            trial, "selector_rib_weight",
            LINEAR_FLOAT_BOUNDS["selector_rib_weight"], local_space=local_space),
        "gumbel_tau_start": suggest_float_param(
            trial, "gumbel_tau_start", LINEAR_FLOAT_BOUNDS["gumbel_tau_start"],
            local_space=local_space),
        "gumbel_tau_end": suggest_float_param(
            trial, "gumbel_tau_end", LINEAR_FLOAT_BOUNDS["gumbel_tau_end"],
            local_space=local_space),
        "num_heads": suggest_categorical_param(
            trial, "num_heads", CATEGORICAL_SPACE["num_heads"], local_space),
        "unified_dim": suggest_categorical_param(
            trial, "unified_dim", CATEGORICAL_SPACE["unified_dim"], local_space),
        "ema_start_epoch": suggest_categorical_param(
            trial, "ema_start_epoch", CATEGORICAL_SPACE["ema_start_epoch"],
            local_space),
    }


def build_search_params(trial, tier, dataset="mosi", n_epochs_max=None, n_epochs_min=None,
                        local_space=None):
    params = dict(DEFAULTS)
    params.update(suggest_tier1(
        trial, dataset, n_epochs_max, n_epochs_min,
        local_space=local_space, search_tier=tier))
    if tier >= 2:
        params.update(suggest_tier2(trial, local_space=local_space))
    if tier >= 3:
        params.update(suggest_tier3(trial, local_space=local_space))
    return params


# ══════════════════════════════════════════════════════════════
# Log parsing
# ══════════════════════════════════════════════════════════════

def parse_best_results(log_path):
    results = {}
    if not os.path.exists(log_path):
        return results
    in_block = False
    with open(log_path, "r") as f:
        for line in f:
            if line.startswith("Best Results"):
                in_block = True
                continue
            if in_block and line.startswith("Last Epoch"):
                break
            if in_block:
                m = RESULT_LINE_RE.match(line)
                if not m:
                    continue
                key, raw = m.groups()
                val = float(raw)
                if key in ("Acc-2", "Acc-7", "Acc-5", "Acc-3"):
                    val /= 100.0
                if key == "Selection score":
                    key = "SelectionScore"
                results[key] = val
    return results


def parse_best_dev_metrics(log_path, dataset="mosi",
                           selection_metric="acc2_composite",
                           stage1_epochs=0):
    """Return (current_epoch, best_dev_metrics).

    Dev lines emitted during stage 1 warmup (1-based epoch <= stage1_epochs)
    are IGNORED when tracking the best dev score. This aligns with train.py
    where `select_start = args.stage1_epochs` — the final reported "Best
    Results" there also only considers stage-2 epochs. Using a stage-1-
    inclusive best would both (a) pollute `trial.report()` with warmup noise
    and (b) cause aggressive prunes when the stage-2 runway is still short.
    """
    if not os.path.exists(log_path):
        return 0, None
    current_epoch = 0
    best = None
    higher_is_better = selection_higher_is_better(selection_metric)
    best_score = None
    with open(log_path, "r") as f:
        for line in f:
            em = EPOCH_LINE_RE.match(line)
            if em:
                current_epoch = int(em.group(1))
                continue
            # Try simsv2 format first (more specific), then fallback
            dm_s = DEV_LINE_SIMSV2_RE.match(line)
            dm = DEV_LINE_RE.match(line) if dm_s is None else None
            # Skip stage-1 warmup dev lines. EPOCH_LINE_RE yields the 1-based
            # epoch index printed by train.py; stage 2 begins when
            # `current_epoch > stage1_epochs`.
            if (dm_s or dm) and current_epoch <= int(stage1_epochs):
                continue
            if dm_s:
                acc2, acc5, acc3, mae, corr, f1 = (float(x) for x in dm_s.groups())
                acc2 /= 100.0
                acc5 /= 100.0
                acc3 /= 100.0
                score = compute_selection_score(
                    selection_metric, acc2=acc2, mae=mae, corr=corr, f1=f1,
                    acc5=acc5, acc3=acc3)
                if better_than(score, best_score, higher_is_better):
                    best_score = score
                    best = {"Acc2": acc2, "Acc5": acc5, "Acc3": acc3,
                            "MAE": mae, "Corr": corr, "F1": f1}
            elif dm:
                acc2, acc7, mae, corr, f1 = (float(x) for x in dm.groups())
                acc2 /= 100.0
                acc7 /= 100.0
                score = compute_selection_score(
                    selection_metric, acc2=acc2, mae=mae, corr=corr, f1=f1,
                    acc7=acc7)
                if better_than(score, best_score, higher_is_better):
                    best_score = score
                    best = {"Acc2": acc2, "Acc7": acc7, "MAE": mae,
                            "Corr": corr, "F1": f1}
    return current_epoch, best


# ══════════════════════════════════════════════════════════════
# Per-dataset pruning — thresholds are in *stage 2* epochs.
# `epoch` is the 1-based epoch reported in train.py's log; `stage1_epochs`
# is the number of warmup epochs sampled by the trial. We NEVER prune
# during stage 1 and give stage 2 at least 10 epochs of grace before
# the first check — previously the first gate fired at s2_ep == 5, which
# the user flagged as too eager (a slow-burning LR/β_ib combo needs a
# longer stage-2 runway before its dev metric becomes diagnostic).
# AND logic: a trial is pruned only when BOTH acc2 AND mae fail.
# ══════════════════════════════════════════════════════════════

def _s2_epoch(epoch, stage1_epochs):
    """Epochs completed inside stage 2 (>=0). 0 while still in stage 1."""
    return max(0, int(epoch) - int(stage1_epochs))


def should_prune_mosi(epoch, metrics, stage1_epochs):
    if metrics is None:
        return False
    s2 = _s2_epoch(epoch, stage1_epochs)
    if s2 < 10:
        return False
    acc2, mae = metrics["Acc2"], metrics["MAE"]
    if s2 >= 40:
        return acc2 < 0.84 and mae > 0.66
    if s2 >= 20:
        return acc2 < 0.80 and mae > 0.70
    if s2 >= 10:
        return acc2 < 0.72 and mae > 0.85
    return False


def should_prune_mosei(epoch, metrics, stage1_epochs):
    if metrics is None:
        return False
    s2 = _s2_epoch(epoch, stage1_epochs)
    if s2 < 15:
        return False
    acc2, mae = metrics["Acc2"], metrics["MAE"]
    if s2 >= 50:
        return acc2 < 0.76 and mae > 0.70
    if s2 >= 30:
        return acc2 < 0.70 and mae > 0.80
    if s2 >= 15:
        return acc2 < 0.55 and mae > 0.95
    return False


def should_prune_simsv2(epoch, metrics, stage1_epochs):
    if metrics is None:
        return False
    s2 = _s2_epoch(epoch, stage1_epochs)
    if s2 < 10:
        return False
    acc2, mae = metrics["Acc2"], metrics["MAE"]
    if s2 >= 30:
        return acc2 < 0.72 and mae > 0.55
    if s2 >= 20:
        return acc2 < 0.68 and mae > 0.58
    if s2 >= 10:
        return acc2 < 0.62 and mae > 0.65
    return False


PRUNE_FN = {
    "mosi": should_prune_mosi,
    "mosei": should_prune_mosei,
    "simsv2": should_prune_simsv2,
}

# ── per-dataset batch candidates: (batch_size, grad_accum) ──
# effective batch = batch_size * grad_accum
BATCH_CANDIDATES = {
    "mosi":   [(8, 4), (8, 8), (16, 2), (16, 4), (32, 1), (32, 2)],
    "mosei":  [(4, 8), (4, 16), (8, 4), (8, 8), (16, 2), (16, 4)],
    "simsv2": [(8, 4), (16, 2), (16, 4), (32, 1), (32, 2), (64, 1)],
}

DATASET_EPOCH_RANGE = {
    "mosi":   (80, 130),
    "mosei":  (30, 50),
    "simsv2": (45, 85),
}


def apply_dataset_bounds_overrides(dataset):
    """Per-dataset overrides for search bounds, applied at startup.

    SIMSV2: widened/repositioned based on top-trial analysis (ema_decay,
    dropout_prob, learning_rate, n_epochs, mse_weight near upper bound,
    batch grid, bottleneck_dim).

    MOSI: mse_weight upper bound raised from 2.0 to 3.5 — Stage2 best trials
    clustered near mse_weight≈2.0 (saved in saved_hparams/ before this change).
    """
    if dataset == "mosi":
        # Tier-3 widened/tightened bounds for v3, based on TOP-12 of msew35_s2_local
        # (MAE-min): all top trials cluster at batch_config=(32,1), bd=96, nL=5,
        # ema_decay=0.995, and stage1_epochs hit upper bound 20.
        LOG_FLOAT_BOUNDS["learning_rate"]    = (1e-5, 3e-5)
        LOG_FLOAT_BOUNDS["ig_learning_rate"] = (2e-4, 1e-3)
        LOG_FLOAT_BOUNDS["beta_ib"]          = (20.0, 50.0)
        LOG_FLOAT_BOUNDS["alpha_ib"]         = (3e-3, 1.5e-2)
        LOG_FLOAT_BOUNDS["weight_decay"]     = (3e-4, 5e-3)
        LINEAR_FLOAT_BOUNDS["mse_weight"]    = (0.5, 3.0)
        LINEAR_FLOAT_BOUNDS["dropout_prob"]  = (0.20, 0.45)   # extend upper from 0.4
        LINEAR_FLOAT_BOUNDS["warmup_proportion"] = (0.05, 0.20)
        INT_BOUNDS["stage1_epochs"]          = (10, 25)       # extend upper from 20
        DATASET_EPOCH_RANGE["mosi"]          = (100, 135)
        CATEGORICAL_SPACE["bottleneck_dim"]  = [96, 128]
        CATEGORICAL_SPACE["num_infogate_layers"] = [4, 5]
        CATEGORICAL_SPACE["ema_decay"]       = [0.995, 0.999]
        BATCH_CANDIDATES["mosi"]             = [(16, 2), (32, 1), (32, 2)]
    if dataset == "simsv2":
        LOG_FLOAT_BOUNDS["learning_rate"] = (5e-6, 1e-4)
        LOG_FLOAT_BOUNDS["alpha_ib"] = (1e-3, 0.1)
        LOG_FLOAT_BOUNDS["weight_decay"] = (1e-4, 0.3)
        LINEAR_FLOAT_BOUNDS["mse_weight"] = (0.0, 5.0)
        LINEAR_FLOAT_BOUNDS["dropout_prob"] = (0.0, 0.20)
        LINEAR_FLOAT_BOUNDS["warmup_proportion"] = (0.05, 0.40)
        INT_BOUNDS["stage1_epochs"] = (4, 16)
        DATASET_EPOCH_RANGE["simsv2"] = (60, 110)
        CATEGORICAL_SPACE["bottleneck_dim"] = [96, 128, 192]
        CATEGORICAL_SPACE["num_infogate_layers"] = [3, 4]
        CATEGORICAL_SPACE["ema_decay"] = [0.995, 0.999, 0.9995, 0.99975, 0.9999]
        BATCH_CANDIDATES["simsv2"] = [(16, 4), (32, 2), (64, 1), (32, 4)]


def better_than(candidate, best_value, higher_is_better):
    if best_value is None:
        return True
    return candidate > best_value if higher_is_better else candidate < best_value


def clone_cli(cli, **updates):
    data = vars(cli).copy()
    data.update(updates)
    return argparse.Namespace(**data)


def sanitize_name(name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def sqlite_uri_to_path(db_uri: str):
    """Return filesystem path for sqlite:///... URIs; None if not a file sqlite URI."""
    if not db_uri or not isinstance(db_uri, str):
        return None
    prefix = "sqlite:///"
    if not db_uri.startswith(prefix):
        return None
    path = db_uri[len(prefix):]
    path = os.path.normpath(path)
    if not os.path.isabs(path):
        path = os.path.join(SCRIPT_DIR, path)
    return os.path.abspath(path)


def infer_artefact_root_from_db_uri(db_uri: str):
    """If DB lives under <run>/db/*.db, return <run> so trial logs/checkpoints stay per-run."""
    db_path = sqlite_uri_to_path(db_uri)
    if not db_path:
        return None
    parent = os.path.dirname(db_path)
    if os.path.basename(parent) == "db":
        return os.path.dirname(parent)
    return None


def resolve_artefact_root(cli):
    explicit = getattr(cli, "artefact_root", None)
    if explicit:
        return os.path.abspath(explicit)
    return infer_artefact_root_from_db_uri(cli.db)


def trial_train_log_dir(cli):
    root = getattr(cli, "artefact_root", None)
    if root:
        return os.path.join(root, "train_logs")
    return os.path.join(SCRIPT_DIR, "logs", "optuna")


def trial_ckpt_base(cli, dataset, stage_label):
    if stage_label:
        sub = f"optuna_{dataset}_{stage_label}"
    else:
        sub = f"optuna_{dataset}"
    root = getattr(cli, "artefact_root", None)
    if root:
        return os.path.join(root, "checkpoints", sub)
    return os.path.join(SCRIPT_DIR, "checkpoints", sub)


def ordered_choice_subset(original_choices, values, best_value=None):
    wanted = set(values)
    if best_value is not None:
        wanted.add(best_value)
    subset = [choice for choice in original_choices if choice in wanted]
    if subset:
        return subset
    if best_value is not None:
        return [best_value]
    return list(original_choices)


def narrow_linear_range(values, bounds, *, pad_ratio=0.2, min_pad_ratio=0.05):
    low_bound, high_bound = bounds
    v_min = min(values)
    v_max = max(values)
    span = v_max - v_min
    full_span = high_bound - low_bound
    pad = max(span * pad_ratio, full_span * min_pad_ratio)
    low = max(low_bound, v_min - pad)
    high = min(high_bound, v_max + pad)
    if low >= high:
        center = min(max(values[0], low_bound), high_bound)
        pad = max(full_span * min_pad_ratio, 1e-6)
        low = max(low_bound, center - pad)
        high = min(high_bound, center + pad)
    if low >= high:
        return bounds
    return (float(low), float(high))


def narrow_log_range(values, bounds, *, expand=1.25):
    low_bound, high_bound = bounds
    v_min = min(values)
    v_max = max(values)
    low = max(low_bound, v_min / expand)
    high = min(high_bound, v_max * expand)
    if low >= high:
        center = min(max(values[0], low_bound), high_bound)
        low = max(low_bound, center / expand)
        high = min(high_bound, center * expand)
    if low >= high:
        return bounds
    return (float(low), float(high))


def narrow_int_range(values, bounds, *, min_pad=1, pad_ratio=0.2):
    low_bound, high_bound = bounds
    v_min = min(values)
    v_max = max(values)
    pad = max(int(round((v_max - v_min) * pad_ratio)), min_pad)
    low = max(low_bound, v_min - pad)
    high = min(high_bound, v_max + pad)
    if low > high:
        return bounds
    return (int(low), int(high))


def get_completed_trials(study, selection_metric):
    completed = [
        trial for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE and trial.value is not None
    ]
    hib = selection_higher_is_better(selection_metric)
    return sorted(completed, key=lambda trial: trial.value, reverse=hib)


def build_local_search_space(study, dataset, search_tier, selection_metric, top_k):
    ranked = get_completed_trials(study, selection_metric)
    top_trials = ranked[:top_k]
    if not top_trials:
        return None, []

    best_params = top_trials[0].params
    params_list = [trial.params for trial in top_trials]
    local_space = {}

    def values_for(name):
        return [params[name] for params in params_list if name in params]

    batch_choices = values_for("batch_config")
    if batch_choices:
        local_space["batch_config"] = ordered_choice_subset(
            list(range(len(BATCH_CANDIDATES[dataset]))),
            batch_choices,
            best_params.get("batch_config"),
        )

    epoch_values = values_for("n_epochs")
    if epoch_values:
        local_space["n_epochs"] = narrow_int_range(
            epoch_values, DATASET_EPOCH_RANGE[dataset], min_pad=5)

    tier1_log_names = ("learning_rate", "ig_learning_rate", "beta_ib")
    tier1_linear_names = ("mse_weight", "dropout_prob")
    tier1_cat_names = ("num_infogate_layers", "bottleneck_dim")

    for name in tier1_log_names:
        vals = values_for(name)
        if vals:
            local_space[name] = narrow_log_range(vals, LOG_FLOAT_BOUNDS[name])
    for name in tier1_linear_names:
        vals = values_for(name)
        if vals:
            local_space[name] = narrow_linear_range(vals, LINEAR_FLOAT_BOUNDS[name])
    for name in tier1_cat_names:
        vals = values_for(name)
        if vals:
            local_space[name] = ordered_choice_subset(
                CATEGORICAL_SPACE[name], vals, best_params.get(name))

    if search_tier >= 2:
        tier2_log_names = ("alpha_ib", "weight_decay")
        tier2_linear_names = ("warmup_proportion",)
        tier2_cat_names = ("ema_decay",)
        stage1_vals = values_for("stage1_epochs")
        if stage1_vals:
            local_space["stage1_epochs"] = narrow_int_range(
                stage1_vals, INT_BOUNDS["stage1_epochs"], min_pad=2)
        for name in tier2_log_names:
            vals = values_for(name)
            if vals:
                local_space[name] = narrow_log_range(vals, LOG_FLOAT_BOUNDS[name])
        for name in tier2_linear_names:
            vals = values_for(name)
            if vals:
                local_space[name] = narrow_linear_range(vals, LINEAR_FLOAT_BOUNDS[name])
        for name in tier2_cat_names:
            vals = values_for(name)
            if vals:
                local_space[name] = ordered_choice_subset(
                    CATEGORICAL_SPACE[name], vals, best_params.get(name))

    if search_tier >= 3:
        tier3_linear_names = (
            "selector_target_temp", "selector_rib_weight",
            "gumbel_tau_start", "gumbel_tau_end",
        )
        tier3_cat_names = ("num_heads", "unified_dim", "ema_start_epoch")
        for name in tier3_linear_names:
            vals = values_for(name)
            if vals:
                local_space[name] = narrow_linear_range(vals, LINEAR_FLOAT_BOUNDS[name])
        for name in tier3_cat_names:
            vals = values_for(name)
            if vals:
                local_space[name] = ordered_choice_subset(
                    CATEGORICAL_SPACE[name], vals, best_params.get(name))

    return local_space, top_trials


def summarize_local_space(local_space):
    if not local_space:
        return "  Local space: full search space (no completed anchors)\n"

    lines = ["  Local space overrides:"]
    for key in sorted(local_space):
        val = local_space[key]
        if isinstance(val, tuple):
            if all(isinstance(item, int) for item in val):
                lines.append(f"    {key}: [{val[0]}, {val[1]}]")
            else:
                lines.append(f"    {key}: [{val[0]:.4g}, {val[1]:.4g}]")
        else:
            lines.append(f"    {key}: {list(val)}")
    return "\n".join(lines) + "\n"


def build_stage_db_uri(stage_study_name, base_db=None):
    if base_db is None:
        db_path = os.path.join(SCRIPT_DIR, "logs", "optuna", f"{stage_study_name}.db")
        return f"sqlite:///{db_path}"
    if base_db.startswith("sqlite:///"):
        base_path = base_db[len("sqlite:///"):]
        root, ext = os.path.splitext(base_path)
        if not ext:
            ext = ".db"
        db_path = f"{root}_{sanitize_name(stage_study_name)}{ext}"
        return f"sqlite:///{db_path}"
    return base_db


def build_two_stage_study_names(cli):
    if cli.study_name is not None:
        base = sanitize_name(cli.study_name)
        return f"{base}_s1_random", f"{base}_s2_local"
    suffix = sanitize_name(cli.selection_metric)
    return (
        f"infogate_{cli.dataset}_s1_random_{suffix}",
        f"infogate_{cli.dataset}_s2_local_{suffix}",
    )


def create_study_for_cli(cli):
    n_startup = cli.n_startup_trials
    if not cli.multi_objective:
        hib = selection_higher_is_better(cli.selection_metric)
        sampler_name = getattr(cli, "sampler_name", "TPE")
        sampler_seed = getattr(cli, "sampler_seed", 128)
        if sampler_name == "random":
            sampler = RandomSampler(seed=sampler_seed)
            mode_label = "single-obj Random"
        else:
            sampler = TPESampler(n_startup_trials=n_startup, seed=sampler_seed)
            mode_label = "single-obj TPE"
        study = optuna.create_study(
            study_name=cli.study_name,
            storage=cli.db,
            direction="maximize" if hib else "minimize",
            sampler=sampler,
            load_if_exists=True,
        )
        # ── resume-safety: samplers are stateless w.r.t. the DB, so after a
        # process restart `RandomSampler(seed=S)` / `TPESampler(seed=S)` would
        # redraw the same first-N parameter vectors. Shift the seed by the
        # number of trials already persisted so the sequence continues where
        # it left off instead of recreating earlier trials.
        existing = len(study.trials)
        if existing > 0:
            shifted = (int(sampler_seed) + int(existing)) & 0x7FFFFFFF
            if sampler_name == "random":
                study.sampler = RandomSampler(seed=shifted)
            else:
                study.sampler = TPESampler(n_startup_trials=n_startup, seed=shifted)
            mode_label += f" [resume: seed+{existing}]"
        return study, mode_label

    mo_sampler = BoTorchSampler(n_startup_trials=n_startup)
    mode_label = f"multi-obj BoTorch(n_startup_trials={n_startup})"
    study = optuna.create_study(
        study_name=cli.study_name,
        storage=cli.db,
        directions=["maximize", "minimize"],
        sampler=mo_sampler,
        load_if_exists=True,
    )
    return study, mode_label


def optimize_with_cleanup(study, cli, ckpt_base):
    def after_trial(study, trial):
        if trial.state == optuna.trial.TrialState.PRUNED:
            d = os.path.join(ckpt_base, f"trial_{trial.number}")
            if os.path.isdir(d):
                shutil.rmtree(d, ignore_errors=True)
        elif trial.state == optuna.trial.TrialState.COMPLETE:
            if not cli.multi_objective:
                cleanup_checkpoints_single(
                    study, ckpt_base,
                    selection_higher_is_better(cli.selection_metric))
            else:
                cleanup_checkpoints_multi(study, ckpt_base)

    # Resume semantics: cli.n_trials is the TARGET total number of finished
    # trials for this study (COMPLETE + PRUNED + FAIL). When we restart a
    # driver against an existing DB we should only top up the remaining budget
    # instead of running cli.n_trials fresh trials on top of what's already
    # there. RUNNING trials at restart time are counted too (they were
    # attempts; if they're dead orphans they'll resolve as FAIL quickly).
    finished_states = (
        optuna.trial.TrialState.COMPLETE,
        optuna.trial.TrialState.PRUNED,
        optuna.trial.TrialState.FAIL,
        optuna.trial.TrialState.RUNNING,
    )
    existing = sum(1 for t in study.trials if t.state in finished_states)
    remaining = max(0, int(cli.n_trials) - existing)
    if remaining <= 0:
        print(f"[resume] study '{cli.study_name}' already has {existing} trials "
              f">= target {cli.n_trials}; skipping optimize().")
        return
    print(f"[resume] study '{cli.study_name}': {existing} existing trials; "
          f"running {remaining} more to reach target {cli.n_trials}.")
    study.optimize(
        lambda trial: objective(trial, cli),
        n_trials=remaining,
        callbacks=[after_trial],
    )


def print_study_header(cli, mode_label, existing_trials):
    print(f"Optuna v2 — {cli.dataset.upper()}")
    print(f"  Study:   {cli.study_name}")
    print(f"  Storage: {cli.db}")
    if getattr(cli, "artefact_root", None):
        print(f"  Run dir: {cli.artefact_root}")
        print(f"  Train logs: {trial_train_log_dir(cli)}")
    print(f"  GPU:     {cli.gpu}")
    print(f"  Mode:    {mode_label}")
    print(f"  Metric:  {cli.selection_metric}")
    print(f"  Tier:    {cli.search_tier}")
    print(f"  Trials:  {cli.n_trials} (existing: {existing_trials})")
    ep_range = DATASET_EPOCH_RANGE[cli.dataset]
    if cli.n_epochs_min is not None:
        ep_range = (cli.n_epochs_min, ep_range[1])
    if cli.n_epochs is not None:
        ep_range = (ep_range[0], cli.n_epochs)
    print(f"  Epochs:  {ep_range[0]}~{ep_range[1]} (searched)")
    print(f"  Batch candidates: {BATCH_CANDIDATES[cli.dataset]}")
    print(f"  Python:  {PYTHON}")
    stage_label = getattr(cli, "stage_label", None)
    if stage_label:
        print(f"  Stage:   {stage_label}")
    print()


# ══════════════════════════════════════════════════════════════
# Warm-start: enqueue top trials from existing studies
# ══════════════════════════════════════════════════════════════

# Historical batch grids used by previous SIMSv2 studies. Used to translate a
# legacy trial's `batch_config` index back to its (bs, accum) tuple, then re-
# index against the current `BATCH_CANDIDATES[dataset]`.
_LEGACY_BATCH_GRIDS = {
    "simsv2": {
        "v1": [(8, 4), (16, 2), (16, 4), (32, 1), (32, 2), (64, 1)],
        "v2": [(16, 4), (32, 2), (64, 1), (32, 4)],
    },
    "mosi": {
        # Both s1_random and s2_local of v1 + msew35 used the same 6-element grid.
        "v1": [(8, 4), (8, 8), (16, 2), (16, 4), (32, 1), (32, 2)],
    },
}


def _legacy_batch_grid(dataset, source_study_name):
    grids = _LEGACY_BATCH_GRIDS.get(dataset, {})
    if not grids:
        return BATCH_CANDIDATES.get(dataset, [])
    name = (source_study_name or "").lower()
    # Match v2 only when it appears as a study version suffix (e.g. simsv2_v2, _v2_)
    # so that the dataset name "simsv2" itself does not trip the check.
    if f"{dataset}_v2" in name or "_v2_" in name or name.endswith("_v2"):
        return grids.get("v2", BATCH_CANDIDATES.get(dataset, []))
    return grids.get("v1", BATCH_CANDIDATES.get(dataset, []))


def _filter_param_for_current_space(key, value, dataset, source_study_name):
    """Return (kept, new_value). Drops a param if it lies outside current bounds.

    n_epochs is clipped instead of dropped so warm-start trials stay close to
    their original training budget.
    """
    if key in LOG_FLOAT_BOUNDS or key in LINEAR_FLOAT_BOUNDS:
        bounds = LOG_FLOAT_BOUNDS.get(key) or LINEAR_FLOAT_BOUNDS.get(key)
        lo, hi = bounds
        return (lo <= float(value) <= hi, float(value))
    if key in INT_BOUNDS:
        lo, hi = INT_BOUNDS[key]
        v = int(value)
        return (lo <= v <= hi, v)
    if key in CATEGORICAL_SPACE:
        choices = CATEGORICAL_SPACE[key]
        return (value in choices, value)
    if key == "n_epochs":
        lo, hi = DATASET_EPOCH_RANGE[dataset]
        v = int(value)
        clipped = max(lo, min(hi, v))
        return (True, clipped)
    if key == "batch_config":
        old_grid = _legacy_batch_grid(dataset, source_study_name)
        new_grid = BATCH_CANDIDATES.get(dataset, [])
        try:
            tup = old_grid[int(value)]
        except (IndexError, TypeError, ValueError):
            return (False, None)
        if tup in new_grid:
            return (True, new_grid.index(tup))
        return (False, None)
    return (False, None)


def enqueue_top_trials_into_study(study, dataset, source_db_uris, top_k):
    """Seed `study` with the top-k completed trials (by Optuna value, lower=better)
    pulled from each `source_db_uri`. Each enqueued trial only carries params that
    pass current search-space validation; missing params will be sampled fresh
    by the running sampler. Skips trials that lose more than half of their params
    to validation.
    """
    if not source_db_uris:
        return 0
    enqueued_total = 0
    print(f"\n[enqueue] dataset={dataset}, top_k_per_db={top_k}")
    for uri in source_db_uris:
        uri = uri.strip()
        if not uri:
            continue
        try:
            summaries = optuna.get_all_study_summaries(storage=uri)
        except Exception as e:
            print(f"[enqueue]   skip {uri}: cannot list studies ({e})")
            continue
        for summary in summaries:
            try:
                src = optuna.load_study(study_name=summary.study_name, storage=uri)
            except Exception as e:
                print(f"[enqueue]   {uri}::{summary.study_name}: load failed ({e})")
                continue
            done = [t for t in src.trials
                    if t.state == optuna.trial.TrialState.COMPLETE
                    and t.value is not None]
            if not done:
                continue
            done.sort(key=lambda t: t.value)  # lower MAE first
            picked = done[:top_k]
            print(f"[enqueue]   from {summary.study_name}: "
                  f"picking {len(picked)} of {len(done)} complete trials")
            for t in picked:
                filtered = {}
                dropped = []
                for k, v in t.params.items():
                    keep, new_v = _filter_param_for_current_space(
                        k, v, dataset, summary.study_name)
                    if keep:
                        filtered[k] = new_v
                    else:
                        dropped.append(k)
                if not t.params:
                    continue
                if len(filtered) < 0.5 * len(t.params):
                    print(f"[enqueue]     trial {t.number}: dropped too many params "
                          f"({len(dropped)}/{len(t.params)}); skip")
                    continue
                try:
                    study.enqueue_trial(filtered, skip_if_exists=True)
                    enqueued_total += 1
                    if dropped:
                        print(f"[enqueue]     trial {t.number}: queued "
                              f"(dropped {len(dropped)}: {dropped})")
                    else:
                        print(f"[enqueue]     trial {t.number}: queued (full match)")
                except Exception as e:
                    print(f"[enqueue]     trial {t.number}: enqueue failed ({e})")
    print(f"[enqueue] total enqueued: {enqueued_total}\n")
    return enqueued_total


def print_study_summary(study, cli):
    print("\n" + "=" * 60)
    print("Search complete!")
    if not cli.multi_objective:
        completed = get_completed_trials(study, cli.selection_metric)
        if not completed:
            print("  No completed trials were found.")
            return
        bt = study.best_trial
        print(f"  Best trial: #{bt.number}")
        print(f"  Best {cli.selection_metric}: {bt.value:.6f}")
        print("  Params:")
        for k, v in bt.params.items():
            print(f"    {k}: {v}")
    else:
        front = study.best_trials
        print(f"  Pareto front: {len(front)} trials")
        for t in sorted(front, key=lambda x: x.values[0], reverse=True):
            print(f"    #{t.number}: composite={t.values[0]:.4f} "
                  f"MAE={t.values[1]:.4f}")
        if front:
            best = max(front, key=lambda t: t.values[0])
            print(f"\n  Best by composite (#{best.number}):")
            for k, v in best.params.items():
                print(f"    {k}: {v}")


# ══════════════════════════════════════════════════════════════
# Checkpoint cleanup
# ══════════════════════════════════════════════════════════════

def pareto_dominates(a, b):
    return (a.values[0] >= b.values[0] and a.values[1] <= b.values[1] and
            (a.values[0] > b.values[0] or a.values[1] < b.values[1]))


def cleanup_checkpoints_multi(study, ckpt_base):
    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    if len(completed) <= KEEP_TOP_K:
        return
    front = [t for t in completed
             if not any(pareto_dominates(o, t) for o in completed if o != t)]
    ranked = sorted(front, key=lambda t: t.values[0], reverse=True)
    if len(ranked) < KEEP_TOP_K:
        rest = sorted([t for t in completed if t not in front],
                      key=lambda t: t.values[0], reverse=True)
        ranked.extend(rest)
    keep = {t.number for t in ranked[:KEEP_TOP_K]}
    for t in completed:
        if t.number not in keep:
            d = os.path.join(ckpt_base, f"trial_{t.number}")
            if os.path.isdir(d):
                shutil.rmtree(d, ignore_errors=True)


def cleanup_checkpoints_single(study, ckpt_base, higher_is_better):
    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    if len(completed) <= KEEP_TOP_K:
        return
    ranked = sorted(completed, key=lambda t: t.value,
                    reverse=higher_is_better)
    keep = {t.number for t in ranked[:KEEP_TOP_K]}
    for t in completed:
        if t.number not in keep:
            d = os.path.join(ckpt_base, f"trial_{t.number}")
            if os.path.isdir(d):
                shutil.rmtree(d, ignore_errors=True)


# ══════════════════════════════════════════════════════════════
# Objective
# ══════════════════════════════════════════════════════════════

def objective(trial, cli):
    params = build_search_params(trial, cli.search_tier, cli.dataset,
                                  getattr(cli, 'n_epochs', None),
                                  getattr(cli, 'n_epochs_min', None),
                                  local_space=getattr(cli, 'local_space', None))
    ds = cli.dataset

    stage_label = getattr(cli, "stage_label", None)
    log_dir = trial_train_log_dir(cli)
    ckpt_base = trial_ckpt_base(cli, ds, stage_label)
    if stage_label:
        log_path = os.path.join(log_dir, f"{ds}_{stage_label}_trial_{trial.number}.log")
    else:
        log_path = os.path.join(log_dir, f"{ds}_trial_{trial.number}.log")
    os.makedirs(log_dir, exist_ok=True)

    trial_ckpt = os.path.join(ckpt_base, f"trial_{trial.number}")
    os.makedirs(trial_ckpt, exist_ok=True)

    ib_hidden_dim = params["unified_dim"]

    cmd = [
        PYTHON, "-u", TRAIN_SCRIPT,
        "--dataset", ds,
        "--n_epochs", str(params["n_epochs"]),
        "--train_batch_size", str(params["train_batch_size"]),
        "--gradient_accumulation_step", str(params["gradient_accumulation_step"]),
        "--checkpoint_dir", trial_ckpt,
        "--selection_metric", cli.selection_metric,
        "--seed", str(params["seed"]),
        "--learning_rate", f"{params['learning_rate']:.6e}",
        "--ig_learning_rate", f"{params['ig_learning_rate']:.6e}",
        "--beta_ib", f"{params['beta_ib']:.4f}",
        "--num_infogate_layers", str(params["num_infogate_layers"]),
        "--bottleneck_dim", str(params["bottleneck_dim"]),
        "--mse_weight", f"{params['mse_weight']:.4f}",
        "--dropout_prob", f"{params['dropout_prob']:.4f}",
        "--alpha_ib", f"{params['alpha_ib']:.6f}",
        "--stage1_epochs", str(params["stage1_epochs"]),
        "--warmup_proportion", f"{params['warmup_proportion']:.4f}",
        "--weight_decay", f"{params['weight_decay']:.6f}",
        "--ema_decay", str(params["ema_decay"]),
        "--selector_target_temp", f"{params['selector_target_temp']:.4f}",
        "--selector_rib_weight", f"{params['selector_rib_weight']:.4f}",
        "--gumbel_tau_start", f"{params['gumbel_tau_start']:.4f}",
        "--gumbel_tau_end", f"{params['gumbel_tau_end']:.4f}",
        "--num_heads", str(params["num_heads"]),
        "--unified_dim", str(params["unified_dim"]),
        "--ib_hidden_dim", str(ib_hidden_dim),
        "--ema_start_epoch", str(params["ema_start_epoch"]),
    ]

    for attr, flag in (
        ("disable_l_lib", "--disable_l_lib"),
        ("disable_l_rib", "--disable_l_rib"),
    ):
        if getattr(cli, attr, False):
            cmd.append(flag)

    # ── banner ──
    print(f"\n{'='*60}")
    print(f"Trial {trial.number} [{ds}]  |  {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"  Tier {cli.search_tier} search")
    for k, v in sorted(params.items()):
        if k in DEFAULTS and params[k] == DEFAULTS[k] and cli.search_tier < 3:
            continue  # skip unchanged defaults
        fmt = f"{v:.4e}" if isinstance(v, float) else str(v)
        print(f"  {k}={fmt}")
    print(f"  log: {log_path}")
    print(f"{'='*60}")

    env = os.environ.copy()
    # If the launcher already set CUDA_VISIBLE_DEVICES (e.g. run_optuna_three_5090.sh
    # pins one physical GPU per process), do NOT overwrite — previously we always set
    # str(cli.gpu) (often 0), which forced every trial onto physical GPU 0.
    if "CUDA_VISIBLE_DEVICES" not in env:
        env["CUDA_VISIBLE_DEVICES"] = str(cli.gpu)

    with open(log_path, "w") as log_f:
        proc = subprocess.Popen(
            cmd, stdout=log_f, stderr=subprocess.STDOUT, env=env)

    s1_ep = int(params.get("stage1_epochs", DEFAULTS["stage1_epochs"]))
    last_reported_epoch = -1
    try:
        while proc.poll() is None:
            time.sleep(15)
            epoch, best_dev = parse_best_dev_metrics(
                log_path, dataset=ds, selection_metric=cli.selection_metric,
                stage1_epochs=s1_ep)

            if best_dev is not None and epoch > last_reported_epoch:
                sel_kw = dict(acc2=best_dev["Acc2"], mae=best_dev["MAE"],
                              corr=best_dev["Corr"], f1=best_dev["F1"])
                if "Acc7" in best_dev:
                    sel_kw["acc7"] = best_dev["Acc7"]
                if "Acc5" in best_dev:
                    sel_kw["acc5"] = best_dev["Acc5"]
                if "Acc3" in best_dev:
                    sel_kw["acc3"] = best_dev["Acc3"]
                selection_value = compute_selection_score(cli.selection_metric, **sel_kw)
                if not cli.multi_objective:
                    trial.report(selection_value, epoch)
                last_reported_epoch = epoch

            if PRUNE_FN[ds](epoch, best_dev, s1_ep):
                a2 = best_dev["Acc2"] if best_dev else 0
                m = best_dev["MAE"] if best_dev else 9
                s2 = max(0, epoch - s1_ep)
                print(f"  Trial {trial.number} PRUNED ep{epoch} (s2_ep={s2}, "
                      f"stage1_epochs={s1_ep}): Acc2={a2*100:.1f}% MAE={m:.4f}")
                proc.send_signal(signal.SIGTERM)
                proc.wait(timeout=10)
                raise optuna.TrialPruned()

        if proc.returncode != 0:
            print(f"  Trial {trial.number} FAILED (rc={proc.returncode})")
            raise optuna.TrialPruned()

    except Exception as e:
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        if isinstance(e, optuna.TrialPruned):
            raise
        print(f"  Trial {trial.number} error: {e}")
        raise optuna.TrialPruned()

    # ── parse final results ──
    results = parse_best_results(log_path)
    if "MAE" not in results:
        print(f"  Trial {trial.number}: no MAE in log")
        raise optuna.TrialPruned()

    mae = results["MAE"]
    acc2 = results.get("Acc-2", 0)
    corr = results.get("Corr", 0)
    f1 = results.get("F1", 0)
    sel_kw = dict(acc2=acc2, mae=mae, corr=corr, f1=f1)
    if ds == "simsv2":
        sel_kw["acc5"] = results.get("Acc-5", 0)
        sel_kw["acc3"] = results.get("Acc-3", 0)
    else:
        sel_kw["acc7"] = results.get("Acc-7", 0)
    composite = compute_selection_score("acc2_composite", **sel_kw)

    print(f"  Trial {trial.number} DONE: composite={composite:.4f} "
          f"Acc2={acc2*100:.2f}% MAE={mae:.4f} Corr={corr:.4f}")

    if not cli.multi_objective:
        return compute_selection_score(cli.selection_metric, **sel_kw)
    return composite, mae


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

def main():
    pa = argparse.ArgumentParser(description="Optuna v2 — MOSI/MOSEI/SIMSV2 search")
    pa.add_argument("--dataset", type=str, default="mosi",
                    choices=["mosi", "mosei", "simsv2"])
    pa.add_argument("--gpu", type=int, default=1,
                    help="Used only when CUDA_VISIBLE_DEVICES is unset: physical GPU id for train subprocess")
    pa.add_argument("--n_trials", type=int, default=200,
                    help="Total trials (should exceed n_startup_trials for a guided phase)")
    pa.add_argument("--n_epochs", type=int, default=None,
                    help="Override epoch range upper bound")
    pa.add_argument("--n_epochs_min", type=int, default=None,
                    help="Override epoch range lower bound")
    pa.add_argument("--search_tier", type=int, default=2, choices=[1, 2, 3])
    pa.add_argument("--study_name", type=str, default=None)
    pa.add_argument("--db", type=str, default=None)
    pa.add_argument(
        "--artefact_root",
        type=str,
        default=None,
        help="Per-run directory for train subprocess logs (train_logs/) and checkpoints "
             "(checkpoints/). If omitted and --db is sqlite under .../<run>/db/*.db, "
             "defaults to .../<run>/. Otherwise legacy layout logs/optuna/ + checkpoints/ is used.",
    )
    pa.add_argument("--selection_metric", type=str,
                    default=DEFAULT_SELECTION_METRIC,
                    choices=SELECTION_METRIC_CHOICES)
    pa.add_argument("--multi_objective", action="store_true",
                    help="Multi-obj BoTorch instead of single-obj TPE")
    pa.add_argument("--n_startup_trials", type=int, default=55,
                    help="Random warmup trials before guided search (TPE/MOTPE); "
                         "NSGA-II fallback uses this as population_size lower bound")
    pa.add_argument("--stage1_trials", type=int, default=60,
                    help="Stage-1 random trials for MOSI two-stage search.")
    pa.add_argument("--stage2_trials", type=int, default=140,
                    help="Stage-2 local TPE trials for MOSI two-stage search.")
    pa.add_argument("--stage2_top_k", type=int, default=8,
                    help="Top-k stage-1 trials used to build the stage-2 local space.")
    pa.add_argument("--disable_two_stage_mosi", action="store_true",
                    help="Disable MOSI-specific two-stage search and fall back to one study.")
    pa.add_argument("--disable_l_lib", action="store_true")
    pa.add_argument("--disable_l_rib", action="store_true")
    pa.add_argument("--stage_label", type=str, default=None,
                    help="Optional stage tag; partitions train_logs/ and checkpoints/ "
                         "so a new study does not overwrite earlier studies' artefacts.")
    pa.add_argument("--enqueue_top_from", type=str, default=None,
                    help="Comma-separated sqlite URIs whose TOP-K complete trials "
                         "(by Optuna value, lower=better) are enqueued as warm-start "
                         "seeds in the new study before sampling begins.")
    pa.add_argument("--enqueue_top_k", type=int, default=10,
                    help="Number of top trials per source DB to enqueue.")
    pa.add_argument("--no_dataset_overrides", action="store_true",
                    help="Skip apply_dataset_bounds_overrides(); use the FULL global "
                         "search space. Use for cold restarts that should NOT inherit "
                         "narrowing derived from prior-GPU runs.")
    cli = pa.parse_args()

    ds = cli.dataset
    if not cli.no_dataset_overrides:
        apply_dataset_bounds_overrides(ds)
    else:
        print(f"[bounds] --no_dataset_overrides set; using full global search space "
              f"for {ds}.")
    if cli.study_name is None:
        if not cli.multi_objective:
            cli.study_name = f"infogate_{ds}_v6_tpe_mae"
        else:
            cli.study_name = f"infogate_{ds}_v5_botorch"

    if cli.db is None:
        log_dir = os.path.join(SCRIPT_DIR, "logs", "optuna")
        os.makedirs(log_dir, exist_ok=True)
        db_path = os.path.join(log_dir, f"{cli.study_name}.db")
        cli.db = f"sqlite:///{db_path}"

    cli.artefact_root = resolve_artefact_root(cli)
    if cli.artefact_root:
        os.makedirs(os.path.join(cli.artefact_root, "train_logs"), exist_ok=True)
        os.makedirs(os.path.join(cli.artefact_root, "checkpoints"), exist_ok=True)
    else:
        log_dir = os.path.join(SCRIPT_DIR, "logs", "optuna")
        ckpt_base = os.path.join(SCRIPT_DIR, "checkpoints", f"optuna_{ds}")
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(ckpt_base, exist_ok=True)

    use_two_stage_mosi = (
        ds == "mosi"
        and not cli.multi_objective
        and not cli.disable_two_stage_mosi
    )

    if use_two_stage_mosi:
        s1_name, s2_name = build_two_stage_study_names(cli)
        stage1_cli = clone_cli(
            cli,
            study_name=s1_name,
            db=build_stage_db_uri(s1_name, cli.db),
            n_trials=cli.stage1_trials,
            sampler_name="random",
            sampler_seed=128,
            stage_label="s1_random",
            local_space=None,
        )
        stage1_study, stage1_mode = create_study_for_cli(stage1_cli)
        print_study_header(stage1_cli, stage1_mode, len(stage1_study.trials))
        optimize_with_cleanup(
            stage1_study,
            stage1_cli,
            trial_ckpt_base(stage1_cli, "mosi", "s1_random"),
        )
        print_study_summary(stage1_study, stage1_cli)

        local_space, top_trials = build_local_search_space(
            stage1_study, ds, cli.search_tier, cli.selection_metric, cli.stage2_top_k)
        top_ids = [trial.number for trial in top_trials]
        print("\n" + "=" * 60)
        print("Preparing MOSI stage 2 local search")
        print(f"  Anchors from stage 1 top-{len(top_ids)} trials: {top_ids}")
        print(summarize_local_space(local_space), end="")

        stage2_cli = clone_cli(
            cli,
            study_name=s2_name,
            db=build_stage_db_uri(s2_name, cli.db),
            n_trials=cli.stage2_trials,
            sampler_name="tpe",
            sampler_seed=256,
            stage_label="s2_local",
            local_space=local_space,
        )
        stage2_study, stage2_mode = create_study_for_cli(stage2_cli)
        print_study_header(stage2_cli, stage2_mode, len(stage2_study.trials))
        optimize_with_cleanup(
            stage2_study,
            stage2_cli,
            trial_ckpt_base(stage2_cli, "mosi", "s2_local"),
        )
        print_study_summary(stage2_study, stage2_cli)
        return

    cli = clone_cli(cli, sampler_name="tpe", sampler_seed=128,
                    stage_label=cli.stage_label, local_space=None)
    study, mode_label = create_study_for_cli(cli)
    if getattr(cli, "enqueue_top_from", None):
        enqueue_top_trials_into_study(
            study, cli.dataset,
            [u.strip() for u in cli.enqueue_top_from.split(",") if u.strip()],
            cli.enqueue_top_k,
        )
    print_study_header(cli, mode_label, len(study.trials))
    optimize_with_cleanup(study, cli, trial_ckpt_base(cli, ds, cli.stage_label))
    print_study_summary(study, cli)


if __name__ == "__main__":
    main()
