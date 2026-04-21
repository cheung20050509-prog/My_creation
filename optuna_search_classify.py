"""Optuna search for InfoGate binary classification (MUSTARD / UR-FUNNY).

Independent from `optuna_search_v2.py` (which targets MOSI/MOSEI/SIMSV2 +
`train.py`). This driver:

  - Drives `train_classify.py` (BCE + sigmoid) instead of `train.py`.
  - Selects on `binary_acc` (higher better); F1 used as secondary tiebreak.
  - Drops regression-only knobs (`mse_weight`, `gumbel_tau_*`).
  - MUSTARD: two-stage Random -> TPE-local (small dev set, cheap trials).
  - UR-FUNNY: single-stage TPE (expensive trials, ~1-1.5h each).

Artefact layout matches `optuna_search_v2.py`:
  logs/<RUN_TAG>/{db,train_logs,checkpoints} when --db sqlite lives under
  <root>/db/*.db (auto-detected) or pass --artefact_root explicitly.
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

from selection_utils import (
    SELECTION_METRIC_CHOICES,
    compute_selection_score,
    selection_higher_is_better,
)


# ── paths ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_SCRIPT = os.path.join(SCRIPT_DIR, "train_classify.py")
PYTHON = sys.executable
KEEP_TOP_K = 5

# ── per-dataset constants ──
SUPPORTED_DATASETS = ("mustard", "ur_funny")

MAX_SEQ_LEN = {
    "mustard": 77,   # HKT default for sarcasm
    "ur_funny": 64,  # HKT default for humor
}

# (train_batch_size, gradient_accumulation_step) candidates per dataset.
# Effective batch = bs * accum.
BATCH_CANDIDATES = {
    "mustard":  [(8, 2), (8, 4), (16, 1), (16, 2), (32, 1)],
    "ur_funny": [(8, 4), (16, 2), (16, 4), (32, 1), (32, 2)],
}

DATASET_EPOCH_RANGE = {
    "mustard":  (30, 80),
    "ur_funny": (20, 50),
}


# ── search space bounds (Tier-2 superset; classification subset of v2) ──
LOG_FLOAT_BOUNDS = {
    "learning_rate":    (5e-6, 5e-5),
    "ig_learning_rate": (5e-5, 2e-3),
    "beta_ib":          (4.0, 64.0),
    "alpha_ib":         (1e-3, 5e-2),
    "weight_decay":     (1e-4, 0.1),
}

LINEAR_FLOAT_BOUNDS = {
    "dropout_prob":         (0.05, 0.40),
    "warmup_proportion":    (0.02, 0.25),
    "selector_target_temp": (0.30, 0.90),
    "selector_rib_weight":  (0.01, 0.15),
}

INT_BOUNDS = {
    "stage1_epochs": (3, 15),
}

CATEGORICAL_SPACE = {
    "num_infogate_layers": [2, 3, 4, 5],
    "bottleneck_dim":      [64, 96, 128, 192],
    "ema_decay":           [0.99, 0.995, 0.999, 0.9995],
    "num_heads":           [2, 4, 8],
    "unified_dim":         [128, 256, 384],
    "ema_start_epoch":     [3, 5, 8, 10],
}

DEFAULTS = {
    "seed": 42,
    "learning_rate": 2e-5,
    "ig_learning_rate": 5e-4,
    "beta_ib": 16.0,
    "alpha_ib": 0.005,
    "num_infogate_layers": 3,
    "bottleneck_dim": 128,
    "dropout_prob": 0.25,
    "stage1_epochs": 8,
    "warmup_proportion": 0.1,
    "weight_decay": 0.01,
    "ema_decay": 0.999,
    "ema_start_epoch": 5,
    "selector_target_temp": 0.6,
    "selector_rib_weight": 0.05,
    "num_heads": 4,
    "unified_dim": 256,
}


# ══════════════════════════════════════════════════════════════
# Log parsing — matches train_classify.py prints (lines 445-450, 497-501)
# ══════════════════════════════════════════════════════════════

EPOCH_LINE_RE = re.compile(r"^Epoch (\d+)/\d+")
DEV_LINE_BIN_RE = re.compile(
    r"\s+Dev\s+Acc=([\d.]+)%\s+F1=([\d.]+)%\s+BCE-prob-MAE=([\d.]+)"
)
SELECT_LINE_RE = re.compile(r"\s+Select\s+\S+=([\d.]+)")
BEST_FIELD_RE = re.compile(
    r"^\s+(Selection score|Acc|F1):\s+([\d.]+)%?\s*$"
)


def parse_best_dev_metrics(log_path, selection_metric="binary_acc"):
    """Scan trial log; return (latest_completed_epoch, best_dev_dict_or_None).

    best_dev_dict = {"Acc2": float [0,1], "F1": float [0,1],
                     "MAE": float (BCE-prob-MAE placeholder)}.

    Selection follows the metric semantics in selection_utils:
      binary_acc -> max Acc2; binary_f1 -> max F1; mae-style -> min MAE.
    """
    if not os.path.exists(log_path):
        return 0, None

    higher_is_better = selection_higher_is_better(selection_metric)
    current_epoch = 0
    best_score = None
    best = None

    with open(log_path, "r") as f:
        for line in f:
            em = EPOCH_LINE_RE.match(line)
            if em:
                current_epoch = int(em.group(1))
                continue
            dm = DEV_LINE_BIN_RE.match(line)
            if not dm:
                continue
            acc, f1, mae = (float(x) for x in dm.groups())
            acc /= 100.0
            f1 /= 100.0
            score = compute_selection_score(
                selection_metric, acc2=acc, mae=mae, corr=0.0, f1=f1)
            if better_than(score, best_score, higher_is_better):
                best_score = score
                best = {"Acc2": acc, "F1": f1, "MAE": mae}
    return current_epoch, best


def parse_best_results(log_path):
    """Parse the 'Best Results (...)' block printed at end of training.

    Stops at the first 'Last Epoch' line. Returns {"Selection score","Acc","F1"}
    with Acc/F1 normalised to [0,1].
    """
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
            if not in_block:
                continue
            m = BEST_FIELD_RE.match(line)
            if not m:
                continue
            key, raw = m.groups()
            val = float(raw)
            if key in ("Acc", "F1"):
                val /= 100.0
            if key == "Selection score":
                key = "SelectionScore"
            results[key] = val
    return results


# ══════════════════════════════════════════════════════════════
# Per-dataset pruning — only on binary_acc (no MAE in classification)
# ══════════════════════════════════════════════════════════════

def should_prune_mustard(epoch, metrics):
    if metrics is None:
        return False
    a = metrics["Acc2"]
    if epoch >= 35:
        return a < 0.62
    if epoch >= 20:
        return a < 0.55
    return False


def should_prune_ur_funny(epoch, metrics):
    if metrics is None:
        return False
    a = metrics["Acc2"]
    if epoch >= 35:
        return a < 0.65
    if epoch >= 20:
        return a < 0.62
    if epoch >= 10:
        return a < 0.58
    return False


PRUNE_FN = {
    "mustard":  should_prune_mustard,
    "ur_funny": should_prune_ur_funny,
}


# ══════════════════════════════════════════════════════════════
# Search-space samplers (mirror v2 tiered design, classification subset)
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


def suggest_tier1(trial, dataset, n_epochs_max=None, n_epochs_min=None,
                  local_space=None):
    candidates = BATCH_CANDIDATES[dataset]
    batch_idx = suggest_categorical_param(
        trial, "batch_config", list(range(len(candidates))), local_space)
    bs, accum = candidates[batch_idx]
    ep_lo, ep_hi = DATASET_EPOCH_RANGE[dataset]
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
        "seed": DEFAULTS["seed"],
        "learning_rate": suggest_float_param(
            trial, "learning_rate", LOG_FLOAT_BOUNDS["learning_rate"], log=True,
            local_space=local_space),
        "ig_learning_rate": suggest_float_param(
            trial, "ig_learning_rate", LOG_FLOAT_BOUNDS["ig_learning_rate"],
            log=True, local_space=local_space),
        "beta_ib": suggest_float_param(
            trial, "beta_ib", LOG_FLOAT_BOUNDS["beta_ib"], log=True,
            local_space=local_space),
        "num_infogate_layers": suggest_categorical_param(
            trial, "num_infogate_layers",
            CATEGORICAL_SPACE["num_infogate_layers"], local_space),
        "bottleneck_dim": suggest_categorical_param(
            trial, "bottleneck_dim", CATEGORICAL_SPACE["bottleneck_dim"],
            local_space),
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
        "selector_target_temp": suggest_float_param(
            trial, "selector_target_temp",
            LINEAR_FLOAT_BOUNDS["selector_target_temp"],
            local_space=local_space),
        "selector_rib_weight": suggest_float_param(
            trial, "selector_rib_weight",
            LINEAR_FLOAT_BOUNDS["selector_rib_weight"],
            local_space=local_space),
    }


def suggest_tier3(trial, local_space=None):
    return {
        "num_heads": suggest_categorical_param(
            trial, "num_heads", CATEGORICAL_SPACE["num_heads"], local_space),
        "unified_dim": suggest_categorical_param(
            trial, "unified_dim", CATEGORICAL_SPACE["unified_dim"],
            local_space),
        "ema_start_epoch": suggest_categorical_param(
            trial, "ema_start_epoch", CATEGORICAL_SPACE["ema_start_epoch"],
            local_space),
    }


def build_search_params(trial, tier, dataset, n_epochs_max=None,
                        n_epochs_min=None, local_space=None):
    params = dict(DEFAULTS)
    params.update(suggest_tier1(
        trial, dataset, n_epochs_max, n_epochs_min, local_space=local_space))
    if tier >= 2:
        params.update(suggest_tier2(trial, local_space=local_space))
    if tier >= 3:
        params.update(suggest_tier3(trial, local_space=local_space))
    return params


# ══════════════════════════════════════════════════════════════
# Helpers (paths, narrowing, cleanup) — ported from v2 then trimmed
# ══════════════════════════════════════════════════════════════

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


def sqlite_uri_to_path(db_uri):
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


def infer_artefact_root_from_db_uri(db_uri):
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
    return os.path.join(SCRIPT_DIR, "logs", "optuna_classify")


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
    subset = [c for c in original_choices if c in wanted]
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
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
    ]
    hib = selection_higher_is_better(selection_metric)
    return sorted(completed, key=lambda t: t.value, reverse=hib)


def build_local_search_space(study, dataset, search_tier, selection_metric, top_k):
    """Mirror v2.build_local_search_space, restricted to classification knobs."""
    ranked = get_completed_trials(study, selection_metric)
    top_trials = ranked[:top_k]
    if not top_trials:
        return None, []

    best_params = top_trials[0].params
    params_list = [t.params for t in top_trials]
    local_space = {}

    def values_for(name):
        return [p[name] for p in params_list if name in p]

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
    tier1_linear_names = ("dropout_prob",)
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
        tier2_linear_names = ("warmup_proportion", "selector_target_temp",
                              "selector_rib_weight")
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
        tier3_cat_names = ("num_heads", "unified_dim", "ema_start_epoch")
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
        db_path = os.path.join(
            SCRIPT_DIR, "logs", "optuna_classify", f"{stage_study_name}.db")
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
    hib = selection_higher_is_better(cli.selection_metric)
    sampler_name = getattr(cli, "sampler_name", "tpe")
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
    return study, mode_label


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


def optimize_with_cleanup(study, cli, ckpt_base):
    def after_trial(study, trial):
        if trial.state == optuna.trial.TrialState.PRUNED:
            d = os.path.join(ckpt_base, f"trial_{trial.number}")
            if os.path.isdir(d):
                shutil.rmtree(d, ignore_errors=True)
        elif trial.state == optuna.trial.TrialState.COMPLETE:
            cleanup_checkpoints_single(
                study, ckpt_base,
                selection_higher_is_better(cli.selection_metric))

    study.optimize(
        lambda trial: objective(trial, cli),
        n_trials=cli.n_trials,
        callbacks=[after_trial],
    )


def print_study_header(cli, mode_label, existing_trials):
    print(f"Optuna classify — {cli.dataset.upper()}")
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
    print(f"  Max seq len: {MAX_SEQ_LEN[cli.dataset]}")
    print(f"  Python:  {PYTHON}")
    stage_label = getattr(cli, "stage_label", None)
    if stage_label:
        print(f"  Stage:   {stage_label}")
    print()


def print_study_summary(study, cli):
    print("\n" + "=" * 60)
    print("Search complete!")
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


# ══════════════════════════════════════════════════════════════
# Objective
# ══════════════════════════════════════════════════════════════

def objective(trial, cli):
    params = build_search_params(
        trial, cli.search_tier, cli.dataset,
        getattr(cli, "n_epochs", None),
        getattr(cli, "n_epochs_min", None),
        local_space=getattr(cli, "local_space", None),
    )
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
        "--max_seq_length", str(MAX_SEQ_LEN[ds]),
        "--n_epochs", str(params["n_epochs"]),
        "--stage1_epochs", str(params["stage1_epochs"]),
        "--train_batch_size", str(params["train_batch_size"]),
        "--gradient_accumulation_step", str(params["gradient_accumulation_step"]),
        "--learning_rate", f"{params['learning_rate']:.6e}",
        "--ig_learning_rate", f"{params['ig_learning_rate']:.6e}",
        "--beta_ib", f"{params['beta_ib']:.4f}",
        "--alpha_ib", f"{params['alpha_ib']:.6f}",
        "--bottleneck_dim", str(params["bottleneck_dim"]),
        "--num_infogate_layers", str(params["num_infogate_layers"]),
        "--unified_dim", str(params["unified_dim"]),
        "--ib_hidden_dim", str(ib_hidden_dim),
        "--num_heads", str(params["num_heads"]),
        "--dropout_prob", f"{params['dropout_prob']:.4f}",
        "--warmup_proportion", f"{params['warmup_proportion']:.4f}",
        "--weight_decay", f"{params['weight_decay']:.6f}",
        "--ema_decay", str(params["ema_decay"]),
        "--ema_start_epoch", str(params["ema_start_epoch"]),
        "--selector_target_temp", f"{params['selector_target_temp']:.4f}",
        "--selector_rib_weight", f"{params['selector_rib_weight']:.4f}",
        "--selection_metric", cli.selection_metric,
        "--checkpoint_dir", trial_ckpt,
        "--seed", str(params["seed"]),
    ]

    for attr, flag in (
        ("disable_l_lib", "--disable_l_lib"),
        ("disable_l_rib", "--disable_l_rib"),
    ):
        if getattr(cli, attr, False):
            cmd.append(flag)

    print(f"\n{'='*60}")
    print(f"Trial {trial.number} [{ds}]  |  {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"  Tier {cli.search_tier} search")
    for k, v in sorted(params.items()):
        if k in DEFAULTS and params[k] == DEFAULTS[k] and cli.search_tier < 3:
            continue
        fmt = f"{v:.4e}" if isinstance(v, float) else str(v)
        print(f"  {k}={fmt}")
    print(f"  log: {log_path}")
    print(f"{'='*60}")

    env = os.environ.copy()
    if "CUDA_VISIBLE_DEVICES" not in env:
        env["CUDA_VISIBLE_DEVICES"] = str(cli.gpu)

    with open(log_path, "w") as log_f:
        proc = subprocess.Popen(
            cmd, stdout=log_f, stderr=subprocess.STDOUT, env=env)

    try:
        while proc.poll() is None:
            time.sleep(15)
            epoch, best_dev = parse_best_dev_metrics(
                log_path, selection_metric=cli.selection_metric)

            if best_dev is not None:
                selection_value = compute_selection_score(
                    cli.selection_metric,
                    acc2=best_dev["Acc2"], mae=best_dev["MAE"],
                    corr=0.0, f1=best_dev["F1"],
                )
                trial.report(selection_value, epoch)

            if PRUNE_FN[ds](epoch, best_dev):
                a = best_dev["Acc2"] if best_dev else 0.0
                f1 = best_dev["F1"] if best_dev else 0.0
                print(f"  Trial {trial.number} PRUNED ep{epoch}: "
                      f"Acc={a*100:.2f}% F1={f1*100:.2f}%")
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

    results = parse_best_results(log_path)
    if "Acc" not in results:
        print(f"  Trial {trial.number}: no Acc in best-results block")
        raise optuna.TrialPruned()

    acc = results["Acc"]
    f1 = results.get("F1", 0.0)
    selection_value = compute_selection_score(
        cli.selection_metric, acc2=acc, mae=0.0, corr=0.0, f1=f1)

    print(f"  Trial {trial.number} DONE: "
          f"{cli.selection_metric}={selection_value:.4f} "
          f"Acc={acc*100:.2f}% F1={f1*100:.2f}%")
    return selection_value


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

def main():
    pa = argparse.ArgumentParser(
        description="Optuna search for InfoGate binary classification "
                    "(MUSTARD / UR-FUNNY)")
    pa.add_argument("--dataset", type=str, default="mustard",
                    choices=list(SUPPORTED_DATASETS))
    pa.add_argument("--gpu", type=int, default=0,
                    help="Used only when CUDA_VISIBLE_DEVICES is unset: "
                         "physical GPU id for train subprocess")
    pa.add_argument("--n_trials", type=int, default=80,
                    help="Total trials for single-stage runs (UR-FUNNY default; "
                         "MUSTARD uses --stage1_trials/--stage2_trials)")
    pa.add_argument("--n_epochs", type=int, default=None,
                    help="Override epoch range upper bound")
    pa.add_argument("--n_epochs_min", type=int, default=None,
                    help="Override epoch range lower bound")
    pa.add_argument("--search_tier", type=int, default=2, choices=[1, 2, 3])
    pa.add_argument("--study_name", type=str, default=None)
    pa.add_argument("--db", type=str, default=None)
    pa.add_argument(
        "--artefact_root", type=str, default=None,
        help="Per-run directory for train subprocess logs (train_logs/) and "
             "checkpoints (checkpoints/). If omitted and --db is sqlite under "
             ".../<run>/db/*.db, defaults to .../<run>/. Otherwise legacy "
             "layout logs/optuna_classify/ + checkpoints/ is used.")
    pa.add_argument("--selection_metric", type=str,
                    default="binary_acc",
                    choices=SELECTION_METRIC_CHOICES)
    pa.add_argument("--n_startup_trials", type=int, default=20,
                    help="Random warmup trials before guided TPE search")
    pa.add_argument("--stage1_trials", type=int, default=40,
                    help="Stage-1 random trials for MUSTARD two-stage search")
    pa.add_argument("--stage2_trials", type=int, default=80,
                    help="Stage-2 local TPE trials for MUSTARD two-stage search")
    pa.add_argument("--stage2_top_k", type=int, default=8,
                    help="Top-k stage-1 trials used to build stage-2 local space")
    pa.add_argument("--disable_two_stage", action="store_true",
                    help="Disable MUSTARD two-stage search and run a single TPE study")
    pa.add_argument("--disable_l_lib", action="store_true")
    pa.add_argument("--disable_l_rib", action="store_true")
    pa.add_argument("--stage_label", type=str, default=None,
                    help="Optional stage tag; partitions train_logs/ and "
                         "checkpoints/ so a new study does not overwrite earlier "
                         "studies' artefacts.")
    cli = pa.parse_args()

    ds = cli.dataset
    if cli.study_name is None:
        suffix = sanitize_name(cli.selection_metric)
        cli.study_name = f"infogate_{ds}_classify_{suffix}"

    if cli.db is None:
        log_dir = os.path.join(SCRIPT_DIR, "logs", "optuna_classify")
        os.makedirs(log_dir, exist_ok=True)
        db_path = os.path.join(log_dir, f"{cli.study_name}.db")
        cli.db = f"sqlite:///{db_path}"

    cli.artefact_root = resolve_artefact_root(cli)
    db_path = sqlite_uri_to_path(cli.db)
    if db_path:
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
    if cli.artefact_root:
        os.makedirs(os.path.join(cli.artefact_root, "train_logs"), exist_ok=True)
        os.makedirs(os.path.join(cli.artefact_root, "checkpoints"), exist_ok=True)
    else:
        log_dir = os.path.join(SCRIPT_DIR, "logs", "optuna_classify")
        ckpt_base = os.path.join(SCRIPT_DIR, "checkpoints", f"optuna_{ds}")
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(ckpt_base, exist_ok=True)

    use_two_stage = (ds == "mustard") and not cli.disable_two_stage

    if use_two_stage:
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
            stage1_study, stage1_cli,
            trial_ckpt_base(stage1_cli, ds, "s1_random"),
        )
        print_study_summary(stage1_study, stage1_cli)

        local_space, top_trials = build_local_search_space(
            stage1_study, ds, cli.search_tier, cli.selection_metric,
            cli.stage2_top_k)
        top_ids = [t.number for t in top_trials]
        print("\n" + "=" * 60)
        print(f"Preparing {ds.upper()} stage 2 local search")
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
            stage2_study, stage2_cli,
            trial_ckpt_base(stage2_cli, ds, "s2_local"),
        )
        print_study_summary(stage2_study, stage2_cli)
        return

    cli = clone_cli(cli, sampler_name="tpe", sampler_seed=128,
                    stage_label=cli.stage_label, local_space=None)
    study, mode_label = create_study_for_cli(cli)
    print_study_header(cli, mode_label, len(study.trials))
    optimize_with_cleanup(study, cli, trial_ckpt_base(cli, ds, cli.stage_label))
    print_study_summary(study, cli)


if __name__ == "__main__":
    main()
