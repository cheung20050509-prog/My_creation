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
from optuna.samplers import NSGAIISampler, TPESampler
from optuna.integration.botorch import BoTorchSampler

try:
    from optuna.samplers import MOTPESampler
except ImportError:  # Optuna < 3.0
    MOTPESampler = None

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


# ══════════════════════════════════════════════════════════════
# Search space (3 tiers)
# ══════════════════════════════════════════════════════════════

def suggest_tier1(trial, dataset="mosi", n_epochs_max=None, n_epochs_min=None):
    candidates = BATCH_CANDIDATES[dataset]
    batch_idx = trial.suggest_categorical(
        "batch_config", list(range(len(candidates))))
    bs, accum = candidates[batch_idx]
    ep_lo, ep_hi = DATASET_EPOCH_RANGE[dataset]
    if n_epochs_min is not None:
        ep_lo = n_epochs_min
    if n_epochs_max is not None:
        ep_hi = n_epochs_max
    return {
        "train_batch_size": bs,
        "gradient_accumulation_step": accum,
        "n_epochs": trial.suggest_int("n_epochs", ep_lo, ep_hi),
        "seed": 128,  # Fixed seed
        "learning_rate": trial.suggest_float("learning_rate", 5e-6, 5e-5, log=True),
        "ig_learning_rate": trial.suggest_float("ig_learning_rate", 5e-5, 2e-3, log=True),
        "beta_ib": trial.suggest_float("beta_ib", 4.0, 64.0, log=True),
        "num_infogate_layers": trial.suggest_categorical("num_infogate_layers", [2, 3, 4, 5]),
        "bottleneck_dim": trial.suggest_categorical("bottleneck_dim", [64, 96, 128, 192]),
        "mse_weight": trial.suggest_float("mse_weight", 0.0, 2.0),
        "dropout_prob": trial.suggest_float("dropout_prob", 0.05, 0.4),
    }


def suggest_tier2(trial):
    return {
        "alpha_ib": trial.suggest_float("alpha_ib", 0.001, 0.05, log=True),
        "stage1_epochs": trial.suggest_int("stage1_epochs", 3, 20),
        "warmup_proportion": trial.suggest_float("warmup_proportion", 0.02, 0.25),
        "weight_decay": trial.suggest_float("weight_decay", 1e-4, 0.1, log=True),
        "ema_decay": trial.suggest_categorical("ema_decay", [0.99, 0.995, 0.999, 0.9995]),
    }


def suggest_tier3(trial):
    return {
        "selector_target_temp": trial.suggest_float("selector_target_temp", 0.1, 1.0),
        "selector_rib_weight": trial.suggest_float("selector_rib_weight", 0.01, 0.2),
        "gumbel_tau_start": trial.suggest_float("gumbel_tau_start", 0.5, 2.0),
        "gumbel_tau_end": trial.suggest_float("gumbel_tau_end", 0.1, 1.0),
        "num_heads": trial.suggest_categorical("num_heads", [2, 4, 8]),
        "unified_dim": trial.suggest_categorical("unified_dim", [128, 256, 384]),
        "ema_start_epoch": trial.suggest_categorical("ema_start_epoch", [3, 5, 8, 10]),
    }


def build_search_params(trial, tier, dataset="mosi", n_epochs_max=None, n_epochs_min=None):
    params = dict(DEFAULTS)
    params.update(suggest_tier1(trial, dataset, n_epochs_max, n_epochs_min))
    if tier >= 2:
        params.update(suggest_tier2(trial))
    if tier >= 3:
        params.update(suggest_tier3(trial))
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


def parse_best_dev_metrics(log_path, dataset="mosi"):
    if not os.path.exists(log_path):
        return 0, None
    current_epoch = 0
    best = None
    best_composite = -1e9
    with open(log_path, "r") as f:
        for line in f:
            em = EPOCH_LINE_RE.match(line)
            if em:
                current_epoch = int(em.group(1))
                continue
            # Try simsv2 format first (more specific), then fallback
            dm_s = DEV_LINE_SIMSV2_RE.match(line)
            dm = DEV_LINE_RE.match(line) if dm_s is None else None
            if dm_s:
                acc2, acc5, acc3, mae, corr, f1 = (float(x) for x in dm_s.groups())
                acc2 /= 100.0
                acc5 /= 100.0
                acc3 /= 100.0
                composite = compute_selection_score(
                    "acc2_composite", acc2=acc2, mae=mae, corr=corr, f1=f1,
                    acc5=acc5, acc3=acc3)
                if composite > best_composite:
                    best_composite = composite
                    best = {"Acc2": acc2, "Acc5": acc5, "Acc3": acc3,
                            "MAE": mae, "Corr": corr, "F1": f1}
            elif dm:
                acc2, acc7, mae, corr, f1 = (float(x) for x in dm.groups())
                acc2 /= 100.0
                acc7 /= 100.0
                composite = compute_selection_score(
                    "acc2_composite", acc2=acc2, mae=mae, corr=corr, f1=f1,
                    acc7=acc7)
                if composite > best_composite:
                    best_composite = composite
                    best = {"Acc2": acc2, "Acc7": acc7, "MAE": mae,
                            "Corr": corr, "F1": f1}
    return current_epoch, best


# ══════════════════════════════════════════════════════════════
# Per-dataset pruning (OR logic: prune only if BOTH fail)
# ══════════════════════════════════════════════════════════════

def should_prune_mosi(epoch, metrics):
    if metrics is None:
        return False
    acc2, mae = metrics["Acc2"], metrics["MAE"]
    if epoch >= 50:
        return acc2 < 0.84 and mae > 0.66
    if epoch >= 30:
        return acc2 < 0.80 and mae > 0.70
    if epoch >= 15:
        return acc2 < 0.72 and mae > 0.85
    return False


def should_prune_mosei(epoch, metrics):
    if metrics is None:
        return False
    acc2, mae = metrics["Acc2"], metrics["MAE"]
    if epoch >= 60:
        return acc2 < 0.76 and mae > 0.70
    if epoch >= 40:
        return acc2 < 0.70 and mae > 0.80
    if epoch >= 25:
        return acc2 < 0.55 and mae > 0.95
    return False


def should_prune_simsv2(epoch, metrics):
    if metrics is None:
        return False
    acc2, mae = metrics["Acc2"], metrics["MAE"]
    if epoch >= 40:
        return acc2 < 0.78 and mae > 0.48
    if epoch >= 20:
        return acc2 < 0.74 and mae > 0.52
    if epoch >= 10:
        return acc2 < 0.68 and mae > 0.60
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
    "mosi":   (60, 150),
    "mosei":  (60, 150),
    "simsv2": (60, 150),
}


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
                                  getattr(cli, 'n_epochs_min', None))
    ds = cli.dataset

    log_dir = os.path.join(SCRIPT_DIR, "logs", "optuna")
    ckpt_base = os.path.join(SCRIPT_DIR, "checkpoints", f"optuna_{ds}")
    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir, f"{ds}_trial_{trial.number}.log")
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

    try:
        while proc.poll() is None:
            time.sleep(15)
            epoch, best_dev = parse_best_dev_metrics(log_path, dataset=ds)

            if best_dev is not None:
                sel_kw = dict(acc2=best_dev["Acc2"], mae=best_dev["MAE"],
                              corr=best_dev["Corr"], f1=best_dev["F1"])
                if "Acc7" in best_dev:
                    sel_kw["acc7"] = best_dev["Acc7"]
                if "Acc5" in best_dev:
                    sel_kw["acc5"] = best_dev["Acc5"]
                if "Acc3" in best_dev:
                    sel_kw["acc3"] = best_dev["Acc3"]
                composite = compute_selection_score("acc2_composite", **sel_kw)
                if cli.single_objective:
                    trial.report(composite, epoch)

            if PRUNE_FN[ds](epoch, best_dev):
                a2 = best_dev["Acc2"] if best_dev else 0
                m = best_dev["MAE"] if best_dev else 9
                print(f"  Trial {trial.number} PRUNED ep{epoch}: "
                      f"Acc2={a2*100:.1f}% MAE={m:.4f}")
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

    if cli.single_objective:
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
    pa.add_argument("--selection_metric", type=str,
                    default=DEFAULT_SELECTION_METRIC,
                    choices=SELECTION_METRIC_CHOICES)
    pa.add_argument("--single_objective", action="store_true",
                    help="Single-obj TPE instead of multi-obj NSGA-II")
    pa.add_argument("--n_startup_trials", type=int, default=55,
                    help="Random warmup trials before guided search (TPE/MOTPE); "
                         "NSGA-II fallback uses this as population_size lower bound")
    pa.add_argument("--disable_l_lib", action="store_true")
    pa.add_argument("--disable_l_rib", action="store_true")
    cli = pa.parse_args()

    ds = cli.dataset
    if cli.study_name is None:
        cli.study_name = f"infogate_{ds}_v5_botorch"

    log_dir = os.path.join(SCRIPT_DIR, "logs", "optuna")
    ckpt_base = os.path.join(SCRIPT_DIR, "checkpoints", f"optuna_{ds}")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_base, exist_ok=True)

    if cli.db is None:
        db_path = os.path.join(log_dir, f"{cli.study_name}.db")
        cli.db = f"sqlite:///{db_path}"

    n_startup = cli.n_startup_trials
    if cli.single_objective:
        hib = selection_higher_is_better(cli.selection_metric)
        study = optuna.create_study(
            study_name=cli.study_name,
            storage=cli.db,
            direction="maximize" if hib else "minimize",
            sampler=TPESampler(n_startup_trials=n_startup),
            load_if_exists=True,
        )
        mo_sampler_name = "TPE"
    else:
        mo_sampler = BoTorchSampler(n_startup_trials=n_startup)
        mo_sampler_name = f"BoTorch(n_startup_trials={n_startup})"
        study = optuna.create_study(
            study_name=cli.study_name,
            storage=cli.db,
            directions=["maximize", "minimize"],
            sampler=mo_sampler,
            load_if_exists=True,
        )

    print(f"Optuna v2 — {ds.upper()}")
    print(f"  Study:   {cli.study_name}")
    print(f"  Storage: {cli.db}")
    print(f"  GPU:     {cli.gpu}")
    print(f"  Mode:    {'single-obj TPE' if cli.single_objective else 'multi-obj ' + mo_sampler_name}")
    print(f"  Metric:  {cli.selection_metric}")
    print(f"  Tier:    {cli.search_tier}")
    print(f"  Trials:  {cli.n_trials} (existing: {len(study.trials)})")
    ep_range = DATASET_EPOCH_RANGE[ds]
    if cli.n_epochs_min is not None:
        ep_range = (cli.n_epochs_min, ep_range[1])
    if cli.n_epochs is not None:
        ep_range = (ep_range[0], cli.n_epochs)
    print(f"  Epochs:  {ep_range[0]}~{ep_range[1]} (searched)")
    bc = BATCH_CANDIDATES[ds]
    print(f"  Batch candidates: {bc}")
    print(f"  Python:  {PYTHON}")
    print()

    def after_trial(study, trial):
        if trial.state == optuna.trial.TrialState.PRUNED:
            d = os.path.join(ckpt_base, f"trial_{trial.number}")
            if os.path.isdir(d):
                shutil.rmtree(d, ignore_errors=True)
        elif trial.state == optuna.trial.TrialState.COMPLETE:
            if cli.single_objective:
                cleanup_checkpoints_single(
                    study, ckpt_base,
                    selection_higher_is_better(cli.selection_metric))
            else:
                cleanup_checkpoints_multi(study, ckpt_base)

    study.optimize(
        lambda trial: objective(trial, cli),
        n_trials=cli.n_trials,
        callbacks=[after_trial],
    )

    # ── summary ──
    print("\n" + "=" * 60)
    print("Search complete!")
    if cli.single_objective:
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


if __name__ == "__main__":
    main()
