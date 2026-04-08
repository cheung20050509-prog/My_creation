"""Optuna v2 — MOSI/MOSEI/SIMSV2 hyperparameter search for InfoGate.
Multi-objective (acc2_composite ↑, MAE ↓) with NSGA-II.
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
EPOCH_LINE_RE = re.compile(r"Epoch (\d+)/\d+")
RESULT_LINE_RE = re.compile(
    r"\s+(Selection score|Acc-2|Acc-7|MAE|Corr|F1):\s+([\d.]+)%?"
)

# ── fixed defaults for params not in active tier ──
DEFAULTS = {
    "seed": 128, "learning_rate": 2e-5, "ig_learning_rate": 5e-4,
    "beta_ib": 32.0, "num_infogate_layers": 3, "bottleneck_dim": 128,
    "mse_weight": 0.5, "dropout_prob": 0.1,
    "gamma_cyc": 1.0, "alpha_ib": 0.01, "alpha_nce": 0.05,
    "alpha_sac": 0.1, "stage1_epochs": 10, "warmup_proportion": 0.1,
    "weight_decay": 1e-3, "ema_decay": 0.999,
    "selector_target_temp": 0.35, "selector_rib_weight": 0.05,
    "gumbel_tau_start": 1.0, "gumbel_tau_end": 0.5,
    "num_heads": 4, "cra_layers": 8, "unified_dim": 256,
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
        "seed": trial.suggest_categorical("seed", [1, 42, 128, 256, 512, 1024, 2024]),
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
        "gamma_cyc": trial.suggest_float("gamma_cyc", 0.1, 3.0),
        "alpha_ib": trial.suggest_float("alpha_ib", 0.001, 0.05, log=True),
        "alpha_nce": trial.suggest_float("alpha_nce", 0.01, 0.2),
        "alpha_sac": trial.suggest_float("alpha_sac", 0.01, 0.3),
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
        "cra_layers": trial.suggest_categorical("cra_layers", [4, 6, 8, 10]),
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
                if key in ("Acc-2", "Acc-7"):
                    val /= 100.0
                if key == "Selection score":
                    key = "SelectionScore"
                results[key] = val
    return results


def parse_best_dev_metrics(log_path):
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
            dm = DEV_LINE_RE.match(line)
            if not dm:
                continue
            acc2, acc7, mae, corr, f1 = (float(x) for x in dm.groups())
            acc2 /= 100.0
            acc7 /= 100.0
            composite = compute_selection_score(
                "acc2_composite", acc2, acc7, mae, corr, f1)
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
    "mosi":   (40, 100),
    "mosei":  (40, 100),
    "simsv2": (30, 80),
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
        "--gamma_cyc", f"{params['gamma_cyc']:.4f}",
        "--alpha_ib", f"{params['alpha_ib']:.6f}",
        "--alpha_nce", f"{params['alpha_nce']:.4f}",
        "--alpha_sac", f"{params['alpha_sac']:.4f}",
        "--stage1_epochs", str(params["stage1_epochs"]),
        "--warmup_proportion", f"{params['warmup_proportion']:.4f}",
        "--weight_decay", f"{params['weight_decay']:.6f}",
        "--ema_decay", str(params["ema_decay"]),
        "--selector_target_temp", f"{params['selector_target_temp']:.4f}",
        "--selector_rib_weight", f"{params['selector_rib_weight']:.4f}",
        "--gumbel_tau_start", f"{params['gumbel_tau_start']:.4f}",
        "--gumbel_tau_end", f"{params['gumbel_tau_end']:.4f}",
        "--num_heads", str(params["num_heads"]),
        "--cra_layers", str(params["cra_layers"]),
        "--unified_dim", str(params["unified_dim"]),
        "--ib_hidden_dim", str(ib_hidden_dim),
        "--ema_start_epoch", str(params["ema_start_epoch"]),
    ]

    for attr, flag in (
        ("disable_l_lib", "--disable_l_lib"),
        ("disable_l_tran", "--disable_l_tran"),
        ("disable_l_rib", "--disable_l_rib"),
        ("disable_sac", "--disable_sac"),
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
    env["CUDA_VISIBLE_DEVICES"] = str(cli.gpu)

    with open(log_path, "w") as log_f:
        proc = subprocess.Popen(
            cmd, stdout=log_f, stderr=subprocess.STDOUT, env=env)

    try:
        while proc.poll() is None:
            time.sleep(15)
            epoch, best_dev = parse_best_dev_metrics(log_path)

            if best_dev is not None:
                composite = compute_selection_score(
                    "acc2_composite",
                    best_dev["Acc2"], best_dev["Acc7"],
                    best_dev["MAE"], best_dev["Corr"], best_dev["F1"])
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
    acc7 = results.get("Acc-7", 0)
    corr = results.get("Corr", 0)
    f1 = results.get("F1", 0)
    composite = compute_selection_score(
        "acc2_composite", acc2, acc7, mae, corr, f1)

    print(f"  Trial {trial.number} DONE: composite={composite:.4f} "
          f"Acc2={acc2*100:.2f}% MAE={mae:.4f} Corr={corr:.4f}")

    if cli.single_objective:
        return compute_selection_score(
            cli.selection_metric, acc2, acc7, mae, corr, f1)
    return composite, mae


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

def main():
    pa = argparse.ArgumentParser(description="Optuna v2 — MOSI/MOSEI/SIMSV2 search")
    pa.add_argument("--dataset", type=str, default="mosi",
                    choices=["mosi", "mosei", "simsv2"])
    pa.add_argument("--gpu", type=int, default=1)
    pa.add_argument("--n_trials", type=int, default=30)
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
    pa.add_argument("--disable_l_lib", action="store_true")
    pa.add_argument("--disable_l_tran", action="store_true")
    pa.add_argument("--disable_l_rib", action="store_true")
    pa.add_argument("--disable_sac", action="store_true")
    cli = pa.parse_args()

    ds = cli.dataset
    if cli.study_name is None:
        cli.study_name = f"infogate_{ds}_v2"

    log_dir = os.path.join(SCRIPT_DIR, "logs", "optuna")
    ckpt_base = os.path.join(SCRIPT_DIR, "checkpoints", f"optuna_{ds}")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_base, exist_ok=True)

    if cli.db is None:
        db_path = os.path.join(log_dir, f"{cli.study_name}.db")
        cli.db = f"sqlite:///{db_path}"

    if cli.single_objective:
        hib = selection_higher_is_better(cli.selection_metric)
        study = optuna.create_study(
            study_name=cli.study_name,
            storage=cli.db,
            direction="maximize" if hib else "minimize",
            sampler=TPESampler(),
            load_if_exists=True,
        )
    else:
        study = optuna.create_study(
            study_name=cli.study_name,
            storage=cli.db,
            directions=["maximize", "minimize"],
            sampler=NSGAIISampler(),
            load_if_exists=True,
        )

    print(f"Optuna v2 — {ds.upper()}")
    print(f"  Study:   {cli.study_name}")
    print(f"  Storage: {cli.db}")
    print(f"  GPU:     {cli.gpu}")
    print(f"  Mode:    {'single-obj' if cli.single_objective else 'multi-obj NSGA-II'}")
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
