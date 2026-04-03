"""
Non-invasive Optuna hyperparameter search for InfoGate.
Spawns train.py as subprocess, parses log for results.
"""

import argparse
import os
import re
import shutil
import signal
import subprocess
import time

import optuna

from selection_utils import (
    DEFAULT_SELECTION_METRIC,
    SELECTION_METRIC_CHOICES,
    build_selection_tiebreak,
    compute_selection_score,
    selection_higher_is_better,
)

PYTHON = os.environ.get(
    "PYTHON", "/root/autodl-tmp/anaconda3/envs/ITHP/bin/python")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_SCRIPT = os.path.join(SCRIPT_DIR, "train.py")
LOG_DIR = os.path.join(SCRIPT_DIR, "logs")
CKPT_DIR = os.path.join(SCRIPT_DIR, "checkpoints")
KEEP_TOP_K = 5

os.makedirs(LOG_DIR, exist_ok=True)

DEV_LINE_RE = re.compile(
    r"\s+Dev\s+Acc2=([\d.]+)%\s+Acc7=([\d.]+)%\s+MAE=([\d.]+)\s+Corr=([\d.]+)\s+F1=([\d.]+)"
)
EPOCH_LINE_RE = re.compile(r"Epoch (\d+)/\d+")
RESULT_LINE_RE = re.compile(r"\s+(Selection score|Acc-2|Acc-7|MAE|Corr|F1):\s+([\d.]+)%?")


def append_ablation_flags(cmd, cli_args):
    for attr, flag in (
        ("disable_l_lib", "--disable_l_lib"),
        ("disable_l_tran", "--disable_l_tran"),
        ("disable_l_rib", "--disable_l_rib"),
        ("disable_sac", "--disable_sac"),
    ):
        if getattr(cli_args, attr, False):
            cmd.append(flag)

# ------------------------------------------------------------------
# Log parsing
# ------------------------------------------------------------------

def parse_best_results(log_path):
    """Parse Best Results block from the end of a training log."""
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
                key, raw_val = m.groups()
                val = float(raw_val)
                if key in ("Acc-2", "Acc-7"):
                    val /= 100.0
                if key == "Selection score":
                    key = "SelectionScore"
                results[key] = val
    return results


def parse_best_dev_metrics(log_path, selection_metric):
    """Parse the best dev metrics seen so far using the active selection objective."""
    if not os.path.exists(log_path):
        return 0, None

    current_epoch = 0
    best_metrics = None
    best_score = None
    best_tiebreak = None
    higher_is_better = selection_higher_is_better(selection_metric)

    with open(log_path, "r") as f:
        for line in f:
            epoch_match = EPOCH_LINE_RE.match(line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
                continue

            dev_match = DEV_LINE_RE.match(line)
            if not dev_match:
                continue

            acc2, acc7, mae, corr, f1 = map(float, dev_match.groups())
            metrics = {
                "Acc2": acc2 / 100.0,
                "Acc7": acc7 / 100.0,
                "MAE": mae,
                "Corr": corr,
                "F1": f1,
            }
            score = compute_selection_score(
                selection_metric,
                metrics["Acc2"],
                metrics["Acc7"],
                metrics["MAE"],
                metrics["Corr"],
                metrics["F1"],
            )
            tiebreak = build_selection_tiebreak(
                metrics["Acc2"],
                metrics["Acc7"],
                metrics["MAE"],
                metrics["Corr"],
                metrics["F1"],
            )

            if best_metrics is None:
                best_metrics = metrics
                best_score = score
                best_tiebreak = tiebreak
                continue

            better_score = score > best_score if higher_is_better else score < best_score
            same_score = abs(score - best_score) <= 1e-12
            if better_score or (same_score and tiebreak > best_tiebreak):
                best_metrics = metrics
                best_score = score
                best_tiebreak = tiebreak

    return current_epoch, best_metrics


def get_current_epoch(log_path):
    """Get the latest epoch number from log."""
    if not os.path.exists(log_path):
        return 0
    current = 0
    with open(log_path, "r") as f:
        for line in f:
            m = re.match(r"Epoch (\d+)/\d+", line)
            if m:
                current = int(m.group(1))
    return current


def should_prune(selection_metric, epoch, best_metrics):
    if best_metrics is None:
        return False, None, None

    if epoch >= 40:
        if selection_metric in ("acc2_composite", "acc2"):
            threshold = 0.82
            value = best_metrics["Acc2"]
            return value < threshold, "best dev Acc2", threshold
        if selection_metric == "acc7":
            threshold = 0.45
            value = best_metrics["Acc7"]
            return value < threshold, "best dev Acc7", threshold
        if selection_metric == "f1":
            threshold = 0.82
            value = best_metrics["F1"]
            return value < threshold, "best dev F1", threshold
        if selection_metric == "corr":
            threshold = 0.78
            value = best_metrics["Corr"]
            return value < threshold, "best dev Corr", threshold
        threshold = 0.65
        value = best_metrics["MAE"]
        return value > threshold, "best dev MAE", threshold

    if epoch >= 20:
        if selection_metric in ("acc2_composite", "acc2"):
            threshold = 0.78
            value = best_metrics["Acc2"]
            return value < threshold, "best dev Acc2", threshold
        if selection_metric == "acc7":
            threshold = 0.42
            value = best_metrics["Acc7"]
            return value < threshold, "best dev Acc7", threshold
        if selection_metric == "f1":
            threshold = 0.78
            value = best_metrics["F1"]
            return value < threshold, "best dev F1", threshold
        if selection_metric == "corr":
            threshold = 0.72
            value = best_metrics["Corr"]
            return value < threshold, "best dev Corr", threshold
        threshold = 0.75
        value = best_metrics["MAE"]
        return value > threshold, "best dev MAE", threshold

    return False, None, None


# ------------------------------------------------------------------
# Checkpoint cleanup: keep only top-K trials
# ------------------------------------------------------------------

def cleanup_checkpoints(study, higher_is_better):
    """Keep only KEEP_TOP_K best trial checkpoints, delete the rest."""
    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    if len(completed) <= KEEP_TOP_K:
        return
    ranked = sorted(completed, key=lambda t: t.value, reverse=higher_is_better)
    keep_nums = {t.number for t in ranked[:KEEP_TOP_K]}
    for t in completed:
        if t.number not in keep_nums:
            d = os.path.join(CKPT_DIR, f"trial_{t.number}")
            if os.path.isdir(d):
                shutil.rmtree(d, ignore_errors=True)


# ------------------------------------------------------------------
# Objective
# ------------------------------------------------------------------

def objective(trial, cli_args):
    # --- Search space (expanded) ---
    seed = trial.suggest_categorical("seed", [1, 42, 128, 256, 512, 1024, 2024])
    lr = trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True)
    ig_lr = trial.suggest_float("ig_learning_rate", 1e-4, 1e-3, log=True)
    mse_weight = trial.suggest_float("mse_weight", 0.3, 1.5)
    dropout = trial.suggest_float("dropout_prob", 0.15, 0.35)
    bottleneck = trial.suggest_categorical("bottleneck_dim", [96, 128, 192])
    beta_ib = trial.suggest_float("beta_ib", 8.0, 32.0)
    alpha_ib = trial.suggest_float("alpha_ib", 0.001, 0.02, log=True)
    ema_decay = trial.suggest_categorical("ema_decay", [0.99, 0.995, 0.999])
    weight_decay = trial.suggest_float("weight_decay", 0.005, 0.05, log=True)
    stage1_epochs = trial.suggest_int("stage1_epochs", 5, 15)
    num_layers = trial.suggest_categorical("num_infogate_layers", [2, 3, 4])
    gamma_cyc = trial.suggest_float("gamma_cyc", 0.5, 2.0)
    warmup = trial.suggest_float("warmup_proportion", 0.05, 0.2)

    n_epochs = cli_args.n_epochs
    log_path = os.path.join(LOG_DIR, f"optuna_trial_{trial.number}.log")
    trial_ckpt_dir = os.path.join(CKPT_DIR, f"trial_{trial.number}")
    os.makedirs(trial_ckpt_dir, exist_ok=True)

    cmd = [
        PYTHON, "-u", TRAIN_SCRIPT,
        "--dataset", "mosi",
        "--n_epochs", str(n_epochs),
        "--stage1_epochs", str(stage1_epochs),
        "--train_batch_size", "16",
        "--gradient_accumulation_step", "2",
        "--learning_rate", f"{lr:.6e}",
        "--ig_learning_rate", f"{ig_lr:.6e}",
        "--unified_dim", "256",
        "--ib_hidden_dim", "256",
        "--bottleneck_dim", str(bottleneck),
        "--num_heads", "4",
        "--num_infogate_layers", str(num_layers),
        "--beta_ib", f"{beta_ib:.4f}",
        "--gamma_cyc", f"{gamma_cyc:.4f}",
        "--alpha_ib", f"{alpha_ib:.6f}",
        "--mse_weight", f"{mse_weight:.4f}",
        "--cra_layers", "8",
        "--dropout_prob", f"{dropout:.4f}",
        "--weight_decay", f"{weight_decay:.6f}",
        "--ema_decay", str(ema_decay),
        "--ema_start_epoch", "5",
        "--warmup_proportion", f"{warmup:.4f}",
        "--checkpoint_dir", trial_ckpt_dir,
        "--selection_metric", cli_args.selection_metric,
        "--seed", str(seed),
    ]
    append_ablation_flags(cmd, cli_args)

    print(f"\n{'='*60}")
    print(f"Trial {trial.number} starting")
    print(f"  seed={seed} lr={lr:.2e} ig_lr={ig_lr:.2e} mse_w={mse_weight:.2f}")
    print(f"  dropout={dropout:.3f} bn={bottleneck} beta_ib={beta_ib:.1f}")
    print(f"  alpha_ib={alpha_ib:.4f} select={cli_args.selection_metric} ema={ema_decay}")
    print(f"  wd={weight_decay:.4f} stage1={stage1_epochs} layers={num_layers}")
    print(f"  gamma_cyc={gamma_cyc:.2f} warmup={warmup:.3f}")
    print(f"  ablate: L_lib={cli_args.disable_l_lib} L_tran={cli_args.disable_l_tran} "
          f"L_rib={cli_args.disable_l_rib} SAC={cli_args.disable_sac}")
    print(f"  log: {log_path}")
    print(f"{'='*60}")

    with open(log_path, "w") as log_f:
        proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT)

    try:
        while proc.poll() is None:
            time.sleep(15)
            epoch, best_dev_metrics = parse_best_dev_metrics(
                log_path, cli_args.selection_metric)

            prune, metric_name, threshold = should_prune(
                cli_args.selection_metric, epoch, best_dev_metrics)
            if prune:
                current_value = None
                if best_dev_metrics is not None and metric_name is not None:
                    if "Acc2" in metric_name:
                        current_value = best_dev_metrics["Acc2"]
                    elif "Acc7" in metric_name:
                        current_value = best_dev_metrics["Acc7"]
                    elif "F1" in metric_name:
                        current_value = best_dev_metrics["F1"]
                    elif "Corr" in metric_name:
                        current_value = best_dev_metrics["Corr"]
                    else:
                        current_value = best_dev_metrics["MAE"]
                value_str = f"{current_value:.4f}" if current_value is not None else "n/a"
                print(f"  Trial {trial.number} pruned at epoch {epoch}: "
                      f"{metric_name}={value_str}, threshold={threshold:.4f}")
                proc.send_signal(signal.SIGTERM)
                proc.wait(timeout=10)
                raise optuna.TrialPruned()

        if proc.returncode != 0:
            print(f"  Trial {trial.number} failed with exit code {proc.returncode}")
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
        raise optuna.TrialPruned()

    results = parse_best_results(log_path)
    if "MAE" not in results:
        print(f"  Trial {trial.number}: could not parse MAE from log")
        raise optuna.TrialPruned()

    mae = results["MAE"]
    acc2 = results.get("Acc-2", 0)
    acc7 = results.get("Acc-7", 0)
    corr = results.get("Corr", 0)
    f1 = results.get("F1", 0)
    objective_value = results.get("SelectionScore")
    if objective_value is None:
        objective_value = compute_selection_score(
            cli_args.selection_metric,
            acc2,
            acc7,
            mae,
            corr,
            f1,
        )

    print(f"  Trial {trial.number} done: {cli_args.selection_metric}={objective_value:.6f} "
          f"MAE={mae:.4f} Acc2={acc2*100:.2f}% Acc7={acc7*100:.2f}% Corr={corr:.4f} F1={f1:.4f}")

    return objective_value


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Optuna search for InfoGate")
    parser.add_argument("--n_trials", type=int, default=30)
    parser.add_argument("--n_epochs", type=int, default=80)
    parser.add_argument("--study_name", type=str, default=None)
    parser.add_argument("--db", type=str, default=None,
                        help="Optuna storage URL. Default: sqlite in logs/")
    parser.add_argument("--selection_metric", type=str,
                        default=DEFAULT_SELECTION_METRIC,
                        choices=SELECTION_METRIC_CHOICES)
    parser.add_argument("--disable_l_lib", action="store_true")
    parser.add_argument("--disable_l_tran", action="store_true")
    parser.add_argument("--disable_l_rib", action="store_true")
    parser.add_argument("--disable_sac", action="store_true")
    cli_args = parser.parse_args()

    if cli_args.study_name is None:
        cli_args.study_name = f"infogate_mosi_{cli_args.selection_metric}"

    if cli_args.db is None:
        db_path = os.path.join(LOG_DIR, f"{cli_args.study_name}.db")
        cli_args.db = f"sqlite:///{db_path}"

    higher_is_better = selection_higher_is_better(cli_args.selection_metric)

    study = optuna.create_study(
        study_name=cli_args.study_name,
        storage=cli_args.db,
        direction="maximize" if higher_is_better else "minimize",
        load_if_exists=True,
    )

    print(f"Optuna study: {cli_args.study_name}")
    print(f"Storage: {cli_args.db}")
    print(f"Selection metric: {cli_args.selection_metric}")
    print(f"Ablations: L_lib={cli_args.disable_l_lib} L_tran={cli_args.disable_l_tran} "
          f"L_rib={cli_args.disable_l_rib} SAC={cli_args.disable_sac}")
    print(f"Trials: {cli_args.n_trials}, Epochs per trial: {cli_args.n_epochs}")
    print(f"Existing trials: {len(study.trials)}")

    def after_trial_callback(study, trial):
        if trial.state == optuna.trial.TrialState.COMPLETE:
            cleanup_checkpoints(study, higher_is_better)
        elif trial.state == optuna.trial.TrialState.PRUNED:
            d = os.path.join(CKPT_DIR, f"trial_{trial.number}")
            if os.path.isdir(d):
                shutil.rmtree(d, ignore_errors=True)

    study.optimize(
        lambda trial: objective(trial, cli_args),
        n_trials=cli_args.n_trials,
        callbacks=[after_trial_callback],
    )

    print("\n" + "=" * 60)
    print("Search complete!")
    print(f"Best trial: #{study.best_trial.number}")
    print(f"Best objective ({cli_args.selection_metric}): {study.best_trial.value:.6f}")
    print("Best params:")
    for k, v in study.best_trial.params.items():
        print(f"  {k}: {v}")

    results_path = os.path.join(LOG_DIR, "optuna_best_params.txt")
    with open(results_path, "w") as f:
        f.write(f"Best trial: #{study.best_trial.number}\n")
        f.write(f"Selection metric: {cli_args.selection_metric}\n")
        f.write(f"Best objective: {study.best_trial.value:.6f}\n")
        f.write("Best params:\n")
        for k, v in study.best_trial.params.items():
            f.write(f"  {k}: {v}\n")
        log_path = os.path.join(LOG_DIR,
                                f"optuna_trial_{study.best_trial.number}.log")
        best_results = parse_best_results(log_path)
        f.write("\nFull results from best trial log:\n")
        for k, v in best_results.items():
            f.write(f"  {k}: {v}\n")
    print(f"Best params saved to {results_path}")


if __name__ == "__main__":
    main()
