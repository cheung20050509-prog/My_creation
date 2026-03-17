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

PYTHON = os.environ.get(
    "PYTHON", "/root/autodl-tmp/anaconda3/envs/ITHP/bin/python")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_SCRIPT = os.path.join(SCRIPT_DIR, "train.py")
LOG_DIR = os.path.join(SCRIPT_DIR, "logs")
CKPT_DIR = os.path.join(SCRIPT_DIR, "checkpoints")
KEEP_TOP_K = 5

os.makedirs(LOG_DIR, exist_ok=True)

# ------------------------------------------------------------------
# Log parsing
# ------------------------------------------------------------------

def parse_best_results(log_path):
    """Parse Best Results block from the end of a training log."""
    results = {}
    if not os.path.exists(log_path):
        return results
    with open(log_path, "r") as f:
        lines = f.readlines()
    in_block = False
    for line in lines:
        if "Best Results" in line and "oracle" not in line.lower():
            in_block = True
            continue
        if in_block:
            m = re.match(r"\s+(Acc-2|Acc-7|MAE|Corr|F1):\s+([\d.]+)", line)
            if m:
                key = m.group(1)
                val = float(m.group(2))
                if key in ("Acc-2", "Acc-7"):
                    val /= 100.0
                results[key] = val
    return results


def parse_dev_mae_at_epoch(log_path, target_epoch):
    """Parse dev MAE at a specific epoch for pruning."""
    if not os.path.exists(log_path):
        return None
    best_dev_mae = None
    with open(log_path, "r") as f:
        for line in f:
            m = re.match(r"\s+Dev\s+.*MAE=([\d.]+)", line)
            if m:
                mae = float(m.group(1))
                if best_dev_mae is None or mae < best_dev_mae:
                    best_dev_mae = mae
    current_epoch = 0
    with open(log_path, "r") as f:
        for line in f:
            m = re.match(r"Epoch (\d+)/\d+", line)
            if m:
                current_epoch = int(m.group(1))
    if current_epoch >= target_epoch:
        return best_dev_mae
    return None


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


# ------------------------------------------------------------------
# Checkpoint cleanup: keep only top-K trials
# ------------------------------------------------------------------

def cleanup_checkpoints(study):
    """Keep only KEEP_TOP_K best trial checkpoints, delete the rest."""
    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    if len(completed) <= KEEP_TOP_K:
        return
    ranked = sorted(completed, key=lambda t: t.value)
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
    # --- Search space (full) ---
    seed = trial.suggest_categorical("seed", [1, 42, 128, 256, 512, 1024, 2024])
    lr = trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True)
    ig_lr = trial.suggest_float("ig_learning_rate", 1e-4, 1e-3, log=True)
    mse_weight = trial.suggest_float("mse_weight", 0.3, 1.5)
    dropout = trial.suggest_float("dropout_prob", 0.15, 0.35)
    bottleneck = trial.suggest_categorical("bottleneck_dim", [64, 96, 128, 192])
    beta_ib = trial.suggest_float("beta_ib", 8.0, 32.0)
    alpha_ib = trial.suggest_float("alpha_ib", 0.001, 0.02, log=True)
    alpha_sac = trial.suggest_float("alpha_sac", 0.005, 0.05, log=True)
    weight_decay = trial.suggest_float("weight_decay", 0.005, 0.05, log=True)
    stage1_epochs = trial.suggest_int("stage1_epochs", 5, 15)
    num_layers = trial.suggest_categorical("num_infogate_layers", [2, 3, 4])
    gamma_cyc = trial.suggest_float("gamma_cyc", 0.3, 2.0)
    warmup = trial.suggest_float("warmup_proportion", 0.05, 0.2)
    n_epochs = trial.suggest_categorical("n_epochs", [60, 80, 100, 120])
    batch_size = trial.suggest_categorical("train_batch_size", [8, 16, 32, 64])
    grad_accum = trial.suggest_categorical("gradient_accumulation_step", [1, 2, 4])
    unified_dim = trial.suggest_categorical("unified_dim", [128, 256, 384])
    ib_hidden_dim = trial.suggest_categorical("ib_hidden_dim", [128, 256])
    cra_layers = trial.suggest_categorical("cra_layers", [4, 6, 8])

    log_path = os.path.join(LOG_DIR, f"optuna_trial_{trial.number}.log")
    trial_ckpt_dir = os.path.join(CKPT_DIR, f"trial_{trial.number}")
    os.makedirs(trial_ckpt_dir, exist_ok=True)

    cmd = [
        PYTHON, "-u", TRAIN_SCRIPT,
        "--dataset", "mosi",
        "--n_epochs", str(n_epochs),
        "--stage1_epochs", str(stage1_epochs),
        "--train_batch_size", str(batch_size),
        "--gradient_accumulation_step", str(grad_accum),
        "--learning_rate", f"{lr:.6e}",
        "--ig_learning_rate", f"{ig_lr:.6e}",
        "--unified_dim", str(unified_dim),
        "--ib_hidden_dim", str(ib_hidden_dim),
        "--bottleneck_dim", str(bottleneck),
        "--num_heads", "4",
        "--num_infogate_layers", str(num_layers),
        "--beta_ib", f"{beta_ib:.4f}",
        "--gamma_cyc", f"{gamma_cyc:.4f}",
        "--alpha_ib", f"{alpha_ib:.6f}",
        "--alpha_sac", f"{alpha_sac:.6f}",
        "--mse_weight", f"{mse_weight:.4f}",
        "--cra_layers", str(cra_layers),
        "--dropout_prob", f"{dropout:.4f}",
        "--weight_decay", f"{weight_decay:.6f}",
        "--ema_start_epoch", "999",
        "--warmup_proportion", f"{warmup:.4f}",
        "--checkpoint_dir", trial_ckpt_dir,
        "--seed", str(seed),
    ]

    print(f"\n{'='*60}")
    print(f"Trial {trial.number} starting")
    print(f"  seed={seed} lr={lr:.2e} ig_lr={ig_lr:.2e} mse_w={mse_weight:.2f}")
    print(f"  dropout={dropout:.3f} bn={bottleneck} beta_ib={beta_ib:.1f}")
    print(f"  alpha_ib={alpha_ib:.4f} alpha_sac={alpha_sac:.4f}")
    print(f"  wd={weight_decay:.4f} stage1={stage1_epochs} layers={num_layers}")
    print(f"  gamma_cyc={gamma_cyc:.2f} warmup={warmup:.3f}")
    print(f"  epochs={n_epochs} bs={batch_size} accum={grad_accum}")
    print(f"  udim={unified_dim} ibhid={ib_hidden_dim} cra={cra_layers}")
    print(f"  log: {log_path}")
    print(f"{'='*60}")

    with open(log_path, "w") as log_f:
        proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT)

    try:
        while proc.poll() is None:
            time.sleep(15)
            epoch = get_current_epoch(log_path)

            # Pruning checks
            if epoch >= 20:
                dev_mae = parse_dev_mae_at_epoch(log_path, 20)
                if dev_mae is not None and dev_mae > 0.75:
                    print(f"  Trial {trial.number} pruned at epoch {epoch}: "
                          f"dev MAE {dev_mae:.4f} > 0.75")
                    proc.send_signal(signal.SIGTERM)
                    proc.wait(timeout=10)
                    raise optuna.TrialPruned()

            if epoch >= 40:
                dev_mae = parse_dev_mae_at_epoch(log_path, 40)
                if dev_mae is not None and dev_mae > 0.65:
                    print(f"  Trial {trial.number} pruned at epoch {epoch}: "
                          f"dev MAE {dev_mae:.4f} > 0.65")
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

    print(f"  Trial {trial.number} done: MAE={mae:.4f} Acc2={acc2*100:.2f}% "
          f"Acc7={acc7*100:.2f}% Corr={corr:.4f} F1={f1:.4f}")

    return mae


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Optuna search for InfoGate")
    parser.add_argument("--n_trials", type=int, default=30)
    parser.add_argument("--n_epochs", type=int, default=80)
    parser.add_argument("--study_name", type=str, default="infogate_mosi")
    parser.add_argument("--db", type=str, default=None,
                        help="Optuna storage URL. Default: sqlite in logs/")
    cli_args = parser.parse_args()

    if cli_args.db is None:
        db_path = os.path.join(LOG_DIR, f"{cli_args.study_name}.db")
        cli_args.db = f"sqlite:///{db_path}"

    sampler = optuna.samplers.TPESampler(
        multivariate=True, n_startup_trials=20)
    study = optuna.create_study(
        study_name=cli_args.study_name,
        storage=cli_args.db,
        direction="minimize",
        load_if_exists=True,
        sampler=sampler,
    )

    print(f"Optuna study: {cli_args.study_name}")
    print(f"Storage: {cli_args.db}")
    print(f"Trials: {cli_args.n_trials}, Epochs per trial: {cli_args.n_epochs}")
    print(f"Existing trials: {len(study.trials)}")

    def after_trial_callback(study, trial):
        if trial.state == optuna.trial.TrialState.COMPLETE:
            cleanup_checkpoints(study)
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
    print(f"Best MAE: {study.best_trial.value:.4f}")
    print("Best params:")
    for k, v in study.best_trial.params.items():
        print(f"  {k}: {v}")

    results_path = os.path.join(LOG_DIR, "optuna_best_params.txt")
    with open(results_path, "w") as f:
        f.write(f"Best trial: #{study.best_trial.number}\n")
        f.write(f"Best MAE: {study.best_trial.value:.4f}\n")
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
