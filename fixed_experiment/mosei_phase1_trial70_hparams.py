"""Frozen hyperparameters for CMU-MOSEI Optuna phase1 trial 70.

Sources (keep in sync when regenerating):
  - ``My_creation/scripts/verify_mosei_phase1_trial70_optuna_train_argv.py`` — TRIAL_70_PARAMS
    and ``format_train_float_argv`` (must match ``optuna_search_v2.objective()``).
  - ``My_creation/logs/optuna/4090D_restart/phase1/run/mosei.log`` — ``Trial 70`` block
    (``train_batch_size=4``, ``gradient_accumulation_step=8``).
  - Paper: ``overleaf_69e83a58/acl_latex.tex`` (ablation: phase1, trial~70).

Phase1 MOSEI Optuna does not pass ``--early_stop_patience`` (see ``run_optuna_4090d_restart.sh``
``launch_mosei``); defaults in ``train.py`` give patience 0 (train full ``n_epochs``).
"""

from __future__ import annotations

# Tier 1 from phase1 ``Trial 70 finished`` + defaults for knobs not in that study's sample space.
TRIAL_70_PARAMS: dict[str, float | int] = {
    "n_epochs": 50,
    "learning_rate": 3.078834774012053e-05,
    "ig_learning_rate": 0.0005115382788183823,
    "beta_ib": 21.749144451134427,
    "num_infogate_layers": 3,
    "bottleneck_dim": 64,
    "mse_weight": 2.145509477945063,
    "dropout_prob": 0.2680557435615756,
    "alpha_ib": 0.01,
    "stage1_epochs": 10,
    "warmup_proportion": 0.1,
    "weight_decay": 1e-3,
    "ema_decay": 0.999,
    "selector_target_temp": 0.35,
    "selector_rib_weight": 0.05,
    "gumbel_tau_start": 1.0,
    "gumbel_tau_end": 0.5,
    "num_heads": 4,
    "unified_dim": 256,
    "ema_start_epoch": 5,
    "seed": 128,
}

TRAIN_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEP = 8
IB_HIDDEN_DIM = int(TRIAL_70_PARAMS["unified_dim"])
SELECTION_METRIC = "mae"


def format_train_float_argv(p: dict[str, float | int]) -> dict[str, str]:
    """Mirror ``optuna_search_v2.objective()`` argv formatting for float-like knobs."""
    return {
        "learning_rate": f"{float(p['learning_rate']):.6e}",
        "ig_learning_rate": f"{float(p['ig_learning_rate']):.6e}",
        "beta_ib": f"{float(p['beta_ib']):.4f}",
        "mse_weight": f"{float(p['mse_weight']):.4f}",
        "dropout_prob": f"{float(p['dropout_prob']):.4f}",
        "alpha_ib": f"{float(p['alpha_ib']):.6f}",
        "warmup_proportion": f"{float(p['warmup_proportion']):.4f}",
        "weight_decay": f"{float(p['weight_decay']):.6f}",
        "ema_decay": str(p["ema_decay"]),
        "selector_target_temp": f"{float(p['selector_target_temp']):.4f}",
        "selector_rib_weight": f"{float(p['selector_rib_weight']):.4f}",
        "gumbel_tau_start": f"{float(p['gumbel_tau_start']):.4f}",
        "gumbel_tau_end": f"{float(p['gumbel_tau_end']):.4f}",
    }


def build_train_argv(*, checkpoint_dir: str) -> list[str]:
    """CLI tokens for ``train.py`` (flag/value pairs), cwd = ``My_creation``."""
    p = dict(TRIAL_70_PARAMS)
    fmt = format_train_float_argv(p)
    ne = int(p["n_epochs"])
    return [
        "--dataset",
        "mosei",
        "--n_epochs",
        str(ne),
        "--train_batch_size",
        str(TRAIN_BATCH_SIZE),
        "--gradient_accumulation_step",
        str(GRADIENT_ACCUMULATION_STEP),
        "--checkpoint_dir",
        checkpoint_dir,
        "--selection_metric",
        SELECTION_METRIC,
        "--seed",
        str(int(p["seed"])),
        "--learning_rate",
        fmt["learning_rate"],
        "--ig_learning_rate",
        fmt["ig_learning_rate"],
        "--beta_ib",
        fmt["beta_ib"],
        "--num_infogate_layers",
        str(int(p["num_infogate_layers"])),
        "--bottleneck_dim",
        str(int(p["bottleneck_dim"])),
        "--mse_weight",
        fmt["mse_weight"],
        "--dropout_prob",
        fmt["dropout_prob"],
        "--alpha_ib",
        fmt["alpha_ib"],
        "--stage1_epochs",
        str(int(p["stage1_epochs"])),
        "--warmup_proportion",
        fmt["warmup_proportion"],
        "--weight_decay",
        fmt["weight_decay"],
        "--ema_decay",
        fmt["ema_decay"],
        "--selector_target_temp",
        fmt["selector_target_temp"],
        "--selector_rib_weight",
        fmt["selector_rib_weight"],
        "--gumbel_tau_start",
        fmt["gumbel_tau_start"],
        "--gumbel_tau_end",
        fmt["gumbel_tau_end"],
        "--num_heads",
        str(int(p["num_heads"])),
        "--unified_dim",
        str(int(p["unified_dim"])),
        "--ib_hidden_dim",
        str(IB_HIDDEN_DIM),
        "--ema_start_epoch",
        str(int(p["ema_start_epoch"])),
    ]
