"""Frozen hyperparameters for CMU-MOSI Optuna phase4_mosi trial 234.

Sources (keep in sync when regenerating):
  - ``My_creation/scripts/verify_mosi_trial234_optuna_train_argv.py`` — TRIAL_234_PARAMS
    and ``format_train_float_argv`` (must match ``optuna_search_v2.objective()``).
  - ``My_creation/run_reproduce_mosi_phase4_mosi_trial234.sh`` — batch sizes,
    ``ib_hidden_dim``, ``early_stop_patience``, etc.
"""

from __future__ import annotations

# Params from phase4_mosi ``Trial 234 finished`` (excluding batch_config); full precision.
TRIAL_234_PARAMS: dict[str, float | int] = {
    "n_epochs": 98,
    "learning_rate": 2.4786171641775438e-05,
    "ig_learning_rate": 0.00022666892434202945,
    "beta_ib": 23.399726112667697,
    "num_infogate_layers": 3,
    "bottleneck_dim": 192,
    "mse_weight": 1.2122529591428757,
    "dropout_prob": 0.24999921019649488,
    "alpha_ib": 0.003725550639603183,
    "stage1_epochs": 7,
    "warmup_proportion": 0.1269077082716429,
    "weight_decay": 0.0007294609745937727,
    "ema_decay": 0.9951979795863604,
    "selector_target_temp": 0.7254914650214497,
    "selector_rib_weight": 0.06363196490946363,
    "gumbel_tau_start": 1.11350211565539,
    "gumbel_tau_end": 0.16846568512700616,
    "num_heads": 8,
    "unified_dim": 128,
    "ema_start_epoch": 5,
    "seed": 128,
}

# batch_config 0 in Optuna search → tier-3 MOSI grid (see reproduce script).
TRAIN_BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEP = 2
IB_HIDDEN_DIM = 128
EARLY_STOP_PATIENCE = 15
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
    p = dict(TRIAL_234_PARAMS)
    fmt = format_train_float_argv(p)
    ne = int(p["n_epochs"])
    return [
        "--dataset",
        "mosi",
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
        "--early_stop_patience",
        str(EARLY_STOP_PATIENCE),
    ]
