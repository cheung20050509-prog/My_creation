"""Frozen hyperparameters for CMU-MOSI Optuna phase4_mosi trial 220.

PRISM runs use ``ablation_study/train.py``; non-``none`` modes only append ``--ablation``.

Sources (keep in sync when regenerating):
  - ``logs/optuna/4090D_restart/phase4_mosi/run/mosi.log`` — ``Trial 220 finished``
  - ``logs/optuna/4090D_restart/phase4_mosi/train_logs/mosi_phase4_mosi_trial_220.log``
  - ``My_creation/scripts/verify_mosi_trial220_optuna_train_argv.py``
"""

from __future__ import annotations

# Params from phase4_mosi ``Trial 220 finished`` (excluding batch_config); full precision.
TRIAL_220_PARAMS: dict[str, float | int] = {
    "n_epochs": 97,
    "learning_rate": 2.9273773228404854e-05,
    "ig_learning_rate": 0.00024500602832556887,
    "beta_ib": 22.973655873057357,
    "num_infogate_layers": 3,
    "bottleneck_dim": 192,
    "mse_weight": 1.2159459857149533,
    "dropout_prob": 0.24995585840082163,
    "alpha_ib": 0.0038406547193778717,
    "stage1_epochs": 6,
    "warmup_proportion": 0.10119382704338378,
    "weight_decay": 0.0007373939875592053,
    "ema_decay": 0.9951883680038079,
    "selector_target_temp": 0.7113722119040652,
    "selector_rib_weight": 0.060310067628194186,
    "gumbel_tau_start": 1.487505343680365,
    "gumbel_tau_end": 0.17652232232384735,
    "num_heads": 8,
    "unified_dim": 128,
    "ema_start_epoch": 5,
    "seed": 128,
}

# batch_config 0 in Optuna search → tier-3 MOSI grid (same as trial 234).
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


def build_train_argv(*, checkpoint_dir: str, ablation: str = "none") -> list[str]:
    """CLI tokens for ``ablation_study/train.py`` (flag/value pairs), cwd = ``My_creation``.

    For other modes, appends ``--ablation <mode>`` only.
    """
    p = dict(TRIAL_220_PARAMS)
    fmt = format_train_float_argv(p)
    ne = int(p["n_epochs"])
    argv = [
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
    if ablation != "none":
        argv.extend(["--ablation", ablation])
    return argv
