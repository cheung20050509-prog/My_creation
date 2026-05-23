"""Frozen hyperparameters for CH-SIMS v2 Optuna tier3_fresh trial 53 (paper PRISM row).

Sources (keep in sync when regenerating):
  - Study: ``infogate_simsv2_tier3_fresh_mmsa``
  - DB: ``logs/optuna/4090D_restart/simsv2_tier3_fresh/db/simsv2.db``
  - Train log: ``.../simsv2_tier3_fresh/train_logs/simsv2_tier3_fresh_trial_53.log``
  - Best Results (test, selection by dev MAE):
    Acc5=55.61% Acc3=73.40% Acc2=80.08% F1=80.13% MAE=0.2907 Corr=0.7050
  - Paper table (``acl_latex.tex``): 55.6 / 73.4 / 80.1 / 80.1 / 0.291 / 0.705
"""

from __future__ import annotations

# Params from Optuna trial 53 (sqlite trial_params + log header).
TRIAL_53_PARAMS: dict[str, float | int] = {
    "n_epochs": 61,
    "learning_rate": 2.4304567421655e-05,
    "ig_learning_rate": 0.000907583956926654,
    "beta_ib": 12.3622402899691,
    "num_infogate_layers": 3,
    "bottleneck_dim": 64,
    "mse_weight": 0.572877136929496,
    "dropout_prob": 0.194106519245586,
    "alpha_ib": 0.00129998678433247,
    "stage1_epochs": 3,
    "warmup_proportion": 0.142570422565119,
    "weight_decay": 0.00175555929968471,
    "ema_decay": 0.997372716935179,
    "selector_target_temp": 0.792342876228235,
    "selector_rib_weight": 0.106334681848934,
    "gumbel_tau_start": 1.72024501861536,
    "gumbel_tau_end": 0.248170386122028,
    "num_heads": 8,
    "unified_dim": 384,
    "ema_start_epoch": 10,
    "seed": 128,
}

TRAIN_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEP = 4
IB_HIDDEN_DIM = int(TRIAL_53_PARAMS["unified_dim"])
EARLY_STOP_PATIENCE = 15
SELECTION_METRIC = "mae"

PAPER_TEST_METRICS = {
    "acc5": 55.6,
    "acc3": 73.4,
    "acc2": 80.1,
    "f1": 80.1,
    "mae": 0.291,
    "corr": 0.705,
}


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
    p = dict(TRIAL_53_PARAMS)
    fmt = format_train_float_argv(p)
    ne = int(p["n_epochs"])
    return [
        "--dataset",
        "simsv2",
        "--simsv2_feature_mode",
        "mmsa",
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
