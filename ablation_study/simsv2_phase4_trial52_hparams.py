"""Frozen hyperparameters for CH-SIMS v2 Optuna 4090D_restart phase4 trial 52.

PRISM runs use ``ablation_study/train.py``; non-``none`` modes append ``--ablation``.
Numeric core matches [`fixed_experiment/simsv2_phase4_trial52_hparams.py`](../fixed_experiment/simsv2_phase4_trial52_hparams.py)
(same knobs as phase6 space3 trial 0; paper row).

Study context: ``4090D_restart/phase4`` SIMSv2. Gold train log:
``logs/optuna/4090D_restart/phase4/train_logs/simsv2_phase4_trial_52.log``.
Selection line: **Best Results (mae, epoch >= 6)** (`stage1_epochs=5`).
"""

from __future__ import annotations

TRAIN_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEP = 4

TRIAL_52_PARAMS: dict[str, float | int] = {
    "n_epochs": 57,
    "learning_rate": 2.0191415275580603e-05,
    "ig_learning_rate": 0.00035238338811010466,
    "beta_ib": 5.440630703880903,
    "num_infogate_layers": 2,
    "bottleneck_dim": 64,
    "mse_weight": 0.7208657261411349,
    "dropout_prob": 0.06418453238018118,
    "alpha_ib": 0.004565520177989885,
    "stage1_epochs": 5,
    "warmup_proportion": 0.07839760585460172,
    "weight_decay": 0.0005782782842106815,
    "ema_decay": 0.9994724074207356,
    "selector_target_temp": 0.40925294131460815,
    "selector_rib_weight": 0.03961433930428535,
    "align_mix_floor": 0.3,
    "gumbel_tau_start": 1.6775306159158943,
    "gumbel_tau_end": 0.5644303121448456,
    "num_heads": 8,
    "unified_dim": 384,
    "ema_start_epoch": 3,
    "seed": 128,
}

IB_HIDDEN_DIM = int(TRIAL_52_PARAMS["unified_dim"])
SELECTION_METRIC = "mae"
EARLY_STOP_PATIENCE = 15


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
        "align_mix_floor": f"{float(p['align_mix_floor']):.4f}",
        "gumbel_tau_start": f"{float(p['gumbel_tau_start']):.4f}",
        "gumbel_tau_end": f"{float(p['gumbel_tau_end']):.4f}",
    }


def build_train_argv(*, checkpoint_dir: str, ablation: str = "none") -> list[str]:
    """CLI tokens for ``ablation_study/train.py`` (flag/value pairs), cwd = ``My_creation``.

    For other modes, appends ``--ablation <mode>`` only.
    """
    p = dict(TRIAL_52_PARAMS)
    fmt = format_train_float_argv(p)
    ne = int(p["n_epochs"])
    argv = [
        "--dataset",
        "simsv2",
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
        "--align_mix_floor",
        fmt["align_mix_floor"],
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
