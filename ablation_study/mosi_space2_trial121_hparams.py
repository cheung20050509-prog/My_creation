"""Frozen hyperparameters for CMU-MOSI Optuna phase4_mosi_space2 trial 121.

Study: ``infogate_mosi_phase4_mosi_4090d_space2`` @
``logs/optuna/4090D_restart/phase4_mosi/db/mosi.db``.

Best Results (reference): Test MAE 0.5939, Acc-2 87.02%, Acc-7 50.07%, Corr 0.8540, F1 0.8696.

Float CLI formatting mirrors ``optuna_search_v2.objective()`` (same as ``mosi_trial234_hparams``).

``batch_config`` 0 → tier-3 MOSI grid: train_batch_size 16, gradient_accumulation_step 2.
"""

from __future__ import annotations

TRIAL_121_PARAMS: dict[str, float | int] = {
    "n_epochs": 96,
    "learning_rate": 3.10498544327124e-05,
    "ig_learning_rate": 0.0002327301375461295,
    "beta_ib": 24.08947741595869,
    "num_infogate_layers": 3,
    "bottleneck_dim": 192,
    "mse_weight": 1.4714739295371693,
    "dropout_prob": 0.27884966454620314,
    "alpha_ib": 0.004116435342639878,
    "stage1_epochs": 7,
    "warmup_proportion": 0.11626737374740492,
    "weight_decay": 0.0007104949909263443,
    "ema_decay": 0.9951841989858001,
    "selector_target_temp": 0.7617380555153733,
    "selector_rib_weight": 0.05773578127779837,
    "gumbel_tau_start": 1.3184732422896817,
    "gumbel_tau_end": 0.1831571994055495,
    "num_heads": 8,
    "unified_dim": 128,
    "ema_start_epoch": 4,
    "seed": 128,
}

TRAIN_BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEP = 2
IB_HIDDEN_DIM = 128
EARLY_STOP_PATIENCE = 15
SELECTION_METRIC = "mae"


def format_train_float_argv(p: dict[str, float | int]) -> dict[str, str]:
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
    p = dict(TRIAL_121_PARAMS)
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
