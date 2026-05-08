#!/usr/bin/env python3
"""Best hyperparameters from 4090D_restart Optuna search, paper-reported trials."""

# ============================================================
# Regression (train.py)
# ============================================================

MOSI = dict(
    dataset="mosi",
    train_batch_size=16,
    gradient_accumulation_step=2,
    n_epochs=98,
    stage1_epochs=7,
    learning_rate=2.4786e-5,
    ig_learning_rate=2.2667e-4,
    beta_ib=23.40,
    alpha_ib=3.7256e-3,
    num_infogate_layers=3,
    bottleneck_dim=192,
    unified_dim=128,
    num_heads=8,
    mse_weight=1.21,
    dropout_prob=0.25,
    warmup_proportion=0.127,
    weight_decay=7.29e-4,
    ema_decay=0.995,
    ema_start_epoch=5,
    selector_target_temp=0.725,
    selector_rib_weight=0.064,
    gumbel_tau_start=1.11,
    gumbel_tau_end=0.168,
    seed=128,
    selection_metric="mae",
    # trial 234, phase4_mosi; test MAE=0.594, Corr=0.857
)

MOSEI = dict(
    dataset="mosei",
    train_batch_size=4,
    gradient_accumulation_step=8,
    n_epochs=50,
    # stage1_epochs not used (no MSelector in phase1)
    learning_rate=3.0788e-5,
    ig_learning_rate=5.1154e-4,
    beta_ib=21.75,
    # alpha_ib not used (default 0.01)
    num_infogate_layers=3,
    bottleneck_dim=64,
    # unified_dim, num_heads use defaults (256, 4)
    mse_weight=2.15,
    dropout_prob=0.268,
    # warmup_proportion, weight_decay use defaults
    seed=128,
    selection_metric="mae",
    # trial 70, phase1; test MAE=0.499, Corr=0.800
)

SIMSv2 = dict(
    dataset="simsv2",
    train_batch_size=8,
    gradient_accumulation_step=4,
    n_epochs=56,
    stage1_epochs=6,
    learning_rate=1.8620e-5,
    ig_learning_rate=4.0172e-4,
    beta_ib=5.08,
    alpha_ib=3.7474e-3,
    num_infogate_layers=3,
    bottleneck_dim=64,
    unified_dim=384,
    num_heads=8,
    mse_weight=0.73,
    dropout_prob=0.067,
    warmup_proportion=0.078,
    weight_decay=6.72e-4,
    ema_decay=0.9995,
    ema_start_epoch=3,
    selector_target_temp=0.443,
    selector_rib_weight=0.038,
    gumbel_tau_start=1.74,
    gumbel_tau_end=0.553,
    seed=128,
    selection_metric="mae",
    # trial 52, phase4; test MAE=0.311, Corr=0.686
)

# ============================================================
# Classification (train_classify.py)
# ============================================================

UR_FUNNY = dict(
    dataset="ur_funny",
    train_batch_size=32,
    gradient_accumulation_step=1,
    n_epochs=38,
    stage1_epochs=3,
    learning_rate=1.2121e-5,
    ig_learning_rate=4.0858e-4,
    beta_ib=32.23,
    alpha_ib=4.3775e-4,
    num_infogate_layers=5,
    bottleneck_dim=96,
    unified_dim=256,
    num_heads=8,
    dropout_prob=0.380,
    warmup_proportion=0.229,
    weight_decay=3.10e-3,
    ema_decay=0.999,
    ema_start_epoch=5,
    selector_target_temp=0.399,
    selector_rib_weight=0.100,
    gumbel_tau_start=0.885,
    gumbel_tau_end=0.386,
    freeze_backbone_stage2=False,
    bce_pos_weight_mode="none",
    focal_gamma=0.28,
    rdrop_alpha=0.87,
    ib_loss_mult_end=0.49,
    early_stop_patience=4,
    selection_smooth_window=2,
    selector_balance_weight=0.041,
    seed=42,
    selection_metric="binary_acc",
    # trial 34, gumbel run; test Acc=74.5%, F1=74.4%
)

MUSTARD = dict(
    dataset="mustard",
    train_batch_size=16,
    gradient_accumulation_step=1,
    n_epochs=53,
    stage1_epochs=10,
    learning_rate=3.9536e-5,
    ig_learning_rate=9.8919e-4,
    beta_ib=6.57,
    alpha_ib=7.3792e-3,
    num_infogate_layers=3,
    bottleneck_dim=128,
    unified_dim=128,
    num_heads=8,
    dropout_prob=0.422,
    warmup_proportion=0.133,
    weight_decay=3.99e-3,
    ema_decay=0.99,
    ema_start_epoch=3,
    selector_target_temp=0.545,
    selector_rib_weight=0.016,
    gumbel_tau_start=0.614,
    gumbel_tau_end=0.429,
    freeze_backbone_stage2=True,
    bce_pos_weight_mode="none",
    focal_gamma=1.29,
    rdrop_alpha=3.99,
    ib_loss_mult_end=0.61,
    early_stop_patience=22,
    selection_smooth_window=1,
    selector_balance_weight=0.041,
    seed=42,
    selection_metric="binary_acc",
    # trial 26, 20260501 run; test Acc=75.0%, F1=74.9%
)
