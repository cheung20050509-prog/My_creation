# Testing stability analysis (multi-seed)

Freeze Optuna hyperparameters; vary only `--seed`. Report **dev-selected** checkpoint **test** metrics from each run's `Best Results` block.

## CMU-MOSI — paper row (`more/mosi` trial 39, MAE ~0.606)

Gold single-seed repro: `ablation_study/runs/mosi_more_trial39/` (see `ablation_study/README.md`).

**Multi-seed driver** (uses `ablation_study/train_fixed_mosi_more_trial39.py`):

```bash
cd My_creation
bash Testing_Stability_Analysis/run_multi_seed_mosi_trial39.sh
# optional: SEEDS="42 128" GPU=0 MAX_PARALLEL=2 OUT=Testing_Stability_Analysis/runs/my_run
```

**Summarize** (MAE, Corr, Acc-7, Acc-2, F1 + mean ± sample stdev):

```bash
python Testing_Stability_Analysis/collect_stability_metrics.py \
  Testing_Stability_Analysis/runs/mosi_t39_<timestamp>
```

Legacy MAE-only CSV: `collect_best_dev_selected_mae.py` (same `mosi_seed*` layout).

## CMU-MOSEI + MOSI trial 234 (older fixed_experiment)

`run_multi_seed_mosi_mosei.sh` — MOSI uses `fixed_experiment/train_fixed_mosi_trial234.py` (phase4 trial 234, MAE ~0.594); MOSEI uses phase1 trial 70.

For appendix stability aligned with the **paper MOSI main table** (trial 39), use **`run_multi_seed_mosi_trial39.sh`** instead of trial 234 for MOSI.

## Environment

- Conda **`ITHP5090`**, `PYTHON` override supported in shell scripts.
- Default `CUDA_VISIBLE_DEVICES=0`, `MAX_PARALLEL=2`.
