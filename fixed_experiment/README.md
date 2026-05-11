# Fixed experiment: frozen Optuna trials (MOSI, MOSEI, SIMSv2)

Reproducible single runs with hyperparameters copied from **`My_creation/logs/optuna/4090D_restart`**. CLI float formatting matches `optuna_search_v2.objective()` so parsed learning rates match the gold Optuna train log headers.

**Environment:** same stack as the Optuna driver is recommended (e.g. conda `ITHP5090`, `transformers==4.29.2`). Override interpreter with `PYTHON=/path/to/python` in the shell wrappers.

### MOSEI + SIMSv2 in parallel (two GPUs)

If you have **two** GPUs, bind one job per card to avoid sequential wall-clock:

```bash
cd My_creation && bash fixed_experiment/run_mosei70_simsv52_parallel.sh
```

Defaults: `MOSEI_GPU=0`, `SIMSV2_GPU=1`. Override, for example:

```bash
MOSEI_GPU=1 SIMSV2_GPU=0 bash fixed_experiment/run_mosei70_simsv52_parallel.sh
```

Or start each run yourself with different `CUDA_VISIBLE_DEVICES` in two terminals.

**Note:** Do not chain `run_mosei…; run_simsv2…` in one `bash -c` if you want parallelism; if you already started such a job, you can `kill` the **outer** sequential `bash -c` PID (the parent of `run_mosei_phase1_trial70.sh`)—MOSEI’s `train.py` typically keeps running while the trailing SIMSv2 step is skipped.

## Vendored Python (snapshot)

These files are **copies** of the corresponding modules under [`My_creation/`](../); imports resolve within `fixed_experiment/`. The copied [`train.py`](train.py) patches paths so **`deberta-v3-base`**, **`bert-base-chinese`**, and **`datasets/{mosi,mosei,simsv2}.pkl`** are read from **`My_creation/`**, not from this directory.

| File | Role |
|------|------|
| [`train.py`](train.py) | Entry training script (path patches for snapshot layout) |
| [`deberta_infogate.py`](deberta_infogate.py) | DeBERTa + InfoGate (MOSI / MOSEI) |
| [`bert_infogate.py`](bert_infogate.py) | BERT path (SIMSv2) |
| [`infogate_modules.py`](infogate_modules.py) | Shared InfoGate blocks |
| [`global_configs.py`](global_configs.py) | Dataset dims / device |
| [`simsv2_metrics.py`](simsv2_metrics.py) | SIMSv2 metrics |
| [`selection_utils.py`](selection_utils.py) | Checkpoint selection helpers |

If you change training logic under `My_creation/`, **refresh these copies** when you want this frozen bundle to stay aligned (for example: `cp ../train.py ../deberta_infogate.py ... fixed_experiment/` then re-apply the small path patches in `fixed_experiment/train.py`).

---

## CMU-MOSI — phase4_mosi trial 234

- **Study:** `infogate_mosi_phase4_mosi_4090d`
- **Files:** [`mosi_trial234_hparams.py`](mosi_trial234_hparams.py), [`train_fixed_mosi_trial234.py`](train_fixed_mosi_trial234.py), [`run_mosi_trial234.sh`](run_mosi_trial234.sh)
- **Outputs:** `fixed_experiment/runs/mosi_trial234/` (`train.log`, `checkpoints/`)

```bash
bash /path/to/My_creation/fixed_experiment/run_mosi_trial234.sh
# or: cd My_creation/fixed_experiment && ./run_mosi_trial234.sh
```

Dry-run:

```bash
cd My_creation && python fixed_experiment/train_fixed_mosi_trial234.py --dry-run
```

Verify argv vs gold log:

```bash
cd My_creation && python scripts/verify_mosi_trial234_optuna_train_argv.py
```

Same hyperparameters as [`run_reproduce_mosi_phase4_mosi_trial234.sh`](../run_reproduce_mosi_phase4_mosi_trial234.sh).

---

## CMU-MOSEI — phase1 trial 70

- **Study:** `infogate_mosei_phase1_4090d` (paper / ablation: phase1, trial 70; see `overleaf_69e83a58/acl_latex.tex`)
- **Gold log:** `logs/optuna/4090D_restart/phase1/train_logs/mosei_phase1_trial_70.log`
- **Files:** [`mosei_phase1_trial70_hparams.py`](mosei_phase1_trial70_hparams.py), [`train_fixed_mosei_phase1_trial70.py`](train_fixed_mosei_phase1_trial70.py), [`run_mosei_phase1_trial70.sh`](run_mosei_phase1_trial70.sh)
- **Outputs:** `fixed_experiment/runs/mosei_phase1_trial70/`

Phase1 MOSEI Optuna does **not** pass `--early_stop_patience` (train full `n_epochs`; `train.py` default patience 0).

```bash
bash /path/to/My_creation/fixed_experiment/run_mosei_phase1_trial70.sh
```

Dry-run / verify:

```bash
cd My_creation && python fixed_experiment/train_fixed_mosei_phase1_trial70.py --dry-run
cd My_creation && python scripts/verify_mosei_phase1_trial70_optuna_train_argv.py
```

---

## CH-SIMS v2 — phase4 trial 52

- **Study:** `infogate_simsv2_phase4_4090d` (paper main row: use final **trial 52** in `phase4/run/simsv2.log`, second `Trial 52 finished` block; Best MAE **0.3113** in `simsv2_phase4_trial_52.log`)
- **Files:** [`simsv2_phase4_trial52_hparams.py`](simsv2_phase4_trial52_hparams.py), [`train_fixed_simsv2_phase4_trial52.py`](train_fixed_simsv2_phase4_trial52.py), [`run_simsv2_phase4_trial52.sh`](run_simsv2_phase4_trial52.sh)
- **Outputs:** `fixed_experiment/runs/simsv2_phase4_trial52/`

Uses `--early_stop_patience 15` (same as `run_optuna_4090d_restart.sh` phase4 SIMSv2 default).

```bash
bash /path/to/My_creation/fixed_experiment/run_simsv2_phase4_trial52.sh
```

Dry-run / verify:

```bash
cd My_creation && python fixed_experiment/train_fixed_simsv2_phase4_trial52.py --dry-run
cd My_creation && python scripts/verify_simsv2_phase4_trial52_optuna_train_argv.py
```

---

## Paper repro bundle

[`scripts/bundle_paper_repro_to_fixed_experiment.sh`](../scripts/bundle_paper_repro_to_fixed_experiment.sh) copies these `*_hparams.py` files into `fixed_experiment/paper_repro_bundle/frozen/` together with thin launchers.
