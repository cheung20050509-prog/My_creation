# Fixed Experiments — PRISM Best Hyperparameters

Self-contained training directory. All code, models (symlinked), and best hyperparameters live here.

## Usage

```bash
# Regression (MOSI / MOSEI / SIMSv2)
bash run_mosi.sh      # CMU-MOSI,  MAE=0.594, Corr=0.857
bash run_mosei.sh     # CMU-MOSEI, MAE=0.499, Corr=0.800
bash run_simsv2.sh    # CH-SIMS v2, MAE=0.311, Corr=0.686

# Classification (UR-FUNNY / MUStARD)
bash run_ur_funny.sh  # UR-FUNNY, Acc=74.5%
bash run_mustard.sh   # MUStARD,   Acc=75.0%

# All five
bash run_all.sh
```

CLI overrides still work — any flag passed on top overrides the default:

```bash
python train_regression.py --dataset mosi --learning_rate 1e-5 --seed 42
python train_classification.py --dataset mustard --focal_gamma 2.0
```

## Structure

- `train_regression.py` / `train_classification.py` — main entry points; auto-load best hparams from `hparams.py` based on `--dataset`
- `hparams.py` — best hyperparameters from 4090D_restart Optuna search as importable Python dicts
- `infogate_modules.py`, `*_infogate.py` — InfoGate model components
- `global_configs.py`, `selection_utils.py`, `simsv2_metrics.py`, `data_humor.py` — utilities
- `optuna_search_v2.py`, `optuna_search_classify.py` — Optuna search drivers (for hyperparameter tuning)
- `deberta-v3-base/`, `bert-base-chinese/`, `albert-base-v2/`, `datasets/` — symlinks to model weights and data

## Logs

Output is written to `logs/{dataset}.log`.
