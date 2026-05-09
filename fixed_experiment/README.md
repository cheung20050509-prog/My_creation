# Fixed experiment: MOSI Optuna trial 234

Single training run with hyperparameters frozen to **study `infogate_mosi_phase4_mosi_4090d`, trial 234** (CMU-MOSI). CLI float formatting matches `optuna_search_v2.objective()` so parsed learning rates match the gold Optuna train log header.

## Layout

- [`mosi_trial234_hparams.py`](mosi_trial234_hparams.py) — canonical params + `build_train_argv()`.
- [`train_fixed_mosi_trial234.py`](train_fixed_mosi_trial234.py) — invokes **this folder’s** [`train.py`](train.py) via subprocess with `cwd` = parent `My_creation/` (for HF caches and any cwd-relative behavior).
- [`run_mosi_trial234.sh`](run_mosi_trial234.sh) — shell entry with `CUDA_VISIBLE_DEVICES`, `PYTORCH_CUDA_ALLOC_CONF`, log tee.

Outputs default to `My_creation/fixed_experiment/runs/mosi_trial234/` (`train.log`, `checkpoints/`).

## Vendored Python (snapshot)

These files are **copies** of the corresponding modules under [`My_creation/`](../); imports resolve within `fixed_experiment/`. The copied [`train.py`](train.py) patches paths so **`deberta-v3-base`** and **`datasets/{mosi,mosei,simsv2}.pkl`** are read from **`My_creation/`**, not from this directory.

| File | Role |
|------|------|
| [`train.py`](train.py) | Entry training script (path patches for snapshot layout) |
| [`deberta_infogate.py`](deberta_infogate.py) | DeBERTa + InfoGate |
| [`bert_infogate.py`](bert_infogate.py) | BERT path (e.g. SIMSv2) |
| [`infogate_modules.py`](infogate_modules.py) | Shared InfoGate blocks |
| [`global_configs.py`](global_configs.py) | Dataset dims / device |
| [`simsv2_metrics.py`](simsv2_metrics.py) | SIMSv2 metrics |
| [`selection_utils.py`](selection_utils.py) | Checkpoint selection helpers |

If you change training logic under `My_creation/`, **refresh these copies** when you want this frozen bundle to stay aligned (for example: `cp ../train.py ../deberta_infogate.py ... fixed_experiment/` then re-apply the small path patches in `fixed_experiment/train.py`).

## Run

From anywhere:

```bash
bash /path/to/My_creation/fixed_experiment/run_mosi_trial234.sh
```

Or:

```bash
cd My_creation/fixed_experiment && ./run_mosi_trial234.sh
```

Ensure `run_mosi_trial234.sh` is executable (`chmod +x run_mosi_trial234.sh`).

Dry-run (print command only):

```bash
cd My_creation && python fixed_experiment/train_fixed_mosi_trial234.py --dry-run
```

## Optional check

From `My_creation`:

```bash
python scripts/verify_mosi_trial234_optuna_train_argv.py
```

Confirms objective-style argv produces the same backbone / InfoGate LR banner lines as `mosi_phase4_mosi_trial_234.log`.

## Relation to other scripts

- Same hyperparameters as [`run_reproduce_mosi_phase4_mosi_trial234.sh`](../run_reproduce_mosi_phase4_mosi_trial234.sh); this folder uses a dedicated output path under `fixed_experiment/runs/` and the **local** `train.py` snapshot above.
