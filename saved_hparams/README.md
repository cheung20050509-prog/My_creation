# Saved hyperparameter snapshots

Use this folder to keep **reproducible exports** of best Optuna trials **before** changing search spaces or restarting long runs.

## Contents

- `mosi_optuna_relaunch_20260415_120746_best_hparams.json` — Stage1/Stage2 best trials (dev MAE), full `params`, and DB paths.
- `archives/` — byte-for-byte **SQLite copies** of the corresponding studies at export time (`*_backup.db`). These are backups; live training DBs remain under `logs/<RUN_TAG>/db/`.

## Restore / reproduce

1. Best dev-MAE configs are in the JSON under `best_stage1` / `best_stage2` and `global_best_*`.
2. To re-run training with a fixed config, pass the `params` fields to `train.py` (same flags as Optuna uses) or load the matching checkpoint under `logs/<RUN_TAG>/checkpoints/...`.

## Git

`*.db` is gitignored globally; **JSON snapshots are intended to be committed**. SQLite backups under `archives/` are usually **not** committed (large binaries) — keep them on disk or sync separately.
