# Optuna layout — My_creation (supplement to ../CLAUDE.md)

Details for `logs/optuna/4090D_restart/` and study resume semantics.

## Layout (4090D restart)

- **Regression**: `phase1|phase2|phase3|phase4/` per dataset drivers (`run/*.log`, `db/*.db`, `train_logs/`).
- **MOSI micro-local**: `phase4_mosi/` (and optionally `phase5_mosi/`): single-study TPE, `--disable_two_stage_mosi`, caps via `MOSI_N_EPOCHS_CAP`, `MOSI_EARLY_STOP_PATIENCE`.
- **Classification**: `classification/<RUN_TAG>/` with its own `db/`, `run/`, `train_logs/`.

## Resume: `--n_trials` is a **target total**

In `optuna_search_v2.py`, `optimize_with_cleanup` sets  
`remaining = max(0, n_trials - existing_finished)`.  
To add **N** new trials, set `--n_trials` to **current finished count + N** (or use the wrapper env vars documented in `run_optuna_4090d_restart.sh`).

## MOSI study names / distributions

- Legacy study: `infogate_mosi_phase4_mosi_4090d` (older trials may use **categorical** distributions for some hyperparameters).
- New-space study: `infogate_mosi_phase4_mosi_4090d_space2` (aligned with current **int/float** suggestions in `optuna_search_v2.py`).

Resuming the **legacy** study with **current** code can trigger Optuna  
`Cannot set different distribution kind to the same parameter name.`  
Prefer continuing **`…4090d_space2`** (or a fresh study suffix) when distributions changed; see script comments and `MOSI_STUDY_SUFFIX` / `MOSI_STUDY_NAME`.

## Reading results

- Driver: grep/tail `Best is trial … with value:` in `**/run/*.log`.
- Test metrics: open `train_logs/*trial_<id>.log` and search `Best Results`.
