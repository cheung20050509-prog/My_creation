# new_space_search — `align_mix_floor` single-stage Tier-3 studies

Launchers:

- `run_new_space_align_floor.sh` — all five datasets **sequentially** (set `GPU`, optional `ENQUEUE_TOP_*`).
- `run_simsv2_new_space_align_floor.sh` — **SIMSv2 only**, for parallel use on the same GPU as MOSI/MOSEI (`FRESH=1` wipes DB/logs/checkpoints and restarts).

## Layout

- `regression/{db,train_logs,checkpoints,run}` — MOSI, MOSEI, SIMSv2 (`optuna_search_v2.py`)
- `classification/{db,train_logs,checkpoints,run}` — MUStARD, UR-FUNNY (`optuna_search_classify.py`)

Each study uses `--search_tier 3`, `--micro_refine none` (regression), MOSI adds `--disable_two_stage_mosi`, MUStARD adds `--disable_two_stage`.

## Warm-start (optional)

Point at **previous** Optuna SQLite DBs with completed trials (comma-separated URIs), e.g.:

```bash
export ENQUEUE_TOP_MOSI="sqlite:////abs/path/to/old_mosi.db"
export ENQUEUE_TOP_MOSEI="sqlite:////abs/path/to/old_mosei.db"
export ENQUEUE_TOP_MUSTARD="sqlite:////abs/path/to/old_mustard.db"
export ENQUEUE_TOP_UR_FUNNY="sqlite:////abs/path/to/old_ur_funny.db"
export ENQUEUE_TOP_K=10
```

**SIMSv2:** do **not** enqueue legacy SIMSv2 trials (`ENQUEUE_TOP_SIMSV2` should stay unset) — MMSA/KuDA data protocol differs from old runs.

Gold / paper trials can be added with Optuna’s `--enqueue_trials_storage` / `--enqueue_trials_study` / `--enqueue_trials_numbers` on a one-off command line if needed (see `optuna_search_v2.py` / `optuna_search_classify.py` help).

## Trial budgets (script defaults)

| Dataset   | `--n_trials` |
|-----------|--------------|
| MOSI      | 80           |
| MOSEI     | 60           |
| SIMSv2    | 100          |
| MUStARD   | 60           |
| UR-FUNNY  | 60           |

Adjust in the shell script if you need more budget.
