# BERT Optuna launchers

This folder does **not** duplicate `infogate_modules.py` or `optuna_search_v2.py`. It only holds
launch scripts and logs for runs that use:

- `My_creation/bert_infogate.py` (text backbone = BERT)
- `My_creation/optuna_search_v2.py` (shared search driver; includes `align_mix_floor` in Tier 3)

Download weights once:

```bash
bash My_creation/bert_optuna/download_bert_base_uncased.sh
```

## Scripts

| Script | Purpose |
|--------|---------|
| `run_bert_optuna_phase1_aligned.sh` | Legacy phase1 two-stage MOSI + single-stage MOSEI (`--search_tier 1`, `--no_dataset_overrides`) |
| `run_bert_new_space_align_floor.sh` | **New** single-stage Tier-3 search for `align_mix_floor` under `logs/new_space_search/regression/` |

## `align_mix_floor` search (BERT)

```bash
export MOSI_GPU=1 MOSEI_GPU=0   # physical GPUs; each driver sees `CUDA_VISIBLE_DEVICES` as GPU 0
bash My_creation/bert_optuna/run_bert_new_space_align_floor.sh
```

Outputs:

- DB: `bert_optuna/logs/new_space_search/regression/db/`
- Logs: `bert_optuna/logs/new_space_search/regression/run/*.log`, `train_logs/`
- Checkpoints: `bert_optuna/logs/new_space_search/regression/checkpoints/`

Warm-start (no old `.db` in repo): trial **121 / 220 / 234** (MOSI) and **70** (MOSEI) from frozen hparam modules, with `align_mix_floor=0.3`. If you have prior BERT Optuna DBs on disk:

```bash
export ENQUEUE_TOP_MOSI="sqlite:////abs/path/to/bert_optuna/logs/phase1/db/mosi.db"
export ENQUEUE_TOP_MOSEI="sqlite:////abs/path/to/bert_optuna/logs/phase1/db/mosei.db"
```

`SEED_ONLY=1` runs enqueue only. `SERIAL=1` runs MOSI then MOSEI on the same GPU schedule.

## vs DeBERTa `new_space_search`

| | DeBERTa (main) | BERT (this folder) |
|--|----------------|-------------------|
| Launcher | `logs/optuna/4090D_restart/new_space_search/run_new_space_align_floor.sh` | `bert_optuna/run_bert_new_space_align_floor.sh` |
| Text backbone | `deberta-v3-base` (default) | local `bert-base-uncase` via `--pretrained_model` |
| Study names | `infogate_*_new_space_align_floor_tpe` | `infogate_*_bert_new_space_align_floor_tpe` |

Do not mix the two DB trees unless you intend to compare backbones fairly.
