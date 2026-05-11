# Fixed experiments (CMU-MOSI, CMU-MOSEI & CH-SIMS v2)

Self-contained snapshots under [`fixed_experiment/`](.) invoke **this folder’s** [`train.py`](train.py) with `cwd` = [`My_creation/`](../). CLI float formatting matches [`optuna_search_v2.objective()`](../optuna_search_v2.py) so parsed learning rates match the gold Optuna train logs.

---

## CMU-MOSI — Optuna trial 234

Single run frozen to **study `infogate_mosi_phase4_mosi_4090d`, trial 234**.

| Entry | Role |
|------|------|
| [`mosi_trial234_hparams.py`](mosi_trial234_hparams.py) | Params + `build_train_argv()` |
| [`train_fixed_mosi_trial234.py`](train_fixed_mosi_trial234.py) | Python launcher |
| [`run_mosi_trial234.sh`](run_mosi_trial234.sh) | Shell entry (`CUDA_VISIBLE_DEVICES`, log tee) |

Outputs: `My_creation/fixed_experiment/runs/mosi_trial234/` (`train.log`, `checkpoints/`).

```bash
bash /path/to/My_creation/fixed_experiment/run_mosi_trial234.sh
# or: cd My_creation/fixed_experiment && ./run_mosi_trial234.sh
```

Dry-run:

```bash
cd My_creation && python fixed_experiment/train_fixed_mosi_trial234.py --dry-run
```

Check:

```bash
cd My_creation && python scripts/verify_mosi_trial234_optuna_train_argv.py
```

Same recipe as [`run_reproduce_mosi_phase4_mosi_trial234.sh`](../run_reproduce_mosi_phase4_mosi_trial234.sh).

---

## CMU-MOSEI — 4090D_restart phase1 trial 70

Single run frozen to **[`logs/optuna/4090D_restart/phase1`](../logs/optuna/4090D_restart/phase1)** study **`infogate_mosei_phase1_4090d`**, **Tier 1**, **trial 70**. Optuna-reported test MAE **0.4994** matches the paper table MOSEI MAE **0.499** (rounding). Phase1 search only sampled tier1 knobs; tier2/3 values are [`optuna_search_v2`](../optuna_search_v2.py) **`DEFAULTS`**.

**Note:** The separate Optuna relaunch line ([`saved_hparams/mosei_best_hparams.json`](../saved_hparams/mosei_best_hparams.json), trial 37, MAE ~0.494) is a different study — better MAE but not the 4090D_restart / paper-row lineage documented above.

| Entry | Role |
|------|------|
| [`mosei_phase1_trial70_hparams.py`](mosei_phase1_trial70_hparams.py) | Params + `build_train_argv()` (no `--early_stop_patience`; matches phase1 driver) |
| [`train_fixed_mosei_phase1_trial70.py`](train_fixed_mosei_phase1_trial70.py) | Python launcher |
| [`run_mosei_phase1_trial70.sh`](run_mosei_phase1_trial70.sh) | Shell entry |

Gold log: [`logs/optuna/4090D_restart/phase1/train_logs/mosei_phase1_trial_70.log`](../logs/optuna/4090D_restart/phase1/train_logs/mosei_phase1_trial_70.log).

Outputs: `My_creation/fixed_experiment/runs/mosei_phase1_trial70/`.

```bash
bash /path/to/My_creation/fixed_experiment/run_mosei_phase1_trial70.sh
# or: cd My_creation/fixed_experiment && ./run_mosei_phase1_trial70.sh
```

Dry-run:

```bash
cd My_creation && python fixed_experiment/train_fixed_mosei_phase1_trial70.py --dry-run
```

Check:

```bash
cd My_creation && python scripts/verify_mosei_phase1_trial70_optuna_train_argv.py
```

Optional DB merge:

```bash
cd My_creation && python scripts/verify_mosei_phase1_trial70_optuna_train_argv.py \
  --storage "sqlite:///$(pwd)/logs/optuna/4090D_restart/phase1/db/mosei.db"
```

PRISM six-mode ablations (same trial 70 knobs, local `ablation_study/train.py`): [`ablation_study/run_prism_ablations_mosei_phase1_trial70.sh`](../ablation_study/run_prism_ablations_mosei_phase1_trial70.sh).

---

## CH-SIMS v2 — 4090D_restart phase6_simsv2 trial 0

Single run frozen to **[`logs/optuna/4090D_restart/phase6_simsv2`](../logs/optuna/4090D_restart/phase6_simsv2)** study **`infogate_simsv2_phase6_simsv2_4090d_space3`**, **trial 0**. Optuna-reported selection value **0.3113**; driver uses **`--selection_metric mae`** and **`--early_stop_patience 15`** (see [`run_optuna_4090d_restart.sh`](../run_optuna_4090d_restart.sh) `phase6_simsv2`). `batch_config` 0 → train batch **8** × grad accum **4**.

| Entry | Role |
|------|------|
| [`simsv2_phase6_trial0_hparams.py`](simsv2_phase6_trial0_hparams.py) | Params + `build_train_argv()` (includes `--early_stop_patience 15`) |
| [`train_fixed_simsv2_phase6_trial0.py`](train_fixed_simsv2_phase6_trial0.py) | Python launcher |
| [`run_simsv2_phase6_trial0.sh`](run_simsv2_phase6_trial0.sh) | Shell entry — **`CUDA_VISIBLE_DEVICES` defaults to 1** |

Gold log: [`logs/optuna/4090D_restart/phase6_simsv2/train_logs/simsv2_phase6_simsv2_trial_0.log`](../logs/optuna/4090D_restart/phase6_simsv2/train_logs/simsv2_phase6_simsv2_trial_0.log).

Outputs: `My_creation/fixed_experiment/runs/simsv2_phase6_trial0/`.

PRISM six-mode ablations (same trial 0 knobs via [`ablation_study/train.py`](../ablation_study/train.py)): [`run_prism_ablations_simsv2_phase6_trial0.sh`](../ablation_study/run_prism_ablations_simsv2_phase6_trial0.sh).

```bash
bash /path/to/My_creation/fixed_experiment/run_simsv2_phase6_trial0.sh
# or: cd My_creation/fixed_experiment && ./run_simsv2_phase6_trial0.sh
# Override GPU: CUDA_VISIBLE_DEVICES=0 bash run_simsv2_phase6_trial0.sh
```

Dry-run:

```bash
cd My_creation && python fixed_experiment/train_fixed_simsv2_phase6_trial0.py --dry-run
```

Check:

```bash
cd My_creation && python scripts/verify_simsv2_phase6_trial0_optuna_train_argv.py
```

Optional DB merge:

```bash
cd My_creation && python scripts/verify_simsv2_phase6_trial0_optuna_train_argv.py \
  --storage "sqlite:///$(pwd)/logs/optuna/4090D_restart/phase6_simsv2/db/simsv2.db"
```

---

## Vendored Python (snapshot)

These files are **copies** of modules under [`My_creation/`](../); [`train.py`](train.py) patches paths so **`deberta-v3-base`** and **`datasets/{mosi,mosei,simsv2}.pkl`** resolve under **`My_creation/`**.

| File | Role |
|------|------|
| [`train.py`](train.py) | Training entry (path patches) |
| [`deberta_infogate.py`](deberta_infogate.py) | DeBERTa + InfoGate |
| [`bert_infogate.py`](bert_infogate.py) | BERT path (e.g. SIMSv2) |
| [`infogate_modules.py`](infogate_modules.py) | Shared InfoGate blocks |
| [`global_configs.py`](global_configs.py) | Dataset dims / device |
| [`simsv2_metrics.py`](simsv2_metrics.py) | SIMSv2 metrics |
| [`selection_utils.py`](selection_utils.py) | Checkpoint selection helpers |

After changing training code upstream, refresh copies and re-apply [`fixed_experiment/train.py`](train.py) path patches if needed.
