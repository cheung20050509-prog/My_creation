# Ablation study: CMU-MOSI (trial 234 / 220), CMU-MOSEI trial 70 & CH-SIMS v2 phase4 trial 52

Prepared by mirroring [`fixed_experiment/`](../fixed_experiment/) into this directory; launcher paths use `ablation_study/` so runs stay isolated under `ablation_study/runs/`. CLI float formatting matches `optuna_search_v2.objective()` so parsed learning rates match the gold Optuna train log headers.

**Environment:** use conda env **`ITHP5090`** with **`transformers==4.29.2`** (same stack as `fixed_experiment`). DeBERTa loads via `from_pretrained` on `My_creation/deberta-v3-base` (no meta-tensor workarounds). **Parallel:** `run_prism_ablations_*.sh` use `GPU_LIST` and optional `JOBS_PER_GPU`. **Default:** one training process per listed GPU per wave (`1,1,…` on multi-GPU); each wave runs in parallel, then the next wave starts. Set `JOBS_PER_GPU=2,1` (etc.) only if you intentionally want multiple jobs on one card.

## CMU-MOSI — Optuna trial 234

Single run (and PRISM ablations) frozen to **study `infogate_mosi_phase4_mosi_4090d`, trial 234**.

| Entry | Role |
|------|------|
| [`mosi_trial234_hparams.py`](mosi_trial234_hparams.py) | Params + `build_train_argv()` |
| [`train_fixed_mosi_trial234.py`](train_fixed_mosi_trial234.py) | Python launcher → this folder’s [`train.py`](train.py), `cwd` = `My_creation/` |
| [`run_mosi_trial234.sh`](run_mosi_trial234.sh) | Shell entry (`CUDA_VISIBLE_DEVICES`, log tee) |

Outputs: `My_creation/ablation_study/runs/mosi_trial234/` (`train.log`, `checkpoints/`).

## CMU-MOSI — Optuna trial 220

Frozen to **study `infogate_mosi_phase4_mosi_4090d`, trial 220** ([`4090D_restart/phase4_mosi`](../logs/optuna/4090D_restart/phase4_mosi)). Optuna-reported objective **0.5947** (dev-selected MAE). Same `batch_config` tier-3 grid as trial 234 (`train_batch_size` 16, `gradient_accumulation_step` 2, `early_stop_patience` 15). Selection line in gold log: **Best Results (mae, epoch >= 7)** (`stage1_epochs=6`).

| Entry | Role |
|------|------|
| [`mosi_trial220_hparams.py`](mosi_trial220_hparams.py) | Params + `build_train_argv(..., ablation=)` |
| [`train_fixed_mosi_trial220.py`](train_fixed_mosi_trial220.py) | Python launcher |
| [`run_mosi_trial220.sh`](run_mosi_trial220.sh) | Single-run shell (`--ablation none`) |

Outputs: `My_creation/ablation_study/runs/mosi_trial220/`.

Gold Optuna log: [`logs/optuna/4090D_restart/phase4_mosi/train_logs/mosi_phase4_mosi_trial_220.log`](../logs/optuna/4090D_restart/phase4_mosi/train_logs/mosi_phase4_mosi_trial_220.log).

```bash
python scripts/verify_mosi_trial220_optuna_train_argv.py
```

## CMU-MOSEI — 4090D_restart phase1 trial 70

Frozen to **study `infogate_mosei_phase1_4090d`, trial 70** (Tier 1). Optuna-reported test MAE **0.4994** matches the paper table MOSEI MAE **0.499** (rounding). No `--early_stop_patience` in argv (matches phase1 driver). Selection line in logs: **Best Results (mae, epoch >= 11)** (after `stage1_epochs=10`).

| Entry | Role |
|------|------|
| [`mosei_phase1_trial70_hparams.py`](mosei_phase1_trial70_hparams.py) | Params + `build_train_argv(..., ablation=)` — keep numeric core in sync with [`fixed_experiment/mosei_phase1_trial70_hparams.py`](../fixed_experiment/mosei_phase1_trial70_hparams.py) |
| [`train_fixed_mosei_phase1_trial70.py`](train_fixed_mosei_phase1_trial70.py) | Python launcher |
| [`run_mosei_phase1_trial70.sh`](run_mosei_phase1_trial70.sh) | Single-run shell (`--ablation none`) |

Outputs: `My_creation/ablation_study/runs/mosei_phase1_trial70/`.

MOSEI ablation t-SNE over checkpoint `h_p` (facet or joint): [`../visualize/tsne_mosei_ablation.py`](../visualize/tsne_mosei_ablation.py).

Gold Optuna log: [`logs/optuna/4090D_restart/phase1/train_logs/mosei_phase1_trial_70.log`](../logs/optuna/4090D_restart/phase1/train_logs/mosei_phase1_trial_70.log).

From `My_creation`, argv check (validates **fixed_experiment** file; ablation `none` here should produce the same tokens):

```bash
python scripts/verify_mosei_phase1_trial70_optuna_train_argv.py
```

## CH-SIMS v2 — Optuna 4090D_restart phase4 trial 52

Frozen to **study `4090D_restart/phase4` SIMSv2**, **trial 52** (paper row; same numeric knobs as phase6 space3 trial 0). Same numeric CLI as [`fixed_experiment/simsv2_phase4_trial52_hparams.py`](../fixed_experiment/simsv2_phase4_trial52_hparams.py): train batch **8** × grad accum **4**, **`--selection_metric mae`**, **`--early_stop_patience 15`**. Selection line in gold log: **Best Results (mae, epoch >= 6)** (`stage1_epochs=5`).

| Entry | Role |
|------|------|
| [`simsv2_phase4_trial52_hparams.py`](simsv2_phase4_trial52_hparams.py) | Params + `build_train_argv(..., ablation=)` |
| [`train_fixed_simsv2_phase4_trial52.py`](train_fixed_simsv2_phase4_trial52.py) | Python launcher |
| [`run_simsv2_phase4_trial52.sh`](run_simsv2_phase4_trial52.sh) | Single-run shell (`--ablation none`; **`CUDA_VISIBLE_DEVICES` defaults to 1**) |

Outputs: `My_creation/ablation_study/runs/simsv2_phase4_trial52/`.

Gold Optuna log: [`logs/optuna/4090D_restart/phase4/train_logs/simsv2_phase4_trial_52.log`](../logs/optuna/4090D_restart/phase4/train_logs/simsv2_phase4_trial_52.log).

From `My_creation`, argv check (validates **fixed_experiment** file; ablation `none` here should produce the same tokens):

```bash
python scripts/verify_simsv2_phase4_trial52_optuna_train_argv.py
```

## PRISM six-mode ablations

Batch scripts run `none`, `no_infogate`, `no_mselector`, `no_ib`, `no_conf_gating`, `no_adaptive_gate` into separate `runs/*` directories. For strict baseline certification vs [`fixed_experiment/`](../fixed_experiment/), do not stack **`none`** with another training job on the same GPU in the same wave (`JOBS_PER_GPU=1` on that card or run `none` alone).

| Dataset | Batch script |
|--------|----------------|
| MOSI trial 234 | [`run_prism_ablations_mosi_trial234.sh`](run_prism_ablations_mosi_trial234.sh) |
| MOSI trial 220 | [`run_prism_ablations_mosi_trial220.sh`](run_prism_ablations_mosi_trial220.sh) |
| MOSEI trial 70 | [`run_prism_ablations_mosei_phase1_trial70.sh`](run_prism_ablations_mosei_phase1_trial70.sh) |
| SIMSv2 phase4 trial 52 | [`run_prism_ablations_simsv2_phase4_trial52.sh`](run_prism_ablations_simsv2_phase4_trial52.sh) |

**Queue MOSI trial220 after MOSEI:** [`run_prism_ablations_mosei_phase1_trial70.sh`](run_prism_ablations_mosei_phase1_trial70.sh) writes its bash PID to [`runs/mosei_phase1_trial70_prism_master.pid`](runs/mosei_phase1_trial70_prism_master.pid) on startup. Start [`queue_mosi_trial220_after_mosei_prism.sh`](queue_mosi_trial220_after_mosei_prism.sh) (e.g. with `nohup`). It **blocks until that PID exits** — i.e. **MOSEI’s six PRISM ablations have finished all waves** — then runs [**`run_prism_ablations_mosi_trial220.sh`**](run_prism_ablations_mosi_trial220.sh) for **MOSI’s six PRISM modes** (`none`, `no_infogate`, …). **`JOBS_PER_GPU=1,1` by default** for MOSI unless you export another value. Override wait target with `MOSEI_MASTER_PID`, or `SKIP_MOSEI_WAIT=1` to run MOSI immediately (debug).

MOSI trial121 (space2) bundle: [`run_prism_ablations_mosi_space2_trial121.sh`](run_prism_ablations_mosi_space2_trial121.sh).

## Vendored Python (snapshot)

These files are **copies** of the corresponding modules under [`My_creation/`](../); imports resolve within `ablation_study/`. The copied [`train.py`](train.py) patches paths so **`deberta-v3-base`** and **`datasets/{mosi,mosei,simsv2}.pkl`** are read from **`My_creation/`**, not from this directory.

| File | Role |
|------|------|
| [`train.py`](train.py) | Entry training script (path patches for snapshot layout) |
| [`deberta_infogate.py`](deberta_infogate.py) | DeBERTa + InfoGate |
| [`bert_infogate.py`](bert_infogate.py) | BERT path (e.g. SIMSv2) |
| [`infogate_modules.py`](infogate_modules.py) | Shared InfoGate blocks |
| [`global_configs.py`](global_configs.py) | Dataset dims / device |
| [`simsv2_metrics.py`](simsv2_metrics.py) | SIMSv2 metrics |
| [`selection_utils.py`](selection_utils.py) | Checkpoint selection helpers |

If you change training logic under `My_creation/`, **refresh these copies** when you want this frozen bundle to stay aligned (for example: `cp ../train.py ../deberta_infogate.py ... ablation_study/` then re-apply the small path patches in `ablation_study/train.py`).

## Run (single baseline)

**MOSI**

```bash
bash /path/to/My_creation/ablation_study/run_mosi_trial234.sh
# or: cd My_creation/ablation_study && ./run_mosi_trial234.sh
```

**MOSI trial 220**

```bash
bash /path/to/My_creation/ablation_study/run_mosi_trial220.sh
```

**MOSEI**

```bash
bash /path/to/My_creation/ablation_study/run_mosei_phase1_trial70.sh
```

**CH-SIMS v2 phase4 trial 52**

```bash
bash /path/to/My_creation/ablation_study/run_simsv2_phase4_trial52.sh
```

Dry-run (print command only):

```bash
cd My_creation && python ablation_study/train_fixed_mosi_trial234.py --dry-run
cd My_creation && python ablation_study/train_fixed_mosi_trial220.py --dry-run
cd My_creation && python ablation_study/train_fixed_mosei_phase1_trial70.py --dry-run
cd My_creation && python ablation_study/train_fixed_simsv2_phase4_trial52.py --dry-run
```

## Optional check (argv vs gold logs)

From `My_creation`:

```bash
python scripts/verify_mosi_trial234_optuna_train_argv.py
python scripts/verify_mosi_trial220_optuna_train_argv.py
python scripts/verify_simsv2_phase4_trial52_optuna_train_argv.py
```

Confirms objective-style argv produces the same backbone / InfoGate LR banner lines as `mosi_phase4_mosi_trial_234.log` / `mosi_phase4_mosi_trial_220.log` / `simsv2_phase4_trial_52.log` (SIMSv2).

## Relation to other scripts

- **MOSI trial 234:** same hyperparameters as [`run_reproduce_mosi_phase4_mosi_trial234.sh`](../run_reproduce_mosi_phase4_mosi_trial234.sh); this folder uses `ablation_study/runs/` and the **local** `train.py` snapshot in this directory.
- **MOSI trial 220:** Optuna-only baseline ([`phase4_mosi/train_logs/mosi_phase4_mosi_trial_220.log`](../logs/optuna/4090D_restart/phase4_mosi/train_logs/mosi_phase4_mosi_trial_220.log)); no separate `fixed_experiment` launcher unless you add one later.
- **MOSEI trial 70:** same hyperparameters as [`fixed_experiment/run_mosei_phase1_trial70.sh`](../fixed_experiment/run_mosei_phase1_trial70.sh) / [`train_fixed_mosei_phase1_trial70.py`](../fixed_experiment/train_fixed_mosei_phase1_trial70.py), but training goes through `ablation_study/train.py` and PRISM `--ablation` when used from this tree.
- **CH-SIMS v2 phase4 trial 52:** same numeric knobs as [`fixed_experiment/simsv2_phase4_trial52_hparams.py`](../fixed_experiment/simsv2_phase4_trial52_hparams.py) / [`run_simsv2_phase4_trial52.sh`](../fixed_experiment/run_simsv2_phase4_trial52.sh); ablation launchers use `ablation_study/runs/` and this folder’s `train.py`.
