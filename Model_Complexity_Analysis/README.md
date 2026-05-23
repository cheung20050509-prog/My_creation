# Model complexity analysis (fixed_experiment)

Case study for the three frozen Optuna configurations documented in [`../fixed_experiment/README.md`](../fixed_experiment/README.md):

| Case | Hyperparameters module | Backbone |
|------|------------------------|----------|
| MOSI trial 234 | [`../fixed_experiment/mosi_trial234_hparams.py`](../fixed_experiment/mosi_trial234_hparams.py) | DeBERTa-v3 |
| MOSEI phase1 trial 70 | [`../fixed_experiment/mosei_phase1_trial70_hparams.py`](../fixed_experiment/mosei_phase1_trial70_hparams.py) | DeBERTa-v3 |
| SIMSv2 phase4 trial 52 | [`../fixed_experiment/simsv2_phase4_trial52_hparams.py`](../fixed_experiment/simsv2_phase4_trial52_hparams.py) | BERT-base-Chinese |

## Python environment

Use conda env **`ITHP5090`** (same default as [`run_optuna_4090d_restart.sh`](../run_optuna_4090d_restart.sh): PyTorch, Transformers, Optuna stack). The driver should also be started with **`ITHP5090`** (see `run_measure.sh` or the explicit `python` path below). Per-case **worker** subprocesses default to the same **`ITHP5090`** interpreter inside `measure_fixed_cases.py` unless you set **`MODEL_COMPLEXITY_PYTHON`** to another `python` binary.

## Dependencies

Inside **`ITHP5090`**, install one FLOP counting library (once):

```bash
conda activate ITHP5090
pip install fvcore
# or: pip install thop
```

`fvcore` is preferred; if it is not installed, the script tries `thop`, then omits FLOPs with a short reason.

## How to run

Working directory must allow imports and data paths like the frozen trainers (same as `train_fixed_*.py`).

```bash
cd /path/to/My_creation
# Recommended: wrapper pins ITHP5090 (override with PYTHON=... if needed)
chmod +x Model_Complexity_Analysis/run_measure.sh
./Model_Complexity_Analysis/run_measure.sh --cases mosi,mosei,simsv2 --output Model_Complexity_Analysis/results.md
```

Equivalent explicit call:

```bash
cd /path/to/My_creation
/root/autodl-tmp/anaconda3/envs/ITHP5090/bin/python -u Model_Complexity_Analysis/measure_fixed_cases.py
```

Optional:

- **`MODEL_COMPLEXITY_PYTHON`** — path to `python` for per-case worker subprocesses (defaults to `ITHP5090` inside `measure_fixed_cases.py`).
- `--output PATH` — write a Markdown table (e.g. `Model_Complexity_Analysis/results.md`).
- `--memory-full-accum` — report peak GPU memory after `gradient_accumulation_step` micro-batch backward passes **without** `optimizer.step()` (closer to worst-case peak within an accumulation window).
- `--skip-epoch-time` — skip full `train_epoch` timing (faster smoke test).

## Metric definitions

- **Parameters / trainable** — `numel()` over `model.parameters()` and over parameters with `requires_grad` (same spirit as [`fixed_experiment/train.py`](../fixed_experiment/train.py) startup logs).
- **FLOPs** — approximate count for **one training forward** at micro-batch shape: `train_batch_size × max_seq_length` (frozen argv does not override `max_seq_length`; default is **50** in `train.py`), `stage=1`, with real-valued labels matching logits shape. Custom heads, gating, and some Transformer ops are only partially supported by generic counters; treat numbers as **comparable estimates**, not audit-grade totals.
- **Peak GPU memory (default)** — `torch.cuda.max_memory_allocated()` after **one** forward+backward at micro-batch size `train_batch_size`, with the same loss as training (`L1 + mse_weight * MSE` on predictions + IB term), including loss scaling by `gradient_accumulation_step` as in `train_epoch`. Does not include optimizer state unless you extend the script; it reflects activations + gradients for that step.
- **Per-epoch training time** — wall time for **one full** `train_epoch` over the real training `DataLoader`, `stage=1`, including optimizer steps, with `torch.cuda.synchronize()` before/after timing when CUDA is used. First epoch in real training may use EMA only after `ema_start_epoch`; here EMA is disabled (`ema=None`) for a stable complexity readout.

## Outputs

The driver prints a Markdown table to stdout and, with `--output`, writes the same table to a file. Each dataset runs in a **separate subprocess** so `fixed_experiment/train.py` parses the correct CLI once per case.
