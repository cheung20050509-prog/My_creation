# InfoGate on no_highway

This directory contains the current InfoGate implementation used in the `no_highway` branch.
It targets multimodal sentiment analysis on CMU-MOSI and CMU-MOSEI with text, acoustic,
and visual inputs.

Compared with the earlier experimental code, this branch keeps the main InfoGate fusion
pipeline but simplifies runtime behavior:

- complete-modality training and evaluation are the supported path
- the older missing-modality evaluation path is not exposed here
- the streamlined forward path is easier to retrain, test, and compare

The code still keeps the bottleneck-based regularization pieces (token-level IB
plus label-level IB), and `test.py` evaluates the standard complete-modality setting.

## Overview

The model combines five pieces:

1. DeBERTa-v3-base text encoder
2. unimodal projectors for text, audio, and vision
3. IB encoders that produce bottleneck features plus confidence estimates
4. MSelector for dynamic primary-modality selection
5. InfoGate fusion layers with confidence-aware attention and adaptive gating

High-level flow:

```text
text/audio/vision
    -> projection to unified hidden size
    -> IB encoders -> bottleneck features + confidence
    -> MSelector dynamically chooses the primary stream (supervised by selective KL divergence on modality quality)
    -> InfoGate cross-attention fuses the two auxiliary streams into the primary stream
    -> The enhanced primary bottleneck stream alone is passed to the prediction head for sentiment scoring
```

## What the Branch Actually Supports

- `train.py`: two-stage training with EMA evaluation and dev-score checkpoint selection
- `test.py`: complete-modality evaluation only
- `train.sh`: default background launcher using the branch's current baseline config
- `test.sh`: convenience wrapper for evaluation
- `reproduce.sh`: tuned MOSI-only reproduction command for an older best-MOSI setup

Important scope notes:

- **The branch name is `no_highway`**, signifying a strict adherence to the Primary-centric philosophy of the MODS paper. The code path explicitly disables any direct concatenation (highway) of textual residual features or auxiliary modalities at the final prediction head, isolating the judgment entirely to the enhanced primary stream to prevent noise bleed.
- The README previously described pending MOSEI and missing-modality work; that is now outdated.
- Current evaluation here is for complete text-audio-vision inputs.

## Current Model Behavior

### Fusion

- IB confidence modulates attention to suppress uncertain auxiliary tokens.
- MSelector chooses the primary modality dynamically for each sample based on sample-level divergence.
- Adaptive gates control how much auxiliary information is injected into the primary stream.
- **Pure Primary-centric Prediction**: The fused bottleneck stream is passed through a LayerNorm and direct classification head without any textual feature expansion or highway concatenation.

### Training

- Stage 1 trains the task objective and bottleneck losses, allowing the primary stream to warm up.
- Stage 2 adds the translation, cyclic regularization terms, and selective KL supervision for the dynamic MSelector.
- Checkpoints are selected by dev score:

```text
dev_score = dev_mae - 0.5 * dev_corr
```

- The best checkpoint is tracked only after the midpoint of training.

## Latest Complete-Modality Results (Dynamic MODS Aligned)

The latest local rerun on this branch was completed on 2026-04-03 using the
strictly MODS-aligned dynamic primary architecture without a highway.

| Dataset | Best Acc2 | Best Acc7 | Best MAE | Best Corr | Best F1 | status |
|---|---:|---:|---:|---:|---:|---:|
| MOSI  | 87.94% | 51.76% | **0.6048** | 0.8540 | 0.8793 | New Low MAE (SOTA level) |
| MOSEI | ~87.21% | ~47.60% | ~0.5989 | ~0.8137 | ~0.8711 | *Training in progress* |

These results were produced from local runs whose logs were written to:

- `logs/full_dynamic_mods_aligned_20260403/train_mosi_full.log`
- `logs/full_dynamic_mods_aligned_20260403/train_mosei_full.log`

and whose best checkpoints were written to:

- `checkpoints_completeonly_20260402/infogate_mosi_best.pt`
- `checkpoints_completeonly_20260402/infogate_mosei_best.pt`

## Literature baselines (SOTA comparison)

The tables below mirror `baseline_table.tex` (compiled into the paper). **†** marks
numbers cited from the respective papers as in that table. The **InfoGate (Ours)**
row is left as placeholders (`—`) until you paste final PRISM/InfoGate numbers.

### CMU-MOSI and CMU-MOSEI

| Baseline | MOSI Acc7↑ | MOSI Acc2↑ | MOSI F1↑ | MOSI MAE↓ | MOSI Corr↑ | MOSEI Acc7↑ | MOSEI Acc2↑ | MOSEI F1↑ | MOSEI MAE↓ | MOSEI Corr↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MFM | 33.3 | 80.0 | 80.1 | 0.948 | 0.664 | 50.8 | 83.4 | 83.4 | 0.580 | 0.722 |
| Self-MM | 45.8 | 84.9 | 84.8 | 0.731 | 0.785 | 53.0 | 85.2 | 85.2 | 0.540 | 0.763 |
| AtCAF† | 46.5 | 88.6 | 88.5 | 0.650 | 0.831 | 55.9 | 87.0 | 86.8 | 0.508 | 0.785 |
| DLF† | 47.1 | 85.1 | 85.0 | 0.731 | 0.781 | 53.9 | 85.4 | 85.3 | 0.536 | 0.764 |
| KuDA† | 47.1 | 86.4 | 86.5 | 0.705 | 0.795 | 52.9 | 86.5 | 86.6 | 0.529 | 0.776 |
| DEVA† | 46.3 | 86.3 | 86.3 | 0.730 | 0.787 | 52.3 | 86.1 | 86.2 | 0.541 | 0.769 |
| C-MIB | 47.7 | 87.8 | 87.8 | 0.662 | 0.835 | 52.7 | 86.9 | 86.8 | 0.542 | 0.784 |
| ITHP | 47.7 | 88.5 | 88.5 | 0.663 | 0.856 | 52.2 | 87.1 | 87.1 | 0.550 | 0.792 |
| Multimodal Boosting | 49.1 | 88.5 | 88.4 | 0.634 | 0.855 | 54.0 | 86.5 | 86.5 | 0.523 | 0.779 |
| CaMIB† | 48.0 | 89.8 | **89.8** | 0.616 | 0.857 | 53.5 | 87.3 | 87.2 | 0.517 | 0.788 |
| DMD | 44.9 | 84.3 | 84.3 | 0.726 | 0.788 | 52.8 | 84.6 | 84.6 | 0.538 | 0.768 |
| EMOE | 45.2 | 84.8 | 84.8 | 0.723 | 0.790 | 52.5 | 85.0 | 85.0 | 0.542 | 0.760 |
| TMSON† | 47.4 | 87.2 | 87.2 | 0.687 | 0.809 | 55.6 | 86.4 | 86.2 | 0.526 | 0.766 |
| MOAC† | 48.6 | 89.0 | 89.0 | **0.605** | **0.857** | 54.3 | **87.6** | **87.6** | 0.512 | **0.793** |
| **InfoGate (Ours)** | — | — | — | — | — | — | — | — | — | — |

### CH-SIMS v2

| Baseline | Acc5↑ | Acc3↑ | Acc2↑ | F1↑ | MAE↓ | Corr↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| EF-LSTM | 53.7 | 73.5 | 80.1 | 80.0 | 0.309 | 0.700 |
| LF-DNN | 51.8 | 71.2 | 77.8 | 77.9 | 0.322 | 0.668 |
| TFN | 53.3 | 70.9 | 78.1 | 78.1 | 0.322 | 0.662 |
| LMF | 51.6 | 70.0 | 77.8 | 77.8 | 0.327 | 0.651 |
| MFN | **55.4** | 72.7 | 79.4 | 79.4 | 0.301 | 0.712 |
| Graph-MFN | 48.9 | 68.6 | 76.6 | 76.6 | 0.334 | 0.644 |
| MISA | 47.5 | 68.9 | 78.2 | 78.3 | 0.342 | 0.671 |
| MAG-BERT | 49.2 | 70.6 | 77.1 | 77.1 | 0.346 | 0.641 |
| Self-MM | 53.5 | 72.7 | 78.7 | 78.6 | 0.315 | 0.691 |
| MMIM | 50.5 | 70.4 | 77.8 | 77.8 | 0.339 | 0.641 |
| AV-MC | 52.1 | 73.2 | 80.6 | 80.7 | 0.301 | 0.721 |
| KuDA† | 53.1 | **74.3** | 80.2 | 80.1 | **0.289** | **0.741** |
| **InfoGate (Ours)** | — | — | — | — | — | — |

**Bold** in baseline rows matches the LaTeX table emphasis in `baseline_table.tex` (best-in-column style marks).

## Repository Layout

```text
deberta_infogate.py      DeBERTa wrapper that attaches InfoGate
infogate_modules.py      IB encoders, MSelector, InfoGate layers, losses
train.py                 main training entry point
test.py                  complete-modality evaluation entry point
train.sh                 default nohup launcher
test.sh                  evaluation wrapper
reproduce.sh             tuned MOSI reproduction command
datasets/                expected location for mosi.pkl and mosei.pkl
deberta-v3-base/         local DeBERTa files used by from_pretrained
```

## Setup

### Requirements

```bash
pip install -r requirements.txt
```

### Data

Place the processed dataset files in `datasets/`:

- `datasets/mosi.pkl`
- `datasets/mosei.pkl`

### Backbone

Place the local DeBERTa-v3-base model files in `deberta-v3-base/`.

## Training

### Default branch configuration

This is the simplest way to launch training with the branch's current default setup:

```bash
./train.sh mosi
./train.sh mosei
```

Defaults in `train.sh`:

- `n_epochs=80`
- `stage1_epochs=8`
- `train_batch_size=16`
- `bottleneck_dim=128`
- `num_infogate_layers=3`
- `beta_ib=16`
- `alpha_ib=0.005`
- `dropout_prob=0.25`
- `seed=42`

The script launches `train.py` with `nohup` and writes logs under `logs/`.

### Direct training command

If you want full control over paths or hyperparameters, call `train.py` directly:

```bash
python train.py \
    --dataset mosei \
    --n_epochs 80 \
    --stage1_epochs 8 \
    --train_batch_size 16 \
    --gradient_accumulation_step 2 \
    --learning_rate 2e-5 \
    --ig_learning_rate 5e-4 \
    --unified_dim 256 \
    --ib_hidden_dim 256 \
    --bottleneck_dim 128 \
    --num_heads 4 \
    --num_infogate_layers 3 \
    --beta_ib 16 \
    --alpha_ib 0.005 \
    --mse_weight 0.5 \
    --dropout_prob 0.25 \
    --weight_decay 0.01 \
    --ema_decay 0.999 \
    --ema_start_epoch 5 \
    --checkpoint_dir checkpoints \
    --seed 42
```

### Tuned MOSI reproduction command

`reproduce.sh` keeps an older MOSI-specific configuration with a smaller bottleneck and
more InfoGate layers:

```bash
./reproduce.sh
```

That script is useful when you want to replay the earlier tuned MOSI setup rather than the
current branch defaults.

## Evaluation

### Default evaluation wrapper

```bash
./test.sh mosi checkpoints/infogate_mosi_best.pt
./test.sh mosei checkpoints/infogate_mosei_best.pt
```

### Evaluate the latest local complete-only rerun

```bash
python test.py --dataset mosi --checkpoint checkpoints_completeonly_20260402/infogate_mosi_best.pt
python test.py --dataset mosei --checkpoint checkpoints_completeonly_20260402/infogate_mosei_best.pt
```

`test.py` prints only the complete-modality result block in this branch.

## Practical Notes

- `*.pt`, `logs/`, and `checkpoints/` are not intended to be versioned.
- Older logs may contain `pred_std=nan` on a single-sample tail batch; this was a logging-only statistics issue and does not indicate training collapse.
- If you want to compare branch behavior, use the logs in `logs/completeonly_20260402/` as the latest clean reference for this branch.

## License

MIT
