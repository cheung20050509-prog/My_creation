# InfoGate on no_highway

This directory contains the current InfoGate implementation used in the `no_highway` branch.
It targets multimodal sentiment analysis on CMU-MOSI and CMU-MOSEI with text, acoustic,
and visual inputs.

Compared with the earlier experimental code, this branch keeps the main InfoGate fusion
pipeline but simplifies runtime behavior:

- complete-modality training and evaluation are the supported path
- the older missing-modality evaluation path is not exposed here
- the streamlined forward path is easier to retrain, test, and compare

The code still keeps the bottleneck-based regularization pieces, including CRA translators,
but `test.py` evaluates the standard complete-modality setting.

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
- `gamma_cyc=1.0`
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
    --gamma_cyc 1.0 \
    --alpha_ib 0.005 \
    --alpha_nce 0.05 \
    --mse_weight 0.5 \
    --cra_layers 8 \
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
