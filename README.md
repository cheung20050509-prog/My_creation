# InfoGate: Information Bottleneck-Guided Adaptive Cross-Attention for Robust Multimodal Fusion

## 1. Introduction

Multimodal Sentiment Analysis (MSA) integrates language, acoustic, and visual modalities to predict sentiment intensity from video utterances. While recent methods based on cross-modal attention and information bottleneck (IB) compression have achieved strong results, they face key limitations: (1) cross-modal attention treats all auxiliary tokens equally, regardless of their information quality; (2) fusion strategies apply fixed injection weights without considering cross-modal consistency; (3) contrastive learning objectives (e.g., InfoNCE) focus on modality alignment but ignore inter-sample sentiment relationships.

We propose **InfoGate**, a framework that leverages IB-derived uncertainty signals to adaptively control every stage of multimodal fusion. Our core insight is that the confidence estimates from Information Bottleneck encoders — computed as `conf = sigmoid(-logvar)` — provide a natural, end-to-end learnable measure of information quality that can guide modality selection, cross-modal attention, and fusion intensity.

## 2. Methods

### 2.1 Architecture Overview

```
text → DeBERTa → proj(768→256) → IBEncoder → bottleneck(96d) + confidence
acoustic → proj(74→256) → IBEncoder → bottleneck(96d) + confidence    
visual → proj(47→256) → IBEncoder → bottleneck(96d) + confidence
    ↓
MSelector (dynamic primary modality selection)
    ↓
InfoGateModule (IB-guided cross-attention + adaptive gating, ×4 layers)
    ↓
Alignment-modulated injection
    ↓
ITHP-style residual fusion with DeBERTa text → prediction
```

### 2.2 Key Components

**IB-Guided Cross-Attention.** Standard multi-head attention modified with two confidence-based modulations: (1) score bias `scores += scale * log(conf_K)` suppresses attention to low-confidence key positions; (2) value gating `V_eff = V * conf` reduces contribution of uncertain tokens. This naturally filters noise from auxiliary modalities at the token level.

**Alignment-Modulated Injection.** Before injecting auxiliary cross-attention output into the primary modality, we compute the cosine similarity between primary and auxiliary bottleneck representations. When modalities agree (high alignment), injection proceeds normally; when they contradict (low alignment, e.g., sarcasm), injection is suppressed. Floor at 0.3 prevents complete information cutoff.

**Adaptive Information Gate.** Per-sample, per-dimension gating that learns which dimensions of the cross-attention output to inject: `g = sigmoid(W2 * ReLU(W1 * [primary || ca_output]))`, `output = g * ca_output`.

**Dynamic Primary Modality Selection (MSelector).** Adopted from MODS (AAAI 2026). Adaptive aggregation + MLP assigns soft weights to each modality per sample, determining which modality leads the fusion process.

### 2.3 Training Objectives

| Loss | Formula | Role |
|------|---------|------|
| L_task | L1 + 0.67*MSE | Regression prediction |
| L_tib | KL + β*reconstruction (cyclic, 9 decoders) | Token-level IB: intra/inter-modal bottleneck regularization |
| L_lib | KL + β*label_prediction (per modality) | Label-level IB: task-aware bottleneck supervision |
| L_tran | translation_MSE + cyclic_MSE (CRA) | Cross-modal bottleneck alignment regularization |
| L_sac | MSE(cosine_sim, exp(-\|y_i-y_j\|)) | Sentiment-aware contrastive learning in bottleneck space |

**Two-stage training:** Stage 1 (epochs 1–12) trains without L_tran to stabilize IB encoders; Stage 2 (epochs 13–80) adds L_tran for cross-modal alignment.

### 2.4 Theoretical Motivation

The Information Bottleneck principle compresses input X into bottleneck B by minimizing I(X;B) while maximizing I(B;Y). The logvar from the variational approximation naturally indicates per-token uncertainty. We repurpose this uncertainty as a universal control signal:

- **Selection stage:** MSelector uses bottleneck features (already IB-compressed) for modality importance estimation
- **Attention stage:** IB confidence modulates cross-attention scores and values (token-level filtering)  
- **Injection stage:** Bottleneck alignment (cosine similarity) gates cross-modal injection (sample-level filtering)

This creates a three-level filtering hierarchy: global (MSelector) → token (IBGuidedAttention) → sample (alignment modulation).

## 3. Results

### CMU-MOSI (dev-score checkpoint selection)

| Method | Acc7↑ | Acc2↑ | F1↑ | MAE↓ | Corr↑ |
|--------|-------|-------|-----|------|-------|
| ITHP (ICLR 2024) | 47.7 | 88.5 | 88.5 | 0.663 | 0.856 |
| CaMIB (ICLR 2026 sub.) | 48.0 | 89.8 | 89.8 | 0.616 | 0.857 |
| MOAC (WWW 2025) | 48.6 | 89.0 | 89.0 | 0.605 | 0.857 |
| Multimodal Boosting | 49.1 | 88.5 | 88.4 | 0.634 | 0.855 |
| **InfoGate (Ours)** | **51.15** | 88.09 | 88.06 | **0.5977** | **0.8629** |

**State-of-the-art on MAE, Acc7, and Corr.** Competitive on Acc2 and F1.

## 4. Discussion

### 4.1 Contributions

1. **IB-Guided Cross-Attention:** Using IB uncertainty to modulate attention scores and values, providing principled token-level noise filtering in cross-modal interaction.
2. **Alignment-Modulated Injection:** Leveraging bottleneck-space cosine similarity as a sample-level consistency check before cross-modal injection, addressing the "two-sided" nature of auxiliary modalities.
3. **Sentiment-Aware Contrastive Learning (L_sac):** Structuring the bottleneck space according to inter-sample sentiment distance, complementing the intra-sample cross-modal alignment achieved by cyclic IB.

### 4.2 Ablation Evidence

| Configuration | MAE | Acc7 |
|--------------|-----|------|
| Full InfoGate | **0.5977** | **51.15%** |
| w/o Alignment Modulation | 0.6093 | 48.85% |
| w/o L_sac | 0.6061 | 48.55% |
| w/o L_tran (CRA) | 0.6099 | 49.77% |

### 4.3 Limitations

- Acc2/F1 (88.09%/88.06%) still below CaMIB (89.8%) and MOAC (89.0%), primarily due to the model's stronger regression focus over binary classification.
- Evaluated only on MOSI; MOSEI evaluation pending.
- The model contains legacy parameters (reverse_proj) from an earlier InfoNCE design that cannot be removed without disrupting the initialization chain.

### 4.4 Future Work

- Evaluation on CMU-MOSEI and missing-modality protocols (CRA infrastructure already in place).
- Segment-level modality selection for longer sequences.
- Investigation of the initialization sensitivity issue in HuggingFace-based models.

## Setup

### Requirements
```bash
pip install -r requirements.txt
```

### Data
Place `mosi.pkl` / `mosei.pkl` in `datasets/` (or symlink).

### Model
Place DeBERTa-v3-base files in `deberta-v3-base/`.

### Train
```bash
python train.py --dataset mosi --n_epochs 80 --stage1_epochs 12 \
    --train_batch_size 16 --gradient_accumulation_step 2 \
    --learning_rate 1.14e-5 --ig_learning_rate 1.93e-4 \
    --bottleneck_dim 96 --num_infogate_layers 4 \
    --beta_ib 15.6 --gamma_cyc 0.582 --alpha_ib 0.00227 \
    --alpha_sac 0.02 --mse_weight 0.67 \
    --dropout_prob 0.195 --weight_decay 0.005 --seed 42
```

### Test
```bash
python test.py --dataset mosi --checkpoint checkpoints/infogate_mosi_best.pt
```

## License

MIT
