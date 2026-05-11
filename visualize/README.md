# Visualization utilities

Scripts here build figures from trained checkpoints; training code stays under [`../ablation_study/`](../ablation_study/).

## Dependencies

- **PyTorch** (same env as training, e.g. conda `ITHP5090`)
- **scikit-learn** (`TSNE`, optional `StandardScaler`)
- **matplotlib**

## MOSEI ablation t-SNE

[`tsne_mosei_ablation.py`](tsne_mosei_ablation.py) loads one or more InfoGate MOSEI checkpoints (`args` + `model_state_dict`), rebuilds the model with [`ablation_study/train.py`](../ablation_study/train.py), extracts pooled **`h_p`** on dev or test, runs t-SNE, and writes **PDF + PNG** (300 dpi) under `--outdir` (default: `visualize/figures/mosei_ablation_tsne/`).

In **facet** mode, points use a calm **viridis**-style continuous map on the regression label (2–98% color range), **serif** typography, **no axis ticks or t-SNE axis labels**, thin **black panel frames**, letter labels **(a), (b), …** centered above each panel, and a **horizontal** colorbar with **Negative** / **Positive** at the ends. `--palette` only affects **joint** mode (model-identity colors).

Run from **`My_creation`** so dataset pickles and `deberta-v3-base` resolve like training:

```bash
cd My_creation

python visualize/tsne_mosei_ablation.py \
  --ckpt "PRISM"=ablation_study/runs/mosei_phase1_trial70/checkpoints/infogate_mosei_best.pt \
  --ckpt "w/o MSelector"=ablation_study/runs/mosei_phase1_trial70_no_mselector/checkpoints/infogate_mosei_best.pt \
  --split test --mode facet --max-samples 4000

python visualize/tsne_mosei_ablation.py \
  --ckpt "PRISM"=ablation_study/runs/mosei_phase1_trial70/checkpoints/infogate_mosei_best.pt \
  --ckpt "w/o MSelector"=ablation_study/runs/mosei_phase1_trial70_no_mselector/checkpoints/infogate_mosei_best.pt \
  --split test --mode joint --max-samples 4000
```

Useful flags: `--perplexity`, `--seed`, `--standard-scale`, `--no-paper-style`, `--outdir PATH`.

In **joint** mode, embeddings from all checkpoints are concatenated and a **single** t-SNE is fit; hue indicates checkpoint name (markers cycle when there are more series than palette slots).

### All six MOSEI trial 70 PRISM ablations (batch)

[`run_tsne_mosei_prism_all.sh`](run_tsne_mosei_prism_all.sh) runs **PRISM** plus `w/o InfoGate`, `w/o MSelector`, `w/o IB`, `w/o ConfGating`, `w/o AdaptiveGate` checkpoints under `ablation_study/runs/mosei_phase1_trial70*`, producing **facet** (six panels, calm viridis sentiment + horizontal bar) and **joint** (one embedding space, hue = mode) in `visualize/figures/mosei_prism_tsne_all/` by default.

```bash
cd My_creation
bash visualize/run_tsne_mosei_prism_all.sh
# Optional: SPLIT=dev MAX_SAMPLES=3000 OUT=visualize/figures/my_tsne bash visualize/run_tsne_mosei_prism_all.sh
```
