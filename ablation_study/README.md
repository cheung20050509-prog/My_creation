# Ablation Study — PRISM / InfoGate

Architectural ablation experiments. Each script runs 6 variants (full model + 5 ablations) on one dataset.

## Usage

```bash
# Single dataset
bash run_mosi.sh       # CMU-MOSI, 6 experiments
bash run_mosei.sh      # CMU-MOSEI
bash run_simsv2.sh     # CH-SIMS v2
bash run_ur_funny.sh   # UR-FUNNY
bash run_mustard.sh    # MUStARD

# All 5 datasets (30 experiments total)
bash run_all.sh
```

## Ablations

| `--ablation` | What changes | Tests |
|-------------|--------------|-------|
| `none` | Full PRISM | Baseline |
| `no_infogate` | Skip confidence-gated cross-attention fusion (B_p → classifier directly) | InfoGate attention |
| `no_mselector` | Fix primary to text modality (no dynamic routing) | Dynamic primary selection |
| `no_ib` | Skip variational IB encoder, zero IB losses | Information bottleneck |
| `no_conf_gating` | Strip confidence signals → standard multi-head attention | IB-guided confidence gating |
| `no_adaptive_gate` | Equal-weight auxiliary fusion (no learned gate, no alignment) | Adaptive gating + alignment |

## Logs

Output: `logs/{dataset}_{ablation}.log` (e.g., `logs/mosi_no_infogate.log`)

## Structure

- `train_regression.py` / `train_classification.py` — entry points with `--ablation` flag
- `infogate_modules.py` — ablation code paths in InfoGate forward
- `deberta-v3-base/`, `bert-base-chinese/`, `albert-base-v2/`, `datasets/` — symlinks to `../fixed_experiment/`
