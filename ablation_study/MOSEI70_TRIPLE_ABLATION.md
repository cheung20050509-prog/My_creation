# MOSEI phase1 trial 70 — triple ablation (`w/o VTB+DPR+InfoGate`)

CLI flag: `--ablation no_ib_no_mselector_no_infogate`

Semantics (see `infogate_modules.py`):

- **VTB off** (`no_ib`): linear bottleneck projection, no stochastic VTB / IB losses.
- **DPR off** (`no_mselector`): fixed language primary; no routing supervision.
- **InfoGate off** (`no_infogate`): no auxiliary fusion; prediction from pooled primary bottleneck only.

Checkpoints and logs:

- Default directory: `ablation_study/runs/mosei_phase1_trial70_no_ib_no_mselector_no_infogate/checkpoints/`

Launch (from `My_creation/`):

```bash
conda run -n ITHP5090 python -u ablation_study/train_fixed_mosei_phase1_trial70.py \
  --ablation no_ib_no_mselector_no_infogate
```

Dry-run argv only:

```bash
python ablation_study/train_fixed_mosei_phase1_trial70.py \
  --ablation no_ib_no_mselector_no_infogate --dry-run
```

After training, copy test metrics into `overleaf_69e83a58/acl_latex.tex` Table `tab:ablation` (replace `---` placeholders).
