# Podium — specially preserved best trials

Optuna 在共享 `ckpt_subdir` 下会**互相覆盖** ckpt（`optuna_mosi_s2_local/trial_X/` 同时被 `regular_s2_local` 和 `msew35_s2_local` 写）。
本目录保存「指标特别突出、不能再被覆盖」的 trial：把对应 ckpt + train log + 完整指标 snapshot 集中存放。

> `*.pt` 仍受根 `.gitignore` 排除，不会进 git；磁盘上保留是为了不被后续 Optuna run 覆盖。
> `snapshot.json` 与 `train_log.log` 会进 git。

## 当前 podium

### `mosi_msew35_s2_trial69_acc2_88p85/`

- **数据集**：MOSI
- **来源**：`infogate_mosi_optuna_relaunch_20260415_120746_msew35_s2_local`，trial 69，seed=128
- **被 podium 的原因**：MOSI 全部 trial 中 **Acc-2 / F1 最高**
  - dev-MAE 选点（epoch 41）→ test: **Acc-2 88.85% / F1 88.83% / MAE 0.5957 / Corr 0.8587 / Acc-7 48.91%**
  - Oracle best test MAE（epoch 35）→ Acc-2 88.55% / F1 88.53% / MAE 0.5930 / Corr 0.8611 / Acc-7 49.05%
  - Oracle best test Acc-2/F1（epoch 42）→ **Acc-2 89.01% / F1 88.98%**
- 与 MAE-podium（msew35_s2 trial 125, MAE 0.5923）相比，MAE 仅差 0.003，但 Acc-2/F1 高 1pt+

复现命令在 `snapshot.json["reproduce_command_template"]`，或用 `multi_seed_verify.sh mosi_acc2`。
