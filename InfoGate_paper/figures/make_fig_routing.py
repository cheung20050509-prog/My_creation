"""
Generate Figure 3 for the InfoGate EMNLP paper.

Loads the MOSI Acc-2/F1 podium checkpoint (trial 69) and computes the
MSelector per-sample distribution and per-modality routing weights on the
MOSI test split.  Produces:
  - figures/fig_routing.pdf  (the figure used in the paper)
  - figures/fig_routing_data.csv  (the underlying numbers, for reproducibility)
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import global_configs
from global_configs import DEVICE
from deberta_infogate import InfoGate_DeBertaForSequenceClassification

CKPT = REPO / "saved_hparams" / "podium" / "mosi_msew35_s2_trial69_acc2_88p85" / "infogate_mosi_best.pt"
OUT_PDF = Path(__file__).with_name("fig_routing.pdf")
OUT_CSV = Path(__file__).with_name("fig_routing_data.csv")
DATASET = "mosi"

# -----------------------------------------------------------------
# data
# -----------------------------------------------------------------

def build_test_loader(model_path, batch_size=64, max_seq_length=50):
    import pickle
    from transformers import DebertaV2Tokenizer
    from torch.utils.data import TensorDataset

    global_configs.set_dataset_config(DATASET)
    A_DIM = global_configs.ACOUSTIC_DIM
    V_DIM = global_configs.VISUAL_DIM

    tok = DebertaV2Tokenizer.from_pretrained(model_path)
    with open(REPO / "datasets" / f"{DATASET}.pkl", "rb") as fh:
        data = pickle.load(fh)
    examples = data["test"]

    feats = []
    for example in examples:
        (words, visual, acoustic), label_id, _segment = example
        tokens, inversions = [], []
        for idx, word in enumerate(words):
            toks = tok.tokenize(word)
            tokens.extend(toks)
            inversions.extend([idx] * len(toks))
        aligned_v = np.array([visual[i] for i in inversions])
        aligned_a = np.array([acoustic[i] for i in inversions])
        if len(tokens) > max_seq_length - 2:
            tokens = tokens[: max_seq_length - 2]
            aligned_a = aligned_a[: max_seq_length - 2]
            aligned_v = aligned_v[: max_seq_length - 2]
        tokens = [tok.cls_token] + tokens + [tok.sep_token]
        az = np.zeros((1, A_DIM))
        aligned_a = np.concatenate((az, aligned_a, az))
        vz = np.zeros((1, V_DIM))
        aligned_v = np.concatenate((vz, aligned_v, vz))
        ids = tok.convert_tokens_to_ids(tokens)
        mask = [1] * len(ids)
        pad = max_seq_length - len(ids)
        ids += [0] * pad
        mask += [0] * pad
        aligned_a = np.concatenate((aligned_a, np.zeros((pad, A_DIM))))
        aligned_v = np.concatenate((aligned_v, np.zeros((pad, V_DIM))))
        feats.append((ids, aligned_v, aligned_a, label_id))

    ds = TensorDataset(
        torch.tensor(np.array([f[0] for f in feats]), dtype=torch.long),
        torch.tensor(np.array([f[1] for f in feats]), dtype=torch.float),
        torch.tensor(np.array([f[2] for f in feats]), dtype=torch.float),
        torch.tensor(np.array([f[3] for f in feats]), dtype=torch.float),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


# -----------------------------------------------------------------
# model
# -----------------------------------------------------------------

class _Cfg:
    pass


def build_model(ckpt_path):
    saved = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = saved["args"]
    cli = _Cfg()
    for k in (
        "unified_dim",
        "ib_hidden_dim",
        "bottleneck_dim",
        "num_heads",
        "num_infogate_layers",
        "dropout_prob",
        "beta_ib",
        "alpha_ib",
        "mse_weight",
        "selector_target_temp",
        "selector_balance_weight",
        "selector_rib_weight",
        "gumbel_tau_start",
        "gumbel_tau_end",
    ):
        setattr(cli, k, getattr(a, k, None))
    cli.use_l_lib = not getattr(a, "disable_l_lib", False)
    cli.use_l_rib = not getattr(a, "disable_l_rib", False)

    model_dir = REPO / "deberta-v3-base"
    model = InfoGate_DeBertaForSequenceClassification.from_pretrained(
        str(model_dir), multimodal_config=cli, num_labels=1
    )
    sd = saved["model_state_dict"]
    incompatible = model.load_state_dict(sd, strict=False)
    if incompatible.missing_keys:
        print(f"WARN missing {len(incompatible.missing_keys)} keys, e.g. {incompatible.missing_keys[:3]}")
    if incompatible.unexpected_keys:
        print(f"WARN unexpected {len(incompatible.unexpected_keys)} keys, e.g. {incompatible.unexpected_keys[:3]}")
    model.to(DEVICE)
    model.eval()
    return model, str(model_dir)


# -----------------------------------------------------------------
# inference + plot
# -----------------------------------------------------------------

@torch.no_grad()
def collect_routing(model, loader):
    primary = []  # 0=acoustic, 1=language, 2=visual
    weights = []  # B x 3 (a, l, v)
    labels = []
    preds = []
    for batch in loader:
        ids, vis, aco, y = (t.to(DEVICE) for t in batch)
        vis = vis.squeeze(1)
        aco = aco.squeeze(1)
        logits, _, loss_dict, _ = model(ids, vis, aco, stage=2)
        # Re-run the MSelector explicitly so we have per-sample tensors,
        # not only the batch-mean diagnostics in loss_dict.
        # The model API does not expose per-sample primary; we therefore
        # patch by re-running the underlying InfoGate forward.
        info = model.dberta.infogate
        attention_mask = (ids != 0).float()
        out = model.dberta.model(input_ids=ids, attention_mask=attention_mask.long())
        text_h = out.last_hidden_state
        # Re-run info_gate forward to capture MSelector
        F_t = info.proj_t(text_h)
        F_a = info.proj_a(aco)
        F_v = info.proj_v(vis)
        B_t, *_ = info.ib_enc_t(F_t)
        B_a, *_ = info.ib_enc_a(F_a)
        B_v, *_ = info.ib_enc_v(F_v)
        _, _, _, w, p_oh, p_idx = info.mselector(B_a, B_t, B_v, attention_mask)
        primary.append(p_idx.cpu().numpy())
        weights.append(w.cpu().numpy())
        labels.append(y.cpu().numpy())
        preds.append(logits.squeeze(-1).cpu().numpy())
    return (
        np.concatenate(primary),
        np.concatenate(weights),
        np.concatenate(labels),
        np.concatenate(preds),
    )


def plot(primary, weights, labels):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.8))

    # Left: bar of primary modality distribution
    counts = np.bincount(primary, minlength=3) / max(len(primary), 1)
    bars = axes[0].bar(
        ["acoustic", "language", "visual"],
        counts * 100,
        edgecolor="black",
        linewidth=0.6,
    )
    for b, c in zip(bars, counts):
        axes[0].text(b.get_x() + b.get_width() / 2, b.get_height() + 1, f"{c*100:.1f}%",
                     ha="center", va="bottom", fontsize=9)
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel("% of MOSI test samples")
    axes[0].set_title("(a) Primary modality (argmax)")

    # Right: average soft routing weight, conditioned on absolute sentiment intensity
    abs_y = np.abs(labels.flatten())
    bins = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0])
    ind = np.digitize(abs_y, bins) - 1
    ind = np.clip(ind, 0, len(bins) - 2)
    means = np.zeros((len(bins) - 1, 3))
    for k in range(len(bins) - 1):
        sel = ind == k
        if sel.sum() > 0:
            means[k] = weights[sel].mean(axis=0)
        else:
            means[k] = np.nan
    bottoms = np.zeros(len(bins) - 1)
    cols = ["#5b9bd5", "#ed7d31", "#70ad47"]
    for m_idx, (mod, col) in enumerate(zip(["acoustic", "language", "visual"], cols)):
        axes[1].bar(
            np.arange(len(bins) - 1),
            means[:, m_idx],
            bottom=bottoms,
            label=mod,
            edgecolor="black",
            linewidth=0.4,
            color=col,
        )
        bottoms += np.nan_to_num(means[:, m_idx])
    axes[1].set_xticks(np.arange(len(bins) - 1))
    axes[1].set_xticklabels([f"[{bins[i]:.1f},{bins[i+1]:.1f})" for i in range(len(bins) - 1)],
                            fontsize=8, rotation=20)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("mean MSelector weight")
    axes[1].set_xlabel("|sentiment intensity|")
    axes[1].set_title("(b) Soft routing vs. intensity")
    axes[1].legend(loc="lower right", fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_PDF}")

    # save raw csv
    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["panel", "key", "acoustic", "language", "visual"])
        w.writerow(["a_distribution", "fraction", *(counts.tolist())])
        for k in range(len(bins) - 1):
            w.writerow([
                "b_routing_by_intensity",
                f"|y| in [{bins[k]:.1f},{bins[k+1]:.1f})",
                *(means[k].tolist()),
            ])
    print(f"wrote {OUT_CSV}")


def main():
    if not CKPT.exists():
        raise FileNotFoundError(f"podium ckpt missing: {CKPT}")
    global_configs.set_dataset_config(DATASET)
    model, model_dir = build_model(str(CKPT))
    loader = build_test_loader(model_dir)
    primary, weights, labels, preds = collect_routing(model, loader)
    print("test set size:", len(primary), "preds range:", preds.min(), preds.max())
    print("primary distribution (a/l/v):", np.bincount(primary, minlength=3) / len(primary))
    plot(primary, weights, labels)


if __name__ == "__main__":
    main()
