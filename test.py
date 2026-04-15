"""InfoGate test script for complete-modality evaluation."""

import argparse
import csv
import os
import random
import pickle
import numpy as np

from sklearn.metrics import accuracy_score, f1_score

import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from transformers import DebertaV2Tokenizer, BertTokenizer
from deberta_infogate import InfoGate_DeBertaForSequenceClassification
from bert_infogate import InfoGate_BertForSequenceClassification
import global_configs
from global_configs import DEVICE

# ============================================================
# CLI
# ============================================================
parser = argparse.ArgumentParser(description="InfoGate Testing")
parser.add_argument("--model", type=str,
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "deberta-v3-base"))
parser.add_argument("--dataset", type=str, choices=["mosi", "mosei", "simsv2"], default="mosi")
parser.add_argument("--max_seq_length", type=int, default=50)
parser.add_argument("--test_batch_size", type=int, default=128)
parser.add_argument("--seed", type=int, default=128)

# InfoGate architecture (must match training)
parser.add_argument("--unified_dim", type=int, default=256)
parser.add_argument("--ib_hidden_dim", type=int, default=256)
parser.add_argument("--bottleneck_dim", type=int, default=128)
parser.add_argument("--num_heads", type=int, default=4)
parser.add_argument("--num_infogate_layers", type=int, default=3)
parser.add_argument("--dropout_prob", type=float, default=0.1)
parser.add_argument("--beta_ib", type=float, default=32)

parser.add_argument("--checkpoint", type=str,
                    default="checkpoints/infogate_mosi_best.pt")
parser.add_argument("--plot_true_vs_pred", action="store_true",
                    help="Save a true score vs prediction scatter plot.")
parser.add_argument("--plot_output", type=str, default="",
                    help="Optional output path for the plot. Defaults under prediction_plots/.")
parser.add_argument("--save_prediction_csv", action="store_true",
                    help="Save the true/prediction pairs as CSV.")
parser.add_argument("--csv_output", type=str, default="",
                    help="Optional output path for the CSV. Defaults under prediction_plots/.")

args = parser.parse_args()


def apply_architecture_from_checkpoint(cli_args, ckpt_path):
    """Align InfoGate architecture hyperparameters with the saved training args.

    Optuna checkpoints are trained with varying widths/depths; `test.py` CLI defaults
    are not guaranteed to match. If the checkpoint contains `args`, override the
    corresponding fields on `cli_args` before model construction.
    """
    if not ckpt_path:
        return
    abs_ckpt = ckpt_path if os.path.isabs(ckpt_path) else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), ckpt_path)
    if not os.path.exists(abs_ckpt):
        return

    ckpt = torch.load(abs_ckpt, map_location="cpu", weights_only=False)
    saved = ckpt.get("args", None)
    if saved is None:
        return

    saved_ds = getattr(saved, "dataset", None)
    if saved_ds is not None and saved_ds != cli_args.dataset:
        print(
            f"WARNING: checkpoint dataset={saved_ds} but CLI --dataset={cli_args.dataset}. "
            f"Proceeding with CLI dataset; verify this is intentional."
        )

    keys = (
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
        "gamma_cyc",
        "cra_layers",
        "cra_dims",
    )
    applied = []
    for k in keys:
        if hasattr(saved, k):
            setattr(cli_args, k, getattr(saved, k))
            applied.append(k)

    for flag, attr in (
        ("disable_l_lib", "use_l_lib"),
        ("disable_l_rib", "use_l_rib"),
    ):
        if hasattr(saved, flag):
            setattr(cli_args, attr, not bool(getattr(saved, flag)))
            applied.append(attr)

    if applied:
        print(f"Applied architecture overrides from checkpoint args: {', '.join(applied)}")


apply_architecture_from_checkpoint(args, args.checkpoint)

if args.dataset == "simsv2":
    if "deberta-v3-base" in args.model:
        args.model = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bert-base-chinese")

global_configs.set_dataset_config(args.dataset)
ACOUSTIC_DIM = global_configs.ACOUSTIC_DIM
VISUAL_DIM = global_configs.VISUAL_DIM
TEXT_DIM = global_configs.TEXT_DIM


# ============================================================
# Data loading
# ============================================================

class InputFeatures:
    __slots__ = ['input_ids', 'visual', 'acoustic', 'input_mask',
                 'segment_ids', 'label_id']

    def __init__(self, input_ids, visual, acoustic, input_mask,
                 segment_ids, label_id):
        self.input_ids = input_ids
        self.visual = visual
        self.acoustic = acoustic
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


def prepare_deberta_input(tokens, visual, acoustic, tokenizer):
    CLS = tokenizer.cls_token
    SEP = tokenizer.sep_token
    tokens = [CLS] + tokens + [SEP]

    az = np.zeros((1, ACOUSTIC_DIM))
    acoustic = np.concatenate((az, acoustic, az))
    vz = np.zeros((1, VISUAL_DIM))
    visual = np.concatenate((vz, visual, vz))

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    segment_ids = [0] * len(input_ids)
    input_mask = [1] * len(input_ids)

    pad = args.max_seq_length - len(input_ids)
    acoustic = np.concatenate((acoustic, np.zeros((pad, ACOUSTIC_DIM))))
    visual = np.concatenate((visual, np.zeros((pad, VISUAL_DIM))))
    input_ids += [0] * pad
    input_mask += [0] * pad
    segment_ids += [0] * pad
    return input_ids, visual, acoustic, input_mask, segment_ids


def convert_to_features(examples, max_seq_length, tokenizer):
    features = []
    if args.dataset == "simsv2":
        # examples is a dict
        num_samples = len(examples["raw_text"])
        for i in range(num_samples):
            words = examples["raw_text"][i]
            visual = examples["vision"][i]
            acoustic = examples["audio"][i]
            label_id = examples["regression_labels"][i]
            
            # Ensure it's a scalar value
            if isinstance(label_id, (list, tuple, np.ndarray)):
                label_id = label_id[0]

            tokens = tokenizer.tokenize(words)
            if len(tokens) > max_seq_length - 2:
                tokens = tokens[:max_seq_length - 2]
                visual = visual[:max_seq_length - 2]
                acoustic = acoustic[:max_seq_length - 2]
            else:
                # pad or truncate if needed, SIMSv2 pre-extracts features usually matching text tokens
                # Let's just truncate visual/acoustic to the length of tokens.
                min_len = min(len(tokens), len(visual), len(acoustic))
                tokens = tokens[:min_len]
                visual = visual[:min_len]
                acoustic = acoustic[:min_len]

            ids, vis, aud, mask, seg = prepare_deberta_input(
                tokens, visual, acoustic, tokenizer)

            features.append(InputFeatures(ids, vis, aud, mask, seg, label_id))
    else:
        for example in examples:
            (words, visual, acoustic), label_id, segment = example
            tokens, inversions = [], []
            for idx, word in enumerate(words):
                toks = tokenizer.tokenize(word)
                tokens.extend(toks)
                inversions.extend([idx] * len(toks))

            aligned_v = np.array([visual[i] for i in inversions])
            aligned_a = np.array([acoustic[i] for i in inversions])

            if len(tokens) > max_seq_length - 2:
                tokens = tokens[:max_seq_length - 2]
                aligned_a = aligned_a[:max_seq_length - 2]
                aligned_v = aligned_v[:max_seq_length - 2]

            ids, vis, aud, mask, seg = prepare_deberta_input(
                tokens, aligned_v, aligned_a, tokenizer)
            features.append(InputFeatures(ids, vis, aud, mask, seg, label_id))
    return features

def get_tokenizer(model):
    if args.dataset == "simsv2":
        return BertTokenizer.from_pretrained(model)
    return DebertaV2Tokenizer.from_pretrained(model)

def get_test_dataloader():
    with open(f"datasets/{args.dataset}.pkl", "rb") as fh:
        data = pickle.load(fh)
    tok = get_tokenizer(args.model)
    feats = convert_to_features(data["test"], args.max_seq_length, tok)
    ds = TensorDataset(
        torch.tensor(np.array([f.input_ids for f in feats]), dtype=torch.long),
        torch.tensor(np.array([f.visual for f in feats]), dtype=torch.float),
        torch.tensor(np.array([f.acoustic for f in feats]), dtype=torch.float),
        torch.tensor(np.array([f.label_id for f in feats]), dtype=torch.float),
    )
    return DataLoader(ds, batch_size=args.test_batch_size, shuffle=False)


# ============================================================
# Model loading
# ============================================================

def set_seed(seed):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model(ckpt_path):
    if args.dataset == "simsv2":
        model = InfoGate_BertForSequenceClassification.from_pretrained(
            args.model, multimodal_config=args, num_labels=1)
    else:
        model = InfoGate_DeBertaForSequenceClassification.from_pretrained(
            args.model, multimodal_config=args, num_labels=1)

    if os.path.exists(ckpt_path):
        print(f"Loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        sd = ckpt['model_state_dict']
        model_keys = set(model.state_dict().keys())
        sd_keys = set(sd.keys())
        missing_in_sd = sorted(model_keys - sd_keys)
        unexpected_in_sd = sorted(sd_keys - model_keys)
        if unexpected_in_sd:
            print(f"  WARNING: {len(unexpected_in_sd)} unexpected keys in checkpoint "
                  f"(first 5): {unexpected_in_sd[:5]}")
        if missing_in_sd:
            print(f"  WARNING: {len(missing_in_sd)} model keys missing in checkpoint "
                  f"(first 5): {missing_in_sd[:5]}")
        try:
            incompatible = model.load_state_dict(sd, strict=True)
            if incompatible.missing_keys or incompatible.unexpected_keys:
                # Should not happen under strict=True, but keep a breadcrumb if API changes.
                print(f"  load_state_dict(strict=True) returned missing="
                      f"{len(incompatible.missing_keys)} unexpected="
                      f"{len(incompatible.unexpected_keys)}")
        except RuntimeError as e:
            print("  strict load_state_dict failed; falling back to strict=False")
            print(f"    {e}")
            model.load_state_dict(sd, strict=False)
    else:
        print(f"WARNING: checkpoint not found at {ckpt_path}")

    model.to(DEVICE)
    model.eval()
    return model


# ============================================================
# Evaluation
# ============================================================

def test_model(model, loader):
    preds, labels = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Testing"):
            batch = tuple(t.to(DEVICE) for t in batch)
            input_ids, visual, acoustic, label_ids = batch
            visual = visual.squeeze(1)
            acoustic = acoustic.squeeze(1)

            logits, _, _, _ = model(input_ids, visual, acoustic, stage=2)

            logits = logits.squeeze(-1).cpu().numpy()
            label_ids = label_ids.cpu().numpy().flatten()

            preds.extend(logits.tolist() if logits.ndim > 0 else [logits.item()])
            labels.extend(label_ids.tolist())

    return np.array(preds), np.array(labels)


def filter_prediction_pairs(preds, labels, use_zero=False):
    preds = np.asarray(preds).flatten()
    labels = np.asarray(labels).flatten()
    if preds.shape != labels.shape:
        raise ValueError(f"pred/label shape mismatch: {preds.shape} vs {labels.shape}")
    mask = np.ones(labels.shape[0], dtype=bool) if use_zero else (labels != 0)
    return preds[mask], labels[mask], mask


def safe_corrcoef(preds, labels):
    preds = np.asarray(preds).flatten()
    labels = np.asarray(labels).flatten()
    if len(preds) < 2:
        return 0.0
    if np.allclose(preds, preds[0]) or np.allclose(labels, labels[0]):
        return 0.0
    corr = np.corrcoef(preds, labels)[0][1]
    return 0.0 if np.isnan(corr) else float(corr)


def ensure_parent_dir(path):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def default_output_path(filename):
    ckpt_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prediction_plots")
    return os.path.join(out_dir, f"{args.dataset}_{ckpt_name}_{filename}")


def save_prediction_csv(preds, labels, output_path, use_zero=False):
    p, y, mask = filter_prediction_pairs(preds, labels, use_zero=use_zero)
    ensure_parent_dir(output_path)
    kept_indices = np.flatnonzero(mask)
    with open(output_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["sample_index", "true_score", "prediction", "abs_error"])
        for idx, true_score, pred in zip(kept_indices, y, p):
            writer.writerow([int(idx), float(true_score), float(pred), float(abs(pred - true_score))])


def save_true_vs_pred_plot(preds, labels, metrics, output_path, use_zero=False):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for --plot_true_vs_pred. Install it with `pip install matplotlib`."
        ) from exc

    p, y, _ = filter_prediction_pairs(preds, labels, use_zero=use_zero)
    if len(p) == 0:
        raise ValueError("No samples available for plotting after label filtering.")

    abs_err = np.abs(p - y)
    lo = float(min(p.min(), y.min()))
    hi = float(max(p.max(), y.max()))
    span = hi - lo
    pad = 0.2 if span <= 0 else span * 0.05
    lo -= pad
    hi += pad

    ensure_parent_dir(output_path)

    fig, ax = plt.subplots(figsize=(7.5, 7.0))
    scatter = ax.scatter(
        y, p,
        c=abs_err,
        cmap="viridis",
        s=28,
        alpha=0.78,
        edgecolors="none",
    )
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="black", linewidth=1.0)
    ax.axhline(0.0, color="0.75", linewidth=0.8)
    ax.axvline(0.0, color="0.75", linewidth=0.8)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("True score")
    ax.set_ylabel("Prediction")
    ax.set_title(f"{args.dataset.upper()} true score vs prediction")

    stats_lines = [
        f"N={len(y)} ({'all labels' if use_zero else 'labels != 0'})",
        f"MAE={metrics['MAE']:.4f}",
        f"Corr={metrics['Corr']:.4f}",
        f"Acc2={metrics['Acc2']:.4f}",
    ]
    if "Acc7" in metrics:
        stats_lines.append(f"Acc7={metrics['Acc7']:.4f}")
    if "Acc5" in metrics:
        stats_lines.append(f"Acc5={metrics['Acc5']:.4f}")
    if "Acc3" in metrics:
        stats_lines.append(f"Acc3={metrics['Acc3']:.4f}")
    stats_lines.append(f"Pred range=[{p.min():.2f}, {p.max():.2f}]")
    stats_lines.append(f"True range=[{y.min():.2f}, {y.max():.2f}]")
    ax.text(
        0.02, 0.98,
        "\n".join(stats_lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.92, "edgecolor": "0.75"},
    )

    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("|prediction - true score|")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def print_prediction_summary(preds, labels, use_zero=False):
    p, y, _ = filter_prediction_pairs(preds, labels, use_zero=use_zero)
    scope = "all labels" if use_zero else "labels != 0"
    print(f"  View: {scope} ({len(y)}/{len(labels)} samples)")
    if len(y) == 0:
        print("  No samples available after filtering.")
        return
    abs_err = np.abs(p - y)
    print(f"  True range: [{y.min():.4f}, {y.max():.4f}]")
    print(f"  Pred range: [{p.min():.4f}, {p.max():.4f}]")
    print(f"  Pred mean/std: {p.mean():.4f} / {p.std():.4f}")
    print(f"  Abs error mean/max: {abs_err.mean():.4f} / {abs_err.max():.4f}")


def compute_metrics(preds, labels, dataset="mosi", use_zero=False):
    # Full set for MAE, Corr, Acc7
    p_full, y_full, _ = filter_prediction_pairs(preds, labels, use_zero=True)
    mae = np.mean(np.abs(p_full - y_full))
    corr = safe_corrcoef(p_full, y_full)

    # Filtered set for Acc2, F1
    p_nz, y_nz, _ = filter_prediction_pairs(preds, labels, use_zero=False)
    pb, yb = (p_nz >= 0), (y_nz >= 0)
    acc2 = accuracy_score(yb, pb)
    f1 = f1_score(yb, pb, average="weighted")

    result = {'MAE': mae, 'Corr': corr, 'Acc2': acc2, 'F1': f1}
    if dataset == "simsv2":
        p5 = np.clip(np.round(p_full * 2), -2, 2).astype(int)
        y5 = np.clip(np.round(y_full * 2), -2, 2).astype(int)
        result['Acc5'] = accuracy_score(y5, p5)
        p3 = np.sign(p_full).astype(int)
        y3 = np.sign(y_full).astype(int)
        result['Acc3'] = accuracy_score(y3, p3)
    else:
        p7 = np.clip(np.round(p_full), -3, 3).astype(int)
        y7 = np.clip(np.round(y_full), -3, 3).astype(int)
        result['Acc7'] = accuracy_score(y7, p7)
    return result


def print_metrics(metrics, prefix=""):
    s = f"{prefix}Acc2: {metrics['Acc2']:.4f}"
    if 'Acc5' in metrics:
        s += f"  Acc5: {metrics['Acc5']:.4f}  Acc3: {metrics['Acc3']:.4f}"
    else:
        s += f"  Acc7: {metrics['Acc7']:.4f}"
    s += (f"  F1: {metrics['F1']:.4f}  MAE: {metrics['MAE']:.4f}  "
          f"Corr: {metrics['Corr']:.4f}")
    print(s)


# ============================================================
# Main
# ============================================================

def main():
    set_seed(args.seed)
    print("=" * 60)
    print(f"InfoGate Test — dataset: {args.dataset}")
    print(f"Checkpoint: {args.checkpoint}")
    print("=" * 60)

    model = load_model(args.checkpoint)
    loader = get_test_dataloader()
    print(f"Test samples: {len(loader.dataset)}")

    print("\n[Complete Modality]")
    preds, labels = test_model(model, loader)
    cm = compute_metrics(preds, labels, dataset=args.dataset)
    print_metrics(cm, "  ")
    print_prediction_summary(preds, labels, use_zero=True)

    if args.save_prediction_csv:
        csv_path = args.csv_output or default_output_path("true_pred_pairs.csv")
        save_prediction_csv(preds, labels, csv_path, use_zero=True)
        print(f"  Saved prediction pairs to: {csv_path}")

    if args.plot_true_vs_pred:
        plot_metrics = compute_metrics(
            preds, labels, dataset=args.dataset, use_zero=True)
        plot_path = args.plot_output or default_output_path("true_vs_pred.png")
        save_true_vs_pred_plot(
            preds, labels, plot_metrics, plot_path, use_zero=True)
        print(f"  Saved true-vs-pred plot to: {plot_path}")


if __name__ == "__main__":
    main()
