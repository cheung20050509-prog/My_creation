"""InfoGate evaluation script for binary classification (UR-FUNNY / MUSTARD).

Mirrors `test.py` but skips regression-only artifacts (MAE/Corr/SIMSV2 plot)
and reports Accuracy + Weighted-F1 + Macro-F1 instead.
"""

import argparse
import csv
import os
import random
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

import torch
from tqdm import tqdm

from transformers import DebertaV2Tokenizer
from deberta_infogate import InfoGate_DeBertaForSequenceClassification
import global_configs
from global_configs import DEVICE
from data_humor import build_humor_loaders


# ============================================================
# CLI
# ============================================================
parser = argparse.ArgumentParser(description="InfoGate Binary Classification Testing")
parser.add_argument("--model", type=str,
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "deberta-v3-base"))
parser.add_argument("--dataset", type=str, choices=["ur_funny", "mustard"], default="ur_funny")
parser.add_argument("--max_seq_length", type=int, default=64)
parser.add_argument("--test_batch_size", type=int, default=128)
parser.add_argument("--seed", type=int, default=42)

# InfoGate architecture (overridable; will be replaced by checkpoint args when present)
parser.add_argument("--unified_dim", type=int, default=256)
parser.add_argument("--ib_hidden_dim", type=int, default=256)
parser.add_argument("--bottleneck_dim", type=int, default=128)
parser.add_argument("--num_heads", type=int, default=4)
parser.add_argument("--num_infogate_layers", type=int, default=3)
parser.add_argument("--dropout_prob", type=float, default=0.1)
parser.add_argument("--beta_ib", type=float, default=16.0)

parser.add_argument("--checkpoint", type=str,
                    default="checkpoints/infogate_ur_funny_best.pt")
parser.add_argument("--save_prediction_csv", action="store_true",
                    help="Save the (true, prob, pred) triples as CSV.")
parser.add_argument("--csv_output", type=str, default="",
                    help="Optional output path for the CSV. Defaults under prediction_plots/.")
parser.add_argument("--threshold", type=float, default=0.5,
                    help="Probability threshold for positive class.")

args = parser.parse_args()


def apply_architecture_from_checkpoint(cli_args, ckpt_path):
    """Align InfoGate architecture hyperparameters with the saved training args."""
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
        "selector_target_temp",
        "selector_balance_weight",
        "selector_rib_weight",
        "gumbel_tau_start",
        "gumbel_tau_end",
        "task_type",
        "max_seq_length",
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
# Hard-code task type for this entry point so the model construction picks the
# binary L_lib / L_rib branches (the inference path doesn't actually need them).
args.task_type = "binary"
# Provide the same backward-compat fields train.py-style helpers expect.
if not hasattr(args, "use_l_lib"):
    args.use_l_lib = True
if not hasattr(args, "use_l_rib"):
    args.use_l_rib = True

global_configs.set_dataset_config(args.dataset)
ACOUSTIC_DIM = global_configs.ACOUSTIC_DIM
VISUAL_DIM = global_configs.VISUAL_DIM
TEXT_DIM = global_configs.TEXT_DIM


# ============================================================
# Data loading
# ============================================================

def get_test_dataloader():
    tokenizer = DebertaV2Tokenizer.from_pretrained(args.model)
    # We only need the test loader; pass dummy values for the unused ones.
    _train_dl, _dev_dl, test_dl, _ = build_humor_loaders(
        dataset=args.dataset,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        acoustic_dim=ACOUSTIC_DIM,
        visual_dim=VISUAL_DIM,
        train_batch_size=args.test_batch_size,
        dev_batch_size=args.test_batch_size,
        test_batch_size=args.test_batch_size,
        gradient_accumulation_step=1,
        n_epochs=1,
    )
    return test_dl


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
    model = InfoGate_DeBertaForSequenceClassification.from_pretrained(
        args.model, multimodal_config=args, num_labels=1)

    if os.path.exists(ckpt_path):
        print(f"Loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        sd = ckpt['model_state_dict']
        try:
            model.load_state_dict(sd, strict=True)
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
            preds.extend(logits.view(-1).cpu().numpy().tolist())
            labels.extend(label_ids.view(-1).cpu().numpy().tolist())

    return np.array(preds), np.array(labels)


def report(preds, labels):
    prob = 1.0 / (1.0 + np.exp(-preds))
    yhat = (prob >= args.threshold).astype(int)
    ytrue = np.asarray(labels).astype(int)

    acc = accuracy_score(ytrue, yhat)
    f1_w = f1_score(ytrue, yhat, average="weighted") if len(ytrue) else 0.0
    f1_m = f1_score(ytrue, yhat, average="macro") if len(ytrue) else 0.0
    p, r, _, _ = precision_recall_fscore_support(
        ytrue, yhat, average="weighted", zero_division=0)

    print("=" * 60)
    print(f"Dataset       : {args.dataset}")
    print(f"Threshold     : {args.threshold}")
    print(f"N samples     : {len(ytrue)}")
    print(f"Accuracy      : {acc*100:.4f} %")
    print(f"F1 (weighted) : {f1_w*100:.4f} %")
    print(f"F1 (macro)    : {f1_m*100:.4f} %")
    print(f"Precision (w) : {p*100:.4f} %")
    print(f"Recall    (w) : {r*100:.4f} %")
    print("Confusion matrix (rows=true, cols=pred):")
    cm = confusion_matrix(ytrue, yhat, labels=[0, 1])
    print(cm)
    print("Per-class report:")
    print(classification_report(ytrue, yhat, digits=4, zero_division=0))
    print("=" * 60)

    return {
        "Acc": acc, "F1_weighted": f1_w, "F1_macro": f1_m,
        "Precision_w": p, "Recall_w": r,
    }


def default_csv_path():
    ckpt_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prediction_plots")
    return os.path.join(out_dir, f"{args.dataset}_{ckpt_name}_predictions.csv")


def save_predictions_csv(preds, labels, output_path):
    prob = 1.0 / (1.0 + np.exp(-preds))
    yhat = (prob >= args.threshold).astype(int)
    ytrue = np.asarray(labels).astype(int)
    parent = os.path.dirname(os.path.abspath(output_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(output_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sample_index", "true_label", "logit", "probability", "prediction"])
        for i, (yt, lg, pr, yp) in enumerate(zip(ytrue, preds, prob, yhat)):
            w.writerow([i, int(yt), float(lg), float(pr), int(yp)])
    print(f"Saved prediction CSV to {output_path}")


def main():
    set_seed(args.seed)
    test_dl = get_test_dataloader()
    model = load_model(args.checkpoint)
    preds, labels = test_model(model, test_dl)
    report(preds, labels)

    if args.save_prediction_csv:
        path = args.csv_output or default_csv_path()
        save_predictions_csv(preds, labels, path)


if __name__ == "__main__":
    main()
