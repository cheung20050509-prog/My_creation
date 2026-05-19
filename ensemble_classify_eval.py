"""Average logits from multiple InfoGate binary checkpoints on dev/test.

Uses the same loader protocol as `test_classify.py` and the same metric rule as
`train_classify.score` (sigmoid + 0.5 threshold, sklearn Acc / weighted-F1).
"""

import argparse
import os
import random

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import torch
from tqdm import tqdm
from transformers import AlbertTokenizer

from albert_infogate import InfoGate_AlbertForSequenceClassification
import global_configs
from global_configs import DEVICE
from data_humor import build_humor_loaders


def score_like_train(preds, y):
    """Match `train_classify.score` for binary classification."""
    preds = np.asarray(preds).flatten()
    y = np.asarray(y).flatten()
    prob = 1.0 / (1.0 + np.exp(-preds))
    yhat = (prob >= 0.5).astype(int)
    ytrue = y.astype(int)
    acc = accuracy_score(ytrue, yhat)
    f1_w = f1_score(ytrue, yhat, average="weighted") if len(ytrue) else 0.0
    return {"acc2": float(acc), "f1": float(f1_w)}


def apply_architecture_from_checkpoint(cli_args, ckpt_path):
    abs_ckpt = ckpt_path if os.path.isabs(ckpt_path) else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), ckpt_path)
    if not os.path.exists(abs_ckpt):
        return
    ckpt = torch.load(abs_ckpt, map_location="cpu", weights_only=False)
    saved = ckpt.get("args", None)
    if saved is None:
        return
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
        "align_mix_floor",
        "gumbel_tau_start",
        "gumbel_tau_end",
        "task_type",
        "max_seq_length",
    )
    for k in keys:
        if hasattr(saved, k):
            setattr(cli_args, k, getattr(saved, k))
    for flag, attr in (
        ("disable_l_lib", "use_l_lib"),
        ("disable_l_rib", "use_l_rib"),
    ):
        if hasattr(saved, flag):
            setattr(cli_args, attr, not bool(getattr(saved, flag)))


def set_seed(seed):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loaders(cli_args):
    tokenizer = AlbertTokenizer.from_pretrained(cli_args.model)
    train_bs = getattr(cli_args, "eval_batch_size", 128)
    return build_humor_loaders(
        dataset=cli_args.dataset,
        tokenizer=tokenizer,
        max_seq_length=cli_args.max_seq_length,
        acoustic_dim=global_configs.ACOUSTIC_DIM,
        visual_dim=global_configs.VISUAL_DIM,
        train_batch_size=train_bs,
        dev_batch_size=train_bs,
        test_batch_size=train_bs,
        gradient_accumulation_step=1,
        n_epochs=1,
        hcf_dim=global_configs.HCF_DIM,
        slice_hkt=True,
    )


def forward_logits(model, loader):
    preds, labels = [], []
    use_hcf = global_configs.HCF_DIM > 0
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Forward"):
            batch = tuple(t.to(DEVICE) for t in batch)
            if use_hcf:
                input_ids, visual, acoustic, hcf, label_ids = batch
                visual = visual.squeeze(1)
                acoustic = acoustic.squeeze(1)
                hcf = hcf.squeeze(1)
            else:
                input_ids, visual, acoustic, label_ids = batch
                visual = visual.squeeze(1)
                acoustic = acoustic.squeeze(1)
                hcf = None
            logits, _, _, _ = model(
                input_ids, visual, acoustic, hcf=hcf, stage=2)
            preds.extend(logits.view(-1).cpu().numpy().tolist())
            labels.extend(label_ids.view(-1).cpu().numpy().tolist())
    return np.asarray(preds, dtype=np.float64), np.asarray(labels)


def load_model_weights(cli_args, ckpt_path):
    model = InfoGate_AlbertForSequenceClassification.from_pretrained(
        cli_args.model, multimodal_config=cli_args, num_labels=1)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    sd = ckpt["model_state_dict"]
    try:
        model.load_state_dict(sd, strict=True)
    except RuntimeError:
        model.load_state_dict(sd, strict=False)
    model.to(DEVICE)
    return model


def main():
    pa = argparse.ArgumentParser(
        description="Logit-average ensemble for InfoGate binary classification")
    pa.add_argument("--checkpoints", nargs="+", required=True,
                  help="Two or more checkpoint paths (.pt)")
    pa.add_argument("--model", type=str,
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "albert-base-v2"))
    pa.add_argument("--dataset", type=str, choices=["ur_funny", "mustard"],
                    default="mustard")
    pa.add_argument("--split", type=str, choices=["test", "dev"], default="test")
    pa.add_argument("--eval_batch_size", type=int, default=128)
    pa.add_argument("--seed", type=int, default=42)
    cli = pa.parse_args()

    apply_architecture_from_checkpoint(cli, cli.checkpoints[0])
    cli.task_type = "binary"
    if not hasattr(cli, "use_l_lib"):
        cli.use_l_lib = True
    if not hasattr(cli, "use_l_rib"):
        cli.use_l_rib = True

    global_configs.set_dataset_config(cli.dataset)

    set_seed(cli.seed)
    _train_dl, dev_dl, test_dl, _ = build_loaders(cli)
    loader = dev_dl if cli.split == "dev" else test_dl

    logit_sum = None
    labels_ref = None
    for i, ckpt_path in enumerate(cli.checkpoints):
        abs_p = ckpt_path if os.path.isabs(ckpt_path) else os.path.join(
            os.path.dirname(os.path.abspath(__file__)), ckpt_path)
        print(f"Checkpoint [{i + 1}/{len(cli.checkpoints)}]: {abs_p}")
        model = load_model_weights(cli, abs_p)
        logits, labels = forward_logits(model, loader)
        del model
        if labels_ref is None:
            labels_ref = labels
            logit_sum = np.zeros_like(logits)
        elif not np.array_equal(labels_ref, labels):
            raise RuntimeError("Label order mismatch between checkpoints.")
        logit_sum += logits

    avg = logit_sum / float(len(cli.checkpoints))
    m = score_like_train(avg, labels_ref)
    print("=" * 60)
    print(f"Ensemble ({len(cli.checkpoints)} models)  split={cli.split}")
    print(f"  Acc (match train score): {m['acc2']*100:.2f}%")
    print(f"  F1  (weighted):         {m['f1']*100:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
