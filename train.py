"""
InfoGate training script.
Two-stage training:
  Stage 1  -- L_task + alpha_ib * L_IB + delta_nce * L_nce
  Stage 2  -- L_task + alpha_ib * L_IB + gamma_cyc * L_cyc + delta_nce * L_nce
"""

import argparse
import os
import random
import pickle
import numpy as np

from sklearn.metrics import accuracy_score, f1_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from transformers import get_linear_schedule_with_warmup, DebertaV2Tokenizer
from torch.optim import AdamW

from deberta_infogate import InfoGate_DeBertaForSequenceClassification
import global_configs
from global_configs import DEVICE

# ============================================================
# CLI
# ============================================================
parser = argparse.ArgumentParser(description="InfoGate Training")
parser.add_argument("--model", type=str,
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "deberta-v3-base"))
parser.add_argument("--dataset", type=str, choices=["mosi", "mosei"], default="mosi")
parser.add_argument("--max_seq_length", type=int, default=50)
parser.add_argument("--train_batch_size", type=int, default=32)
parser.add_argument("--dev_batch_size", type=int, default=128)
parser.add_argument("--test_batch_size", type=int, default=128)
parser.add_argument("--n_epochs", type=int, default=50)
parser.add_argument("--stage1_epochs", type=int, default=10)
parser.add_argument("--dropout_prob", type=float, default=0.1)
parser.add_argument("--learning_rate", type=float, default=2e-5)
parser.add_argument("--gradient_accumulation_step", type=int, default=1)
parser.add_argument("--warmup_proportion", type=float, default=0.1)
parser.add_argument("--weight_decay", type=float, default=1e-3)
parser.add_argument("--seed", type=int, default=128)

# InfoGate-specific
parser.add_argument("--unified_dim", type=int, default=256)
parser.add_argument("--ib_hidden_dim", type=int, default=256)
parser.add_argument("--bottleneck_dim", type=int, default=128)
parser.add_argument("--num_heads", type=int, default=4)
parser.add_argument("--num_infogate_layers", type=int, default=3)
parser.add_argument("--beta_ib", type=float, default=32)
parser.add_argument("--gamma_cyc", type=float, default=1.0)
parser.add_argument("--alpha_ib", type=float, default=0.01)
parser.add_argument("--alpha_nce", type=float, default=0.05)
parser.add_argument("--cra_layers", type=int, default=8)
parser.add_argument("--cra_dims", default="64,32,16", type=str)

parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")

args = parser.parse_args()

if isinstance(args.cra_dims, str):
    args.cra_dims = [int(x) for x in args.cra_dims.split(',')]

global_configs.set_dataset_config(args.dataset)
ACOUSTIC_DIM = global_configs.ACOUSTIC_DIM
VISUAL_DIM = global_configs.VISUAL_DIM
TEXT_DIM = global_configs.TEXT_DIM


# ============================================================
# Data
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
    return DebertaV2Tokenizer.from_pretrained(model)


def get_dataset(data):
    tok = get_tokenizer(args.model)
    feats = convert_to_features(data, args.max_seq_length, tok)
    return TensorDataset(
        torch.tensor(np.array([f.input_ids for f in feats]), dtype=torch.long),
        torch.tensor(np.array([f.visual for f in feats]), dtype=torch.float),
        torch.tensor(np.array([f.acoustic for f in feats]), dtype=torch.float),
        torch.tensor(np.array([f.label_id for f in feats]), dtype=torch.float),
    )


def setup_data():
    with open(f"datasets/{args.dataset}.pkl", "rb") as fh:
        data = pickle.load(fh)

    train_ds = get_dataset(data["train"])
    dev_ds = get_dataset(data["dev"])
    test_ds = get_dataset(data["test"])

    n_opt = int(len(train_ds) / args.train_batch_size
                / args.gradient_accumulation_step) * args.n_epochs

    train_dl = DataLoader(train_ds, batch_size=args.train_batch_size, shuffle=True)
    dev_dl = DataLoader(dev_ds, batch_size=args.dev_batch_size, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=args.test_batch_size, shuffle=False)
    return train_dl, dev_dl, test_dl, n_opt


# ============================================================
# Seed & model setup
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


def build_model(n_opt):
    model = InfoGate_DeBertaForSequenceClassification.from_pretrained(
        args.model, multimodal_config=args, num_labels=1)
    model.to(DEVICE)

    params = list(model.named_parameters())
    no_decay = {"bias", "LayerNorm.bias", "LayerNorm.weight"}
    groups = [
        {"params": [p for n, p in params if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in params if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = AdamW(groups, lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_proportion * n_opt),
        num_training_steps=n_opt,
    )
    return model, optimizer, scheduler


# ============================================================
# InfoNCE (from MODS)
# ============================================================

def compute_infonce(nce_extras, temperature=0.07):
    h_p = nce_extras['h_p']
    total = h_p.new_tensor(0.0)
    for h_key, f_key in [('h_a', 'F_a'), ('h_l', 'F_l'), ('h_v', 'F_v')]:
        h_m = F.normalize(nce_extras[h_key], dim=-1)
        f_hp = F.normalize(nce_extras[f_key](h_p), dim=-1)
        sim = torch.mm(h_m, f_hp.t()) / temperature
        labels = torch.arange(h_p.size(0), device=h_p.device)
        total = total + F.cross_entropy(sim, labels)
    return total / 3.0


# ============================================================
# Train / Eval / Test
# ============================================================

def train_epoch(model, loader, optimizer, scheduler, stage):
    model.train()
    total_loss, steps = 0.0, 0
    sum_task, sum_ib, sum_nce = 0.0, 0.0, 0.0
    sum_detail = {}

    for step, batch in enumerate(tqdm(loader, desc=f"Train (stage {stage})")):
        batch = tuple(t.to(DEVICE) for t in batch)
        input_ids, visual, acoustic, label_ids = batch
        visual = visual.squeeze(1)
        acoustic = acoustic.squeeze(1)

        v_norm = (visual - visual.min()) / (visual.max() - visual.min() + 1e-8)
        a_norm = (acoustic - acoustic.min()) / (acoustic.max() - acoustic.min() + 1e-8)

        logits, ib_loss, loss_dict, nce_extras = model(
            input_ids, v_norm, a_norm, labels=label_ids, stage=stage)

        l_task = F.l1_loss(logits.view(-1), label_ids.view(-1))
        l_nce = compute_infonce(nce_extras) if nce_extras is not None else 0.0

        loss = l_task + ib_loss + args.alpha_nce * l_nce

        if args.gradient_accumulation_step > 1:
            loss = loss / args.gradient_accumulation_step
        loss.backward()
        total_loss += loss.item()
        sum_task += l_task.item()
        sum_ib += ib_loss.item()
        sum_nce += (l_nce.item() if torch.is_tensor(l_nce) else l_nce)
        for k, v in loss_dict.items():
            sum_detail[k] = sum_detail.get(k, 0.0) + v
        steps += 1

        if (step + 1) % args.gradient_accumulation_step == 0:
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    n = max(steps, 1)
    detail = {k: v / n for k, v in sum_detail.items()}
    return total_loss / n, sum_task / n, sum_ib / n, sum_nce / n, detail


def eval_epoch(model, loader, stage=2):
    model.eval()
    total_loss, steps = 0.0, 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Val"):
            batch = tuple(t.to(DEVICE) for t in batch)
            input_ids, visual, acoustic, label_ids = batch
            visual = visual.squeeze(1)
            acoustic = acoustic.squeeze(1)

            v_norm = (visual - visual.min()) / (visual.max() - visual.min() + 1e-8)
            a_norm = (acoustic - acoustic.min()) / (acoustic.max() - acoustic.min() + 1e-8)

            logits, _, _, _ = model(input_ids, v_norm, a_norm, stage=stage)
            loss = F.l1_loss(logits.view(-1), label_ids.view(-1))
            total_loss += loss.item()
            steps += 1
    return total_loss / max(steps, 1)


def test_epoch(model, loader, stage=2):
    model.eval()
    preds, labels, all_w = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Test"):
            batch = tuple(t.to(DEVICE) for t in batch)
            input_ids, visual, acoustic, label_ids = batch
            visual = visual.squeeze(1)
            acoustic = acoustic.squeeze(1)

            v_norm = (visual - visual.min()) / (visual.max() - visual.min() + 1e-8)
            a_norm = (acoustic - acoustic.min()) / (acoustic.max() - acoustic.min() + 1e-8)

            logits, _, _, _ = model(input_ids, v_norm, a_norm, stage=stage)
            preds.extend(logits.view(-1).cpu().numpy().tolist())
            labels.extend(label_ids.view(-1).cpu().numpy().tolist())

    return np.array(preds).flatten(), np.array(labels).flatten()


def score(preds, y, use_zero=False):
    preds = preds.flatten()
    y = y.flatten()
    nz = np.array([i for i, e in enumerate(y) if e != 0 or use_zero])
    p, y2 = preds[nz], y[nz]
    mae = np.mean(np.abs(p - y2))
    corr = np.corrcoef(p, y2)[0][1] if len(p) > 1 else 0.0
    pb = p >= 0
    yb = y2 >= 0
    acc2 = accuracy_score(yb, pb)
    f1 = f1_score(yb, pb, average="weighted")
    p7 = np.clip(np.round(p), -3, 3).astype(int) + 3
    y7 = np.clip(np.round(y2), -3, 3).astype(int) + 3
    acc7 = accuracy_score(y7, p7)
    return acc2, acc7, mae, corr, f1


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("InfoGate Training")
    print(f"  Dataset        : {args.dataset}")
    print(f"  Epochs         : {args.n_epochs} (stage1: {args.stage1_epochs})")
    print(f"  Batch size     : {args.train_batch_size}")
    print(f"  Learning rate  : {args.learning_rate}")
    print(f"  InfoGate layers: {args.num_infogate_layers}")
    print(f"  Bottleneck dim : {args.bottleneck_dim}")
    print(f"  beta_ib        : {args.beta_ib}")
    print(f"  gamma_cyc      : {args.gamma_cyc}")
    print(f"  alpha_nce      : {args.alpha_nce}")
    print("=" * 60)

    set_seed(args.seed)
    train_dl, dev_dl, test_dl, n_opt = setup_data()
    model, optimizer, scheduler = build_model(n_opt)

    total_p = sum(p.numel() for p in model.parameters())
    train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total_p:,} total, {train_p:,} trainable")
    print("=" * 60)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(args.checkpoint_dir, f"infogate_{args.dataset}_best.pt")
    best_val = float('inf')
    best_results = None

    for epoch in range(args.n_epochs):
        stage = 1 if epoch < args.stage1_epochs else 2
        tr_loss, tr_task, tr_ib, tr_nce, tr_detail = train_epoch(
            model, train_dl, optimizer, scheduler, stage)
        val_loss = eval_epoch(model, dev_dl, stage=stage)

        print(f"\nEpoch {epoch + 1}/{args.n_epochs}  [stage {stage}]")
        print(f"  Loss  total={tr_loss:.4f}  task={tr_task:.4f}  "
              f"ib={tr_ib:.4f}  nce={tr_nce:.4f}")
        detail_str = "  ".join(f"{k}={v:.4f}" for k, v in tr_detail.items())
        print(f"  Detail  {detail_str}")
        print(f"  Val loss: {val_loss:.4f}")

        preds, labels = test_epoch(model, test_dl, stage=stage)
        acc2, acc7, mae, corr, f1 = score(preds, labels)
        print(f"  Test  Acc2={acc2*100:.2f}%  Acc7={acc7*100:.2f}%  "
              f"MAE={mae:.4f}  Corr={corr:.4f}  F1={f1:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            best_results = (acc2, acc7, mae, corr, f1)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'valid_loss': val_loss,
                'test_results': best_results,
                'args': args,
            }, ckpt_path)
            print(f"  >> Best model saved to {ckpt_path}")

    print("\n" + "=" * 60)
    print("Best Results:")
    if best_results:
        acc2, acc7, mae, corr, f1 = best_results
        print(f"  Acc-2: {acc2*100:.2f}%")
        print(f"  Acc-7: {acc7*100:.2f}%")
        print(f"  MAE:   {mae:.4f}")
        print(f"  Corr:  {corr:.4f}")
        print(f"  F1:    {f1:.4f}")


if __name__ == '__main__':
    main()
