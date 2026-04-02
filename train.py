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
parser.add_argument("--ig_learning_rate", type=float, default=5e-4)
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
parser.add_argument("--alpha_sac", type=float, default=0.1)
parser.add_argument("--mse_weight", type=float, default=0.5)
parser.add_argument("--cra_layers", type=int, default=8)
parser.add_argument("--cra_dims", default="64,32,16", type=str)

parser.add_argument("--ema_decay", type=float, default=0.999)
parser.add_argument("--ema_start_epoch", type=int, default=5)

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

    no_decay = {"bias", "LayerNorm.bias", "LayerNorm.weight"}

    backbone_prefix = "dberta.model."
    ig_lr = getattr(args, 'ig_learning_rate', 5e-4)

    backbone_decay, backbone_no_decay = [], []
    ig_decay, ig_no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_nd = any(nd in n for nd in no_decay)
        if n.startswith(backbone_prefix):
            (backbone_no_decay if is_nd else backbone_decay).append(p)
        else:
            (ig_no_decay if is_nd else ig_decay).append(p)

    groups = [
        {"params": backbone_decay,    "lr": args.learning_rate, "weight_decay": args.weight_decay},
        {"params": backbone_no_decay, "lr": args.learning_rate, "weight_decay": 0.0},
        {"params": ig_decay,          "lr": ig_lr,              "weight_decay": args.weight_decay},
        {"params": ig_no_decay,       "lr": ig_lr,              "weight_decay": 0.0},
    ]
    optimizer = AdamW(groups, lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_proportion * n_opt),
        num_training_steps=n_opt,
    )
    return model, optimizer, scheduler


# ============================================================
# EMA
# ============================================================

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {n: p.clone().detach()
                       for n, p in model.named_parameters() if p.requires_grad}
        self.backup = {}

    @torch.no_grad()
    def update(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.shadow[n].mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def apply(self, model):
        self.backup = {n: p.data.clone() for n, p in model.named_parameters()
                       if p.requires_grad and n in self.shadow}
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                p.data.copy_(self.shadow[n])

    def restore(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.backup:
                p.data.copy_(self.backup[n])
        self.backup = {}

    def state_dict(self):
        return {n: v.clone() for n, v in self.shadow.items()}

    def load_state_dict(self, state_dict):
        for n, v in state_dict.items():
            if n in self.shadow:
                self.shadow[n].copy_(v)

    def reset(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.shadow[n].copy_(p.data)


# ============================================================
# Sentiment-Aware Contrastive + Ordinal Ranking Losses
# ============================================================

def compute_sentiment_contrastive(h_fused, labels, temperature=0.1):
    """Sentiment-distance-weighted contrastive loss in bottleneck space."""
    B = h_fused.size(0)
    if B < 2:
        return h_fused.new_tensor(0.0)

    h_norm = F.normalize(h_fused, dim=-1)
    sim = torch.mm(h_norm, h_norm.t())

    label_dist = torch.abs(labels.unsqueeze(0) - labels.unsqueeze(1))
    target_sim = torch.exp(-label_dist / temperature)

    mask = ~torch.eye(B, dtype=torch.bool, device=h_fused.device)
    return F.mse_loss(sim[mask], target_sim[mask])


# ============================================================
# Train / Eval / Test
# ============================================================

def train_epoch(model, loader, optimizer, scheduler, stage, ema=None):
    model.train()
    total_loss, steps = 0.0, 0
    sum_task, sum_ib = 0.0, 0.0
    sum_detail = {}

    train_pbar = tqdm(loader, desc=f"Train (stage {stage})")
    for step, batch in enumerate(train_pbar):
        batch = tuple(t.to(DEVICE) for t in batch)
        input_ids, visual, acoustic, label_ids = batch
        visual = visual.squeeze(1)
        acoustic = acoustic.squeeze(1)

        v_norm = (visual - visual.min()) / (visual.max() - visual.min() + 1e-8)
        a_norm = (acoustic - acoustic.min()) / (acoustic.max() - acoustic.min() + 1e-8)

        logits, ib_loss, loss_dict, h_pooled = model(
            input_ids, v_norm, a_norm, labels=label_ids, stage=stage)

        pred_flat = logits.view(-1)
        label_flat = label_ids.view(-1)
        
        # Log prediction distribution
        loss_dict['pred_mean'] = pred_flat.mean().item()
        loss_dict['pred_std'] = pred_flat.std().item()

        l_task = F.l1_loss(pred_flat, label_flat) + args.mse_weight * F.mse_loss(pred_flat, label_flat)

        l_sac = compute_sentiment_contrastive(h_pooled, label_flat) if h_pooled is not None else 0.0

        loss = l_task + ib_loss + args.alpha_sac * l_sac

        if args.gradient_accumulation_step > 1:
            loss = loss / args.gradient_accumulation_step
        loss.backward()
        total_loss += loss.item()
        sum_task += l_task.item()
        sum_ib += ib_loss.item()
        for k, v in loss_dict.items():
            sum_detail[k] = sum_detail.get(k, 0.0) + v
        steps += 1
        
        # Real-time progress monitoring
        train_pbar.set_postfix({"task": f"{l_task.item():.3f}", "ib": f"{ib_loss.item():.3f}", "p_std": f"{loss_dict['pred_std']:.2f}"})

        if (step + 1) % args.gradient_accumulation_step == 0:
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            if ema is not None:
                ema.update(model)

    n = max(steps, 1)
    detail = {k: v / n for k, v in sum_detail.items()}
    return total_loss / n, sum_task / n, sum_ib / n, detail


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
            pred_flat = logits.view(-1)
            label_flat = label_ids.view(-1)
            loss = F.l1_loss(pred_flat, label_flat) + args.mse_weight * F.mse_loss(pred_flat, label_flat)
            total_loss += loss.item()
            steps += 1
    return total_loss / max(steps, 1)


def test_epoch(model, loader, stage=2, desc="Test"):
    model.eval()
    preds, labels, all_w = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
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
    print(f"  LR (backbone)  : {args.learning_rate}")
    print(f"  LR (InfoGate)  : {args.ig_learning_rate}")
    print(f"  InfoGate layers: {args.num_infogate_layers}")
    print(f"  Bottleneck dim : {args.bottleneck_dim}")
    print(f"  beta_ib        : {args.beta_ib}")
    print(f"  gamma_cyc      : {args.gamma_cyc}")
    print(f"  mse_weight     : {args.mse_weight}")
    print("=" * 60)

    set_seed(args.seed)
    train_dl, dev_dl, test_dl, n_opt = setup_data()
    model, optimizer, scheduler = build_model(n_opt)
    ema = EMA(model, decay=args.ema_decay)

    total_p = sum(p.numel() for p in model.parameters())
    train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total_p:,} total, {train_p:,} trainable")
    print(f"EMA: decay={args.ema_decay}, start_epoch={args.ema_start_epoch}")
    print("=" * 60)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(args.checkpoint_dir, f"infogate_{args.dataset}_best.pt")
    select_start = args.n_epochs // 2
    best_dev_score = float('inf')
    best_results = None
    best_test_mae = float('inf')
    best_test_results = None
    last_test_results = None

    for epoch in range(args.n_epochs):
        stage = 1 if epoch < args.stage1_epochs else 2
        eval_with_ema = epoch >= args.ema_start_epoch
        if epoch == args.ema_start_epoch:
            ema.reset(model)
        tr_loss, tr_task, tr_ib, tr_detail = train_epoch(
            model, train_dl, optimizer, scheduler, stage,
            ema=ema if eval_with_ema else None)

        print(f"\nEpoch {epoch + 1}/{args.n_epochs}  [stage {stage}]"
              f"{'  (EMA eval)' if eval_with_ema else ''}")
        print(f"  Loss  total={tr_loss:.4f}  task={tr_task:.4f}  "
              f"ib={tr_ib:.4f}")
        detail_str = "  ".join(f"{k}={v:.4f}" for k, v in tr_detail.items()
                                if k.startswith('L_'))
        print(f"  Detail  {detail_str}")
        # Diagnostics: MSelector weights, primary selection, confidence, and prediction bounds
        diag_keys = ['w_acoustic', 'w_language', 'w_visual',
                     'primary_a', 'primary_l', 'primary_v',
                     'conf_t', 'conf_a', 'conf_v', 'fusion_conf',
                     'pred_mean', 'pred_std']
        diag_vals = {k: tr_detail[k] for k in diag_keys if k in tr_detail}
        if diag_vals:
            w_str = (f"w=[a:{diag_vals.get('w_acoustic',0):.3f} "
                     f"l:{diag_vals.get('w_language',0):.3f} "
                     f"v:{diag_vals.get('w_visual',0):.3f}]")
            p_str = (f"primary=[a:{diag_vals.get('primary_a',0):.2f} "
                     f"l:{diag_vals.get('primary_l',0):.2f} "
                     f"v:{diag_vals.get('primary_v',0):.2f}]")
            c_str = (f"conf=[t:{diag_vals.get('conf_t',0):.3f} "
                     f"a:{diag_vals.get('conf_a',0):.3f} "
                     f"v:{diag_vals.get('conf_v',0):.3f} "
                     f"fused:{diag_vals.get('fusion_conf',0):.3f}]")
            pred_str = (f"pred=[mean:{diag_vals.get('pred_mean',0):.3f} "
                        f"std:{diag_vals.get('pred_std',0):.3f}]")
            print(f"  Diag  {w_str}  {p_str}  {c_str}\n  Stats {pred_str}")

        if eval_with_ema:
            ema.apply(model)

        dev_preds, dev_labels = test_epoch(model, dev_dl, stage=stage, desc="Dev")
        dev_acc2, dev_acc7, dev_mae, dev_corr, dev_f1 = score(dev_preds, dev_labels)
        print(f"  Dev   Acc2={dev_acc2*100:.2f}%  Acc7={dev_acc7*100:.2f}%  "
              f"MAE={dev_mae:.4f}  Corr={dev_corr:.4f}  F1={dev_f1:.4f}")

        preds, labels = test_epoch(model, test_dl, stage=stage)
        acc2, acc7, mae, corr, f1 = score(preds, labels)
        print(f"  Test  Acc2={acc2*100:.2f}%  Acc7={acc7*100:.2f}%  "
              f"MAE={mae:.4f}  Corr={corr:.4f}  F1={f1:.4f}")

        last_test_results = (acc2, acc7, mae, corr, f1)

        if epoch >= select_start:
            dev_score = dev_mae - 0.5 * dev_corr
            if dev_score < best_dev_score:
                best_dev_score = dev_score
                best_results = (acc2, acc7, mae, corr, f1)
                save_dict = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'dev_mae': dev_mae,
                    'dev_corr': dev_corr,
                    'dev_score': dev_score,
                    'test_results': best_results,
                    'args': args,
                }
                if eval_with_ema:
                    save_dict['ema_state_dict'] = ema.state_dict()
                torch.save(save_dict, ckpt_path)
                print(f"  >> Best model saved (dev score={dev_score:.4f}, "
                      f"MAE={dev_mae:.4f}, Corr={dev_corr:.4f}) to {ckpt_path}")

        if mae < best_test_mae:
            best_test_mae = mae
            best_test_results = (acc2, acc7, mae, corr, f1, epoch + 1)

        if eval_with_ema:
            ema.restore(model)

    print("\n" + "=" * 60)
    print(f"Best Results (dev score, epoch >= {select_start + 1}):")
    if best_results:
        acc2, acc7, mae, corr, f1 = best_results
        print(f"  Acc-2: {acc2*100:.2f}%")
        print(f"  Acc-7: {acc7*100:.2f}%")
        print(f"  MAE:   {mae:.4f}")
        print(f"  Corr:  {corr:.4f}")
        print(f"  F1:    {f1:.4f}")
    print(f"\nLast Epoch ({args.n_epochs}) Results:")
    if last_test_results:
        acc2, acc7, mae, corr, f1 = last_test_results
        print(f"  Acc-2: {acc2*100:.2f}%")
        print(f"  Acc-7: {acc7*100:.2f}%")
        print(f"  MAE:   {mae:.4f}")
        print(f"  Corr:  {corr:.4f}")
        print(f"  F1:    {f1:.4f}")
    print("\nBest Test MAE (oracle, for reference only):")
    if best_test_results:
        acc2, acc7, mae, corr, f1, ep = best_test_results
        print(f"  Epoch: {ep}")
        print(f"  Acc-2: {acc2*100:.2f}%")
        print(f"  Acc-7: {acc7*100:.2f}%")
        print(f"  MAE:   {mae:.4f}")
        print(f"  Corr:  {corr:.4f}")
        print(f"  F1:    {f1:.4f}")


if __name__ == '__main__':
    main()
