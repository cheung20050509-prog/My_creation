"""InfoGate training script for complete-modality optimization.

Copy vendored under ``fixed_experiment/`` for MOSI trial 234 frozen runs.
Paths to ``deberta-v3-base`` and ``datasets/*.pkl`` resolve to ``My_creation/`` parent.
"""

import argparse
import math
import os
import random
import pickle
import sys
import numpy as np

from sklearn.metrics import accuracy_score, f1_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from transformers import get_linear_schedule_with_warmup, DebertaV2Tokenizer, BertTokenizer
from torch.optim import AdamW

from deberta_infogate import InfoGate_DeBertaForSequenceClassification
from bert_infogate import InfoGate_BertForSequenceClassification
import global_configs
from global_configs import DEVICE
from simsv2_metrics import compute_simsv2_kuda_metrics
from selection_utils import (
    DEFAULT_SELECTION_METRIC,
    SELECTION_METRIC_CHOICES,
    build_selection_tiebreak,
    compute_selection_score,
    selection_higher_is_better,
)

# Snapshot copy under fixed_experiment/: tokenizer weights + pickles live in My_creation/.
_FIXED_EXP_DIR = os.path.dirname(os.path.abspath(__file__))
_MY_CREATION_DIR = os.path.dirname(_FIXED_EXP_DIR)
if _MY_CREATION_DIR not in sys.path:
    sys.path.insert(0, _MY_CREATION_DIR)

from simsv2_mmsa_data import (
    build_tensor_dataset as build_simsv2_mmsa_dataset,
    format_seq_lens_report,
    unpack_batch as unpack_simsv2_batch,
    uses_simsv2_mmsa,
)

# ============================================================
# CLI
# ============================================================
parser = argparse.ArgumentParser(description="InfoGate Training")
parser.add_argument("--model", type=str,
                    default=os.path.join(_MY_CREATION_DIR, "deberta-v3-base"))
parser.add_argument("--dataset", type=str, choices=["mosi", "mosei", "simsv2"], default="mosi")
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
parser.add_argument("--alpha_ib", type=float, default=0.01)
parser.add_argument("--selector_target_temp", type=float, default=0.35,
                    help="Temperature for modality-quality routing targets.")
parser.add_argument("--selector_balance_weight", type=float, default=0.0,
                    help="Batch-level routing entropy regularization weight.")
parser.add_argument("--selector_rib_weight", type=float, default=0.05,
                    help="Overall weight of the routing supervision loss.")
parser.add_argument("--disable_l_lib", action="store_true",
                    help="Ablate the label-level IB loss.")
parser.add_argument("--disable_l_rib", action="store_true",
                    help="Ablate the routing IB prior loss.")
parser.add_argument("--mse_weight", type=float, default=0.5)

parser.add_argument("--ema_decay", type=float, default=0.999)
parser.add_argument("--ema_start_epoch", type=int, default=5)

parser.add_argument("--gumbel_tau_start", type=float, default=1.0,
                    help="Gumbel-Softmax temperature at start of training.")
parser.add_argument("--gumbel_tau_end", type=float, default=0.5,
                    help="Gumbel-Softmax temperature at end of training (annealed).")

parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
parser.add_argument("--selection_metric", type=str,
                    default=DEFAULT_SELECTION_METRIC,
                    choices=SELECTION_METRIC_CHOICES)
parser.add_argument("--early_stop_patience", type=int, default=0,
                    help="Stop after this many dev epochs (epoch>=stage1) without "
                         "selection-metric improvement; 0 disables.")
parser.add_argument("--early_stop_min_delta", type=float, default=0.0,
                    help="When >0, a dev improvement must exceed this margin on the "
                         "selection score (tiebreak-only wins still count).")
parser.add_argument(
    "--simsv2_feature_mode",
    type=str,
    default="mmsa",
    choices=["mmsa", "legacy"],
    help="SIMSv2 only: mmsa uses text_bert + A/V lengths; legacy re-tokenizes raw_text.",
)

args = parser.parse_args()

if args.dataset == "simsv2":
    if "deberta-v3-base" in args.model:
        args.model = os.path.join(_MY_CREATION_DIR, "bert-base-chinese")

if isinstance(args.model, str):
    pass # Added to ensure valid code block

args.use_l_lib = not args.disable_l_lib
args.use_l_rib = not args.disable_l_rib

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


def simsv2_mmsa_enabled():
    return uses_simsv2_mmsa(args.dataset, args.simsv2_feature_mode)


def convert_to_features(examples, max_seq_length, tokenizer):
    features = []
    if args.dataset == "simsv2" and simsv2_mmsa_enabled():
        raise RuntimeError("convert_to_features should not be used when simsv2_feature_mode=mmsa")
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

            tokens, inversions = [] , []
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


def get_dataset(data):
    if simsv2_mmsa_enabled():
        return build_simsv2_mmsa_dataset(data, args.max_seq_length)
    tok = get_tokenizer(args.model)
    feats = convert_to_features(data, args.max_seq_length, tok)
    return TensorDataset(
        torch.tensor(np.array([f.input_ids for f in feats]), dtype=torch.long),
        torch.tensor(np.array([f.visual for f in feats]), dtype=torch.float),
        torch.tensor(np.array([f.acoustic for f in feats]), dtype=torch.float),
        torch.tensor(np.array([f.label_id for f in feats]), dtype=torch.float),
    )


def setup_data():
    ds_path = os.path.join(_MY_CREATION_DIR, "datasets", f"{args.dataset}.pkl")
    with open(ds_path, "rb") as fh:
        data = pickle.load(fh)

    if simsv2_mmsa_enabled():
        print(f"  SIMSv2 data    : MMSA fields (text_bert + A/V lengths)")
        print(f"  {format_seq_lens_report(data['train'], args.max_seq_length)}")

    train_ds = get_dataset(data["train"])
    if "dev" in data:
        dev_ds = get_dataset(data["dev"])
    elif "valid" in data:
        dev_ds = get_dataset(data["valid"])
    else:
        raise KeyError("Could not find validation data split ('dev' or 'valid')")
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
    if args.dataset == "simsv2":
        model = InfoGate_BertForSequenceClassification.from_pretrained(
            args.model, multimodal_config=args, num_labels=1)
        backbone_prefix = "bert.model."
    else:
        model = InfoGate_DeBertaForSequenceClassification.from_pretrained(
            args.model, multimodal_config=args, num_labels=1)
        backbone_prefix = "dberta.model."
    model.to(DEVICE)

    no_decay = {"bias", "LayerNorm.bias", "LayerNorm.weight"}
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


def toggle_state(enabled):
    return "on" if enabled else "off"


def _forward_model(model, input_ids, visual, acoustic, label_ids, stage,
                   input_mask=None, segment_ids=None):
    fwd_kw = dict(labels=label_ids, stage=stage)
    if simsv2_mmsa_enabled():
        fwd_kw["attention_mask"] = input_mask
        fwd_kw["token_type_ids"] = segment_ids
    return model(input_ids, visual, acoustic, **fwd_kw)


# ============================================================
# Train / Eval / Test
# ============================================================

def train_epoch(model, loader, optimizer, scheduler, stage, ema=None):
    model.train()
    total_loss, steps = 0.0, 0
    sum_task, sum_ib = 0.0, 0.0
    sum_detail = {}
    use_mmsa = simsv2_mmsa_enabled()

    train_pbar = tqdm(loader, desc=f"Train (stage {stage})")
    for step, batch in enumerate(train_pbar):
        batch = tuple(t.to(DEVICE) for t in batch)
        input_ids, visual, acoustic, label_ids, input_mask, segment_ids = unpack_simsv2_batch(
            batch, use_mmsa)
        visual = visual.squeeze(1)
        acoustic = acoustic.squeeze(1)

        logits, ib_loss, loss_dict, _ = _forward_model(
            model, input_ids, visual, acoustic, label_ids, stage,
            input_mask, segment_ids)

        pred_flat = logits.view(-1)
        label_flat = label_ids.view(-1)
        
        # Log prediction distribution
        loss_dict['pred_mean'] = pred_flat.mean().item()
        loss_dict['pred_std'] = pred_flat.std(unbiased=False).item()

        l_task = F.l1_loss(pred_flat, label_flat) + args.mse_weight * F.mse_loss(pred_flat, label_flat)

        loss = l_task + ib_loss

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


def test_epoch(model, loader, stage=2, desc="Test"):
    model.eval()
    preds, labels, all_w = [], [], []
    use_mmsa = simsv2_mmsa_enabled()
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            batch = tuple(t.to(DEVICE) for t in batch)
            input_ids, visual, acoustic, label_ids, input_mask, segment_ids = unpack_simsv2_batch(
                batch, use_mmsa)
            visual = visual.squeeze(1)
            acoustic = acoustic.squeeze(1)

            logits, _, _, _ = _forward_model(
                model, input_ids, visual, acoustic, label_ids, stage,
                input_mask, segment_ids)
            preds.extend(logits.view(-1).cpu().numpy().tolist())
            labels.extend(label_ids.view(-1).cpu().numpy().tolist())

    return np.array(preds).flatten(), np.array(labels).flatten()


def score(preds, y, use_zero=False):
    preds = np.asarray(preds).flatten()
    y = np.asarray(y).flatten()

    if args.dataset == "simsv2":
        return compute_simsv2_kuda_metrics(preds, y)

    # Full set for MAE, Corr, Acc7
    mae = np.mean(np.abs(preds - y))
    corr = np.corrcoef(preds, y)[0][1] if len(preds) > 1 else 0.0

    # Filtered set for Acc2, F1
    nz = np.array([i for i, e in enumerate(y) if e != 0])
    p_nz, y_nz = preds[nz], y[nz]
    pb = p_nz >= 0
    yb = y_nz >= 0
    acc2 = accuracy_score(yb, pb)
    f1 = f1_score(yb, pb, average="weighted")

    result = {"acc2": acc2, "mae": mae, "corr": corr, "f1": f1}
    p7 = np.clip(np.round(preds), -3, 3).astype(int) + 3
    y7 = np.clip(np.round(y), -3, 3).astype(int) + 3
    result["acc7"] = accuracy_score(y7, p7)
    return result


def fmt_metrics(m, prefix=""):
    """Format metrics dict for printing."""
    s = f"{prefix}Acc2={m['acc2']*100:.2f}%"
    if "acc5" in m:
        s += f"  Acc5={m['acc5']*100:.2f}%  Acc3={m['acc3']*100:.2f}%"
    else:
        s += f"  Acc7={m['acc7']*100:.2f}%"
    s += f"  MAE={m['mae']:.4f}  Corr={m['corr']:.4f}  F1={m['f1']:.4f}"
    return s


def selection_kwargs(m):
    """Build keyword args for compute_selection_score / build_selection_tiebreak."""
    kw = dict(acc2=m["acc2"], mae=m["mae"], corr=m["corr"], f1=m["f1"])
    if "acc7" in m:
        kw["acc7"] = m["acc7"]
    if "acc5" in m:
        kw["acc5"] = m["acc5"]
    if "acc3" in m:
        kw["acc3"] = m["acc3"]
    return kw


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("InfoGate Training")
    print(f"  Dataset        : {args.dataset}")
    if args.dataset == "simsv2":
        print(f"  SIMSv2 mode    : {args.simsv2_feature_mode}")
    print(f"  Epochs         : {args.n_epochs} (stage1: {args.stage1_epochs})")
    print(f"  Batch size     : {args.train_batch_size}")
    print(f"  LR (backbone)  : {args.learning_rate}")
    print(f"  LR (InfoGate)  : {args.ig_learning_rate}")
    print(f"  InfoGate layers: {args.num_infogate_layers}")
    print(f"  Bottleneck dim : {args.bottleneck_dim}")
    print(f"  beta_ib        : {args.beta_ib}")
    print(f"  mse_weight     : {args.mse_weight}")
    print(f"  selector_temp  : {args.selector_target_temp}")
    print(f"  selector_bal   : {args.selector_balance_weight}")
    print(f"  selector_rib_w : {args.selector_rib_weight}")
    print(f"  Select by      : {args.selection_metric}")
    print(f"  Loss terms     : L_lib={toggle_state(args.use_l_lib)} "
            f"L_rib={toggle_state(args.use_l_rib)}")
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
    select_start = args.stage1_epochs
    select_higher_is_better = selection_higher_is_better(args.selection_metric)
    best_selection_score = float('-inf') if select_higher_is_better else float('inf')
    best_selection_tiebreak = None
    best_results = None
    best_test_mae = float('inf')
    best_test_results = None
    last_test_results = None
    last_completed_epoch = 0
    early_patience = max(0, int(getattr(args, "early_stop_patience", 0) or 0))
    early_min_delta = float(getattr(args, "early_stop_min_delta", 0.0) or 0.0)
    no_improve_epochs = 0

    for epoch in range(args.n_epochs):
        stage = 1 if epoch < args.stage1_epochs else 2
        # Gumbel-Softmax tau annealing: linear decay from start to end
        tau_s = getattr(args, 'gumbel_tau_start', 1.0)
        tau_e = getattr(args, 'gumbel_tau_end', 0.5)
        tau = tau_s + (tau_e - tau_s) * epoch / max(args.n_epochs - 1, 1)
        # Update tau on the model's DPR router (``mselector``)
        m = model.module if hasattr(model, 'module') else model
        ig = m.dberta.infogate if hasattr(m, 'dberta') else m.bert.infogate
        ig.mselector.gumbel_tau = tau

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
        # Diagnostics: DPR weights, primary selection, confidence, and prediction bounds
        diag_keys = ['w_acoustic', 'w_language', 'w_visual',
                     'target_acoustic', 'target_language', 'target_visual',
                     'primary_a', 'primary_l', 'primary_v',
                     'err_acoustic', 'err_language', 'err_visual',
                     'conf_t', 'conf_a', 'conf_v', 'fusion_conf',
                     'routing_entropy', 'pred_mean', 'pred_std']
        diag_vals = {k: tr_detail[k] for k in diag_keys if k in tr_detail}
        if diag_vals:
            w_str = (f"w=[a:{diag_vals.get('w_acoustic',0):.3f} "
                     f"l:{diag_vals.get('w_language',0):.3f} "
                     f"v:{diag_vals.get('w_visual',0):.3f}]")
            tgt_str = (f"target=[a:{diag_vals.get('target_acoustic',0):.3f} "
                       f"l:{diag_vals.get('target_language',0):.3f} "
                       f"v:{diag_vals.get('target_visual',0):.3f}]")
            p_str = (f"primary=[a:{diag_vals.get('primary_a',0):.2f} "
                     f"l:{diag_vals.get('primary_l',0):.2f} "
                     f"v:{diag_vals.get('primary_v',0):.2f}]")
            err_str = (f"err=[a:{diag_vals.get('err_acoustic',0):.3f} "
                       f"l:{diag_vals.get('err_language',0):.3f} "
                       f"v:{diag_vals.get('err_visual',0):.3f}]")
            c_str = (f"conf=[t:{diag_vals.get('conf_t',0):.3f} "
                     f"a:{diag_vals.get('conf_a',0):.3f} "
                     f"v:{diag_vals.get('conf_v',0):.3f} "
                     f"fused:{diag_vals.get('fusion_conf',0):.3f}]")
            rib_str = f"route_H={diag_vals.get('routing_entropy',0):.3f}"
            pred_str = (f"pred=[mean:{diag_vals.get('pred_mean',0):.3f} "
                        f"std:{diag_vals.get('pred_std',0):.3f}]")
            print(f"  Diag  {w_str}  {tgt_str}  {p_str}\n"
                  f"  Qual  {err_str}  {c_str}  {rib_str}  gumbel_tau={tau:.3f}\n"
                  f"  Stats {pred_str}")

        if eval_with_ema:
            ema.apply(model)

        dev_preds, dev_labels = test_epoch(model, dev_dl, stage=stage, desc="Dev")
        dev_m = score(dev_preds, dev_labels)
        dev_kw = selection_kwargs(dev_m)
        selection_score = compute_selection_score(args.selection_metric, **dev_kw)
        selection_tiebreak = build_selection_tiebreak(**dev_kw)
        print(f"  Dev   {fmt_metrics(dev_m)}")
        score_str = f"acc2_composite={selection_score:.4f}" if args.selection_metric == "acc2_composite" else f"{args.selection_metric}={selection_score:.6f}"
        print(f"  Select {score_str}")

        preds, labels = test_epoch(model, test_dl, stage=stage)
        test_m = score(preds, labels)
        print(f"  Test  {fmt_metrics(test_m)}")

        last_test_results = test_m

        if epoch >= select_start:
            prev_best_selection_score = best_selection_score
            prev_best_selection_tiebreak = best_selection_tiebreak
            if best_selection_tiebreak is None:
                should_save = True
            else:
                better_score = selection_score > best_selection_score if select_higher_is_better else selection_score < best_selection_score
                same_score = abs(selection_score - best_selection_score) <= 1e-12
                should_save = better_score or (same_score and selection_tiebreak > best_selection_tiebreak)

            if should_save:
                best_selection_score = selection_score
                best_selection_tiebreak = selection_tiebreak
                best_results = test_m
                save_dict = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'dev_mae': dev_m["mae"],
                    'dev_corr': dev_m["corr"],
                    'selection_metric': args.selection_metric,
                    'selection_score': selection_score,
                    'ablation': {
                        'use_l_lib': args.use_l_lib,
                        'use_l_rib': args.use_l_rib,
                    },
                    'test_results': best_results,
                    'args': args,
                }
                if eval_with_ema:
                    save_dict['ema_state_dict'] = ema.state_dict()
                torch.save(save_dict, ckpt_path)
                print(f"  >> Best model saved ({args.selection_metric}={selection_score:.6f}, "
                      f"MAE={dev_m['mae']:.4f}, Corr={dev_m['corr']:.4f}) to {ckpt_path}")

            if early_patience > 0:
                improved_for_early_stop = should_save
                if early_min_delta > 0.0 and should_save and prev_best_selection_tiebreak is not None:
                    same_score_prev = abs(selection_score - prev_best_selection_score) <= 1e-12
                    if same_score_prev:
                        improved_for_early_stop = True
                    elif math.isfinite(prev_best_selection_score):
                        if select_higher_is_better:
                            improved_for_early_stop = (
                                selection_score - prev_best_selection_score
                            ) > early_min_delta
                        else:
                            improved_for_early_stop = (
                                prev_best_selection_score - selection_score
                            ) > early_min_delta
                    else:
                        improved_for_early_stop = True
                if improved_for_early_stop:
                    no_improve_epochs = 0
                else:
                    no_improve_epochs += 1

        if test_m["mae"] < best_test_mae:
            best_test_mae = test_m["mae"]
            best_test_results = {**test_m, "epoch": epoch + 1}

        if eval_with_ema:
            ema.restore(model)

        last_completed_epoch = epoch + 1

        if (
            early_patience > 0
            and epoch >= select_start
            and no_improve_epochs >= early_patience
        ):
            print(
                f"\n  >> Early stop: no dev selection improvement for "
                f"{early_patience} epoch(s) (epoch {epoch + 1}, "
                f"best {args.selection_metric}={best_selection_score:.6f})."
            )
            break

    print("\n" + "=" * 60)
    acc_label_hi = "Acc-5" if args.dataset == "simsv2" else "Acc-7"
    print(f"Best Results ({args.selection_metric}, epoch >= {select_start + 1}):")
    if best_results:
        print(f"  Selection score: {best_selection_score:.6f}")
        print(f"  Acc-2: {best_results['acc2']*100:.2f}%")
        if args.dataset == "simsv2":
            print(f"  Acc-5: {best_results['acc5']*100:.2f}%")
            print(f"  Acc-3: {best_results['acc3']*100:.2f}%")
        else:
            print(f"  Acc-7: {best_results['acc7']*100:.2f}%")
        print(f"  MAE:   {best_results['mae']:.4f}")
        print(f"  Corr:  {best_results['corr']:.4f}")
        print(f"  F1:    {best_results['f1']:.4f}")
    le = last_completed_epoch if last_completed_epoch > 0 else args.n_epochs
    print(f"\nLast Epoch ({le}) Results:")
    if last_test_results:
        print(f"  Acc-2: {last_test_results['acc2']*100:.2f}%")
        if args.dataset == "simsv2":
            print(f"  Acc-5: {last_test_results['acc5']*100:.2f}%")
            print(f"  Acc-3: {last_test_results['acc3']*100:.2f}%")
        else:
            print(f"  Acc-7: {last_test_results['acc7']*100:.2f}%")
        print(f"  MAE:   {last_test_results['mae']:.4f}")
        print(f"  Corr:  {last_test_results['corr']:.4f}")
        print(f"  F1:    {last_test_results['f1']:.4f}")
    print("\nBest Test MAE (oracle, for reference only):")
    if best_test_results:
        print(f"  Epoch: {best_test_results['epoch']}")
        print(f"  Acc-2: {best_test_results['acc2']*100:.2f}%")
        if args.dataset == "simsv2":
            print(f"  Acc-5: {best_test_results['acc5']*100:.2f}%")
            print(f"  Acc-3: {best_test_results['acc3']*100:.2f}%")
        else:
            print(f"  Acc-7: {best_test_results['acc7']*100:.2f}%")
        print(f"  MAE:   {best_test_results['mae']:.4f}")
        print(f"  Corr:  {best_test_results['corr']:.4f}")
        print(f"  F1:    {best_test_results['f1']:.4f}")


if __name__ == '__main__':
    main()
