"""InfoGate test script for complete-modality evaluation."""

import argparse
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

args = parser.parse_args()

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
        missing = [k for k in model.state_dict() if k not in sd]
        if missing:
            print(f"  {len(missing)} keys missing — loading with strict=False")
            model.load_state_dict(sd, strict=False)
        else:
            model.load_state_dict(sd)
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


def compute_metrics(preds, labels, dataset="mosi", use_zero=False):
    preds = np.asarray(preds).flatten()
    labels = np.asarray(labels).flatten()
    nz = np.array([i for i, e in enumerate(labels) if e != 0 or use_zero])
    p, y = preds[nz], labels[nz]
    mae = np.mean(np.abs(p - y))
    corr = np.corrcoef(p, y)[0][1]
    pb, yb = (p >= 0), (y >= 0)
    acc2 = accuracy_score(yb, pb)
    f1 = f1_score(yb, pb, average="weighted")
    result = {'MAE': mae, 'Corr': corr, 'Acc2': acc2, 'F1': f1}
    if dataset == "simsv2":
        p5 = np.clip(np.round(p * 2), -2, 2).astype(int)
        y5 = np.clip(np.round(y * 2), -2, 2).astype(int)
        result['Acc5'] = accuracy_score(y5, p5)
        p3 = np.sign(preds).astype(int)
        y3 = np.sign(labels).astype(int)
        result['Acc3'] = accuracy_score(y3, p3)
    else:
        p7 = np.clip(np.round(p), -3, 3).astype(int)
        y7 = np.clip(np.round(y), -3, 3).astype(int)
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


if __name__ == "__main__":
    main()
