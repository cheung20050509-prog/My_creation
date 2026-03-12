"""
InfoGate test script.
Supports complete-modality and missing-modality evaluation protocols.
"""

import argparse
import os
import random
import pickle
import numpy as np

from sklearn.metrics import accuracy_score, f1_score

import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from transformers import DebertaV2Tokenizer
from deberta_infogate import InfoGate_DeBertaForSequenceClassification
import global_configs
from global_configs import DEVICE

# ============================================================
# CLI
# ============================================================
parser = argparse.ArgumentParser(description="InfoGate Testing")
parser.add_argument("--model", type=str,
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "deberta-v3-base"))
parser.add_argument("--dataset", type=str, choices=["mosi", "mosei"], default="mosi")
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
parser.add_argument("--gamma_cyc", type=float, default=10)
parser.add_argument("--cra_layers", type=int, default=8)
parser.add_argument("--cra_dims", default="64,32,16", type=str)

parser.add_argument("--checkpoint", type=str,
                    default="checkpoints/infogate_mosi_best.pt")
parser.add_argument("--complete_only", action="store_true")
parser.add_argument("--missing_modality", type=str, default=None,
                    choices=["text", "acoustic", "visual", "ta", "tv", "av"])
parser.add_argument("--missing_rate", type=float, default=0.0)

args = parser.parse_args()

if isinstance(args.cra_dims, str):
    args.cra_dims = [int(x) for x in args.cra_dims.split(',')]

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


def get_test_dataloader():
    with open(f"datasets/{args.dataset}.pkl", "rb") as fh:
        data = pickle.load(fh)
    tok = DebertaV2Tokenizer.from_pretrained(args.model)
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
# Modality mask
# ============================================================

def build_modality_mask(batch_size, device, missing_modality=None, missing_rate=0.0):
    mask = torch.ones(batch_size, 3, device=device)
    idx_map = {'t': 0, 'a': 1, 'v': 2}
    fixed_map = {
        'text': ('t',), 'acoustic': ('a',), 'visual': ('v',),
        'ta': ('t', 'a'), 'tv': ('t', 'v'), 'av': ('a', 'v'),
    }
    if missing_modality is not None:
        for m in fixed_map[missing_modality]:
            mask[:, idx_map[m]] = 0.0
    elif missing_rate > 0.0:
        patterns = [('t',), ('a',), ('v',), ('t', 'a'), ('t', 'v'), ('a', 'v')]
        for i in range(batch_size):
            if random.random() < missing_rate:
                for m in random.choice(patterns):
                    mask[i, idx_map[m]] = 0.0
    return mask


# ============================================================
# Evaluation
# ============================================================

def test_model(model, loader, missing_modality=None, missing_rate=0.0):
    preds, labels = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Testing"):
            batch = tuple(t.to(DEVICE) for t in batch)
            input_ids, visual, acoustic, label_ids = batch
            visual = visual.squeeze(1)
            acoustic = acoustic.squeeze(1)

            mm = build_modality_mask(
                input_ids.size(0), input_ids.device,
                missing_modality=missing_modality,
                missing_rate=missing_rate)

            v_n = (visual - visual.min()) / (visual.max() - visual.min() + 1e-8)
            a_n = (acoustic - acoustic.min()) / (acoustic.max() - acoustic.min() + 1e-8)

            logits, _, _, _ = model(
                input_ids, v_n, a_n, stage=2, modality_mask=mm)

            logits = logits.squeeze(-1).cpu().numpy()
            label_ids = label_ids.cpu().numpy().flatten()

            preds.extend(logits.tolist() if logits.ndim > 0 else [logits.item()])
            labels.extend(label_ids.tolist())

    return np.array(preds), np.array(labels)


def compute_metrics(preds, labels, use_zero=False):
    nz = np.array([i for i, e in enumerate(labels) if e != 0 or use_zero])
    p, y = preds[nz], labels[nz]
    mae = np.mean(np.abs(p - y))
    corr = np.corrcoef(p, y)[0][1]
    pb, yb = (p >= 0), (y >= 0)
    acc2 = accuracy_score(yb, pb)
    f1 = f1_score(yb, pb, average="weighted")
    p7 = np.clip(np.round(p), -3, 3).astype(int)
    y7 = np.clip(np.round(y), -3, 3).astype(int)
    acc7 = accuracy_score(y7, p7)
    return {'MAE': mae, 'Corr': corr, 'Acc2': acc2, 'Acc7': acc7, 'F1': f1}


def print_metrics(metrics, prefix=""):
    print(f"{prefix}Acc2: {metrics['Acc2']:.4f}  Acc7: {metrics['Acc7']:.4f}  "
          f"F1: {metrics['F1']:.4f}  MAE: {metrics['MAE']:.4f}  "
          f"Corr: {metrics['Corr']:.4f}")


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

    # --- Complete modality ---
    print("\n[Complete Modality]")
    preds, labels = test_model(model, loader)
    cm = compute_metrics(preds, labels)
    print_metrics(cm, "  ")

    if args.complete_only:
        return

    # --- Fixed missing ---
    if args.missing_modality:
        print(f"\n[Missing: {args.missing_modality}]")
        p, l = test_model(model, loader, missing_modality=args.missing_modality)
        print_metrics(compute_metrics(p, l), "  ")

    # --- Random missing ---
    if args.missing_rate > 0:
        print(f"\n[Random Missing Rate: {args.missing_rate}]")
        p, l = test_model(model, loader, missing_rate=args.missing_rate)
        print_metrics(compute_metrics(p, l), "  ")

    # --- Full protocol (6 fixed + 7 random) ---
    if not args.missing_modality and args.missing_rate == 0:
        print("\n" + "=" * 60)
        print("Fixed Missing Protocol (6 configs)")
        print("=" * 60)
        fixed_cfgs = [
            ('visual', 'u={l,a} (miss v)'),
            ('acoustic', 'u={l,v} (miss a)'),
            ('text', 'u={a,v} (miss t)'),
            ('av', 'u={l}   (miss a,v)'),
            ('tv', 'u={a}   (miss t,v)'),
            ('ta', 'u={v}   (miss t,a)'),
        ]
        fixed_metrics = []
        for mm, desc in fixed_cfgs:
            p, l = test_model(model, loader, missing_modality=mm)
            m = compute_metrics(p, l)
            fixed_metrics.append(m)
            print(f"  {desc}: Acc2={m['Acc2']:.4f}  F1={m['F1']:.4f}  MAE={m['MAE']:.4f}")

        avg = {k: np.mean([m[k] for m in fixed_metrics]) for k in fixed_metrics[0]}
        print(f"\n  ** Fixed AVG: ", end="")
        print_metrics(avg)

        print("\n" + "=" * 60)
        print("Random Missing Protocol (7 rates)")
        print("=" * 60)
        rand_metrics = []
        for mr in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
            p, l = test_model(model, loader, missing_rate=mr)
            m = compute_metrics(p, l)
            rand_metrics.append(m)
            print(f"  MR={mr}: Acc2={m['Acc2']:.4f}  F1={m['F1']:.4f}  MAE={m['MAE']:.4f}")

        avg = {k: np.mean([m[k] for m in rand_metrics]) for k in rand_metrics[0]}
        print(f"\n  ** Random AVG: ", end="")
        print_metrics(avg)


if __name__ == "__main__":
    main()
