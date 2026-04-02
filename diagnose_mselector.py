"""
Diagnostic script: analyze MSelector primary modality distribution
on MOSI and MOSEI test sets using saved checkpoints.

Reports:
  - Per-sample primary_idx distribution (0=acoustic, 1=language, 2=visual)
  - Mean MSelector weights [w_a, w_l, w_v]
  - Per-sample weight statistics
"""

import os
import sys
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from collections import Counter

# Add project to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from transformers import DebertaV2Tokenizer
from deberta_infogate import InfoGate_DeBertaForSequenceClassification
import global_configs
from global_configs import DEVICE

MODEL_DIR = os.path.join(SCRIPT_DIR, "deberta-v3-base")
MAX_SEQ_LENGTH = 50


def prepare_deberta_input(tokens, visual, acoustic, tokenizer, acoustic_dim, visual_dim):
    CLS = tokenizer.cls_token
    SEP = tokenizer.sep_token
    tokens = [CLS] + tokens + [SEP]

    az = np.zeros((1, acoustic_dim))
    acoustic = np.concatenate((az, acoustic, az))
    vz = np.zeros((1, visual_dim))
    visual = np.concatenate((vz, visual, vz))

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    segment_ids = [0] * len(input_ids)
    input_mask = [1] * len(input_ids)

    pad = MAX_SEQ_LENGTH - len(input_ids)
    acoustic = np.concatenate((acoustic, np.zeros((pad, acoustic_dim))))
    visual = np.concatenate((visual, np.zeros((pad, visual_dim))))
    input_ids += [0] * pad
    input_mask += [0] * pad
    segment_ids += [0] * pad

    return input_ids, visual, acoustic, input_mask, segment_ids


def load_data(dataset_name):
    global_configs.set_dataset_config(dataset_name)
    acoustic_dim = global_configs.ACOUSTIC_DIM
    visual_dim = global_configs.VISUAL_DIM

    pkl_path = os.path.join(SCRIPT_DIR, "datasets", f"{dataset_name}.pkl")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    tokenizer = DebertaV2Tokenizer.from_pretrained(MODEL_DIR)

    features = []
    for example in data["test"]:
        (words, visual, acoustic), label_id, segment = example
        tokens, inversions = [], []
        for idx, word in enumerate(words):
            toks = tokenizer.tokenize(word)
            tokens.extend(toks)
            inversions.extend([idx] * len(toks))

        aligned_v = np.array([visual[i] for i in inversions])
        aligned_a = np.array([acoustic[i] for i in inversions])

        if len(tokens) > MAX_SEQ_LENGTH - 2:
            tokens = tokens[:MAX_SEQ_LENGTH - 2]
            aligned_a = aligned_a[:MAX_SEQ_LENGTH - 2]
            aligned_v = aligned_v[:MAX_SEQ_LENGTH - 2]

        ids, vis, aud, mask, seg = prepare_deberta_input(
            tokens, aligned_v, aligned_a, tokenizer, acoustic_dim, visual_dim)
        features.append((ids, vis, aud, label_id))

    ds = TensorDataset(
        torch.tensor(np.array([f[0] for f in features]), dtype=torch.long),
        torch.tensor(np.array([f[1] for f in features]), dtype=torch.float),
        torch.tensor(np.array([f[2] for f in features]), dtype=torch.float),
        torch.tensor(np.array([f[3] for f in features]), dtype=torch.float),
    )
    return DataLoader(ds, batch_size=64, shuffle=False)


def analyze_mselector(model, loader, dataset_name):
    """Run inference and collect MSelector statistics."""
    model.eval()
    infogate = model.dberta.infogate  # access the InfoGate module

    all_weights = []
    all_primary_idx = []

    # Hook into the forward to capture MSelector outputs
    # We'll monkey-patch the forward temporarily
    original_forward = infogate.forward

    def hooked_forward(text, acoustic, visual, labels=None, stage=1,
                       attention_mask=None):
        Bs, T = text.size(0), text.size(1)
        device = text.device

        if attention_mask is None:
            tok_mask = torch.ones(Bs, T, device=device, dtype=text.dtype)
        else:
            tok_mask = attention_mask.float()

        F_t = infogate.proj_t(text)
        F_a = infogate.proj_a(acoustic)
        F_v = infogate.proj_v(visual)

        B_t, mu_t, lv_t, conf_t = infogate.ib_enc_t(F_t)
        B_a, mu_a, lv_a, conf_a = infogate.ib_enc_a(F_a)
        B_v, mu_v, lv_v, conf_v = infogate.ib_enc_v(F_v)

        B = {'t': B_t, 'a': B_a, 'v': B_v}
        conf = {'t': conf_t, 'a': conf_a, 'v': conf_v}

        # MSelector
        B_a_w, B_l_w, B_v_w, weights, primary_idx = infogate.mselector(
            B['a'], B['t'], B['v'], tok_mask)

        # Collect data
        all_weights.append(weights.detach().cpu())
        all_primary_idx.append(primary_idx.detach().cpu())

        # Continue with original logic
        return original_forward(
            text, acoustic, visual, labels=labels, stage=stage,
            attention_mask=attention_mask,
        )

    infogate.forward = hooked_forward

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Analyzing {dataset_name}"):
            batch = tuple(t.to(DEVICE) for t in batch)
            input_ids, visual, acoustic, label_ids = batch
            visual = visual.squeeze(1)
            acoustic = acoustic.squeeze(1)

            model(input_ids, visual, acoustic, stage=2)

    # Restore
    infogate.forward = original_forward

    # Analyze
    all_weights = torch.cat(all_weights, dim=0).numpy()  # [N, 3]
    all_primary_idx = torch.cat(all_primary_idx, dim=0).numpy()  # [N]

    print(f"\n{'='*60}")
    print(f"MSelector Analysis for {dataset_name.upper()} test set")
    print(f"  Total samples: {len(all_primary_idx)}")
    print(f"{'='*60}")

    # Primary distribution
    counter = Counter(all_primary_idx.tolist())
    names = {0: 'Acoustic', 1: 'Language(Text)', 2: 'Visual'}
    print(f"\nPrimary Selection Distribution:")
    for idx in [0, 1, 2]:
        count = counter.get(idx, 0)
        pct = count / len(all_primary_idx) * 100
        bar = '█' * int(pct / 2)
        print(f"  {names[idx]:20s}: {count:5d} ({pct:5.1f}%) {bar}")

    # Weight statistics
    print(f"\nMSelector Weight Statistics [w_acoustic, w_language, w_visual]:")
    print(f"  Mean:   [{all_weights[:, 0].mean():.4f}, {all_weights[:, 1].mean():.4f}, {all_weights[:, 2].mean():.4f}]")
    print(f"  Std:    [{all_weights[:, 0].std():.4f}, {all_weights[:, 1].std():.4f}, {all_weights[:, 2].std():.4f}]")
    print(f"  Min:    [{all_weights[:, 0].min():.4f}, {all_weights[:, 1].min():.4f}, {all_weights[:, 2].min():.4f}]")
    print(f"  Max:    [{all_weights[:, 0].max():.4f}, {all_weights[:, 1].max():.4f}, {all_weights[:, 2].max():.4f}]")
    print(f"  Median: [{np.median(all_weights[:, 0]):.4f}, {np.median(all_weights[:, 1]):.4f}, {np.median(all_weights[:, 2]):.4f}]")

    # Analyze by label sentiment
    print(f"\nDominant modality (highest weight) per sample:")
    dominant = np.argmax(all_weights, axis=1)
    dom_counter = Counter(dominant.tolist())
    for idx in [0, 1, 2]:
        count = dom_counter.get(idx, 0)
        pct = count / len(dominant) * 100
        print(f"  {names[idx]:20s}: {count:5d} ({pct:5.1f}%)")

    return all_weights, all_primary_idx


def load_model_from_ckpt(ckpt_path, dataset_name):
    """Load model from checkpoint."""
    global_configs.set_dataset_config(dataset_name)

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    args = ckpt.get('args', None)

    if args is None:
        # Create dummy args
        class Args:
            pass
        args = Args()
        args.dataset = dataset_name
        args.unified_dim = 256
        args.ib_hidden_dim = 256
        args.bottleneck_dim = 128
        args.num_heads = 4
        args.num_infogate_layers = 3
        args.dropout_prob = 0.1
        args.beta_ib = 32
        args.gamma_cyc = 1.0
        args.alpha_ib = 0.01
        args.cra_layers = 8
        args.cra_dims = [64, 32, 16]

    model = InfoGate_DeBertaForSequenceClassification.from_pretrained(
        MODEL_DIR, multimodal_config=args, num_labels=1)

    # Load state dict
    state_dict = ckpt['model_state_dict']
    model.load_state_dict(state_dict, strict=False)
    model.to(DEVICE)
    model.eval()

    print(f"Loaded checkpoint: {ckpt_path}")
    print(f"  Epoch: {ckpt.get('epoch', '?')}")
    print(f"  Dev MAE: {ckpt.get('dev_mae', '?')}")

    return model


def main():
    ckpt_dir = os.path.join(SCRIPT_DIR, "checkpoints")

    # MOSI
    mosi_ckpt = os.path.join(ckpt_dir, "infogate_mosi_best.pt")
    if os.path.exists(mosi_ckpt):
        print("\n" + "="*60)
        print("Loading MOSI model...")
        model = load_model_from_ckpt(mosi_ckpt, "mosi")
        loader = load_data("mosi")
        analyze_mselector(model, loader, "mosi")
        del model
        torch.cuda.empty_cache()
    else:
        print(f"MOSI checkpoint not found: {mosi_ckpt}")

    # MOSEI
    mosei_ckpt = os.path.join(ckpt_dir, "infogate_mosei_best.pt")
    if os.path.exists(mosei_ckpt):
        print("\n" + "="*60)
        print("Loading MOSEI model...")
        model = load_model_from_ckpt(mosei_ckpt, "mosei")
        loader = load_data("mosei")
        analyze_mselector(model, loader, "mosei")
        del model
        torch.cuda.empty_cache()
    else:
        print(f"MOSEI checkpoint not found: {mosei_ckpt}")


if __name__ == "__main__":
    main()
