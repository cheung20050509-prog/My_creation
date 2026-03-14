"""
InfoGate: Information Bottleneck-Guided Adaptive Cross-Attention
for Robust Multimodal Fusion

Addresses five limitations of PCCA (MODS, AAAI 2026):
L1. Unfiltered cross-attention         -> IB bottleneck filtering
L2. Equal-weight auxiliary fusion       -> Adaptive information gates
L3. No uncertainty awareness            -> Confidence-modulated attention
L4. No cross-modal consistency          -> Cyclic IB + translation losses
L5. No missing modality robustness      -> CRA translators
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================
# Building Blocks (adapted from CyIN / MODS)
# ============================================================

class IBEncoder(nn.Module):
    """
    Information Bottleneck Encoder.
    F_u -> (mu, logvar) -> B via reparameterization.
    Additionally returns per-token confidence: conf = sigma(-logvar).
    """
    def __init__(self, input_dim=256, hidden_dim=256, bottleneck_dim=128, dropout=0.3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, bottleneck_dim * 2),
        )
        self.bottleneck_dim = bottleneck_dim

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_dim) or (batch, input_dim)
        Returns:
            B: bottleneck latent (same shape minus last dim -> bottleneck_dim)
            mu, logvar: distribution parameters
            conf: per-element confidence in [0, 1]
        """
        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=-1)
        B = self.reparameterize(mu, logvar)
        conf = torch.sigmoid(-logvar)
        return B, mu, logvar, conf


class IBDecoder(nn.Module):
    """Information Bottleneck Decoder: B -> F_reconstructed."""
    def __init__(self, bottleneck_dim=128, hidden_dim=256, output_dim=256, dropout=0.3):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, B):
        return self.decoder(B)


class ResidualAutoencoder(nn.Module):
    """RA(x) = Dec(Enc(x)) + x  (from CyIN / MMIN)."""
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.enc = nn.Linear(in_dim, hidden_dim)
        self.dec = nn.Linear(hidden_dim, in_dim)

    def forward(self, x):
        return self.dec(F.relu(self.enc(x))) + x


class CRA(nn.Module):
    """Cascaded Residual Autoencoder for cross-modal translation in IB space."""
    def __init__(self, bottleneck_dim=128, num_layers=8, hidden_dims=None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 32, 16]
        self.layers = nn.ModuleList([
            ResidualAutoencoder(bottleneck_dim, hidden_dims[i % len(hidden_dims)])
            for i in range(num_layers)
        ])

    def forward(self, B):
        x = B
        for layer in self.layers:
            x = layer(x)
        return x


class PositionwiseFFN(nn.Module):
    def __init__(self, hidden_dim, ffn_dim=None, dropout=0.1):
        super().__init__()
        ffn_dim = ffn_dim or hidden_dim * 4
        self.fc1 = nn.Linear(hidden_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(F.gelu(self.fc1(x))))


# ============================================================
# Novel Components
# ============================================================

class IBGuidedMultiHeadAttention(nn.Module):
    """
    Multi-head attention with IB-guided confidence modulation (core novelty).

    Two modifications over standard attention:
      1. Score bias:  scores += scale * log(conf_K + eps)
         -> uncertain key positions receive lower attention weight
      2. Value gating: V_eff = V * conf_V
         -> uncertain value positions contribute less to the output

    The confidence signal conf = sigma(-logvar) comes from the IB encoder,
    which is trained to minimize I(B; X) while maximizing I(B; Y).  Tokens
    with high logvar (high uncertainty) are therefore task-irrelevant.
    """
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)
        self.W_o = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.conf_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, query, key, value, key_confidence=None):
        """
        Args:
            query:          [B, T_q, D]
            key:            [B, T_k, D]
            value:          [B, T_k, D]
            key_confidence: [B, T_k, D] from IB encoder, or None for standard attention
        """
        B = query.size(0)

        Q = self.W_q(query).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(key).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(value).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if key_confidence is not None:
            conf_pos = key_confidence.mean(dim=-1)              # [B, T_k]
            conf_bias = torch.log(conf_pos.clamp(min=1e-6))    # [B, T_k]
            scores = scores + self.conf_scale * conf_bias.unsqueeze(1).unsqueeze(2)
            V = V * conf_pos.unsqueeze(1).unsqueeze(-1)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        ctx = torch.matmul(attn, V)
        ctx = ctx.transpose(1, 2).contiguous().view(B, -1, self.hidden_dim)
        return self.W_o(ctx)


class AdaptiveInfoGate(nn.Module):
    """
    Adaptive information gate (novel).

    Learns per-sample, per-dimension gating weights for an auxiliary
    cross-attention contribution, replacing PCCA's equal-weight sum.
        g = sigma(W_2 ReLU(W_1 [primary || ca_output]))
        output = g * ca_output
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

    def forward(self, primary, ca_output):
        g = self.gate(torch.cat([primary, ca_output], dim=-1))
        return g * ca_output


# ============================================================
# InfoGate Layer & Module
# ============================================================

class InfoGateLayer(nn.Module):
    """
    Single InfoGate layer.

    Flow:
    1. Pre-LN on all streams
    2. IB-guided CA: aux1->primary, aux2->primary  (confidence-modulated)
    3. IB-guided SA: primary self-attention
    4. Adaptive gating:  g_i * CA_ai  replaces equal-weight sum
    5. Bidirectional: primary->aux1, primary->aux2
    6. FFN + skip for all three streams
    """
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        # aux -> primary cross-attention
        self.ca_a1_to_p = IBGuidedMultiHeadAttention(hidden_dim, num_heads, dropout)
        self.ca_a2_to_p = IBGuidedMultiHeadAttention(hidden_dim, num_heads, dropout)
        # primary self-attention
        self.sa_p = IBGuidedMultiHeadAttention(hidden_dim, num_heads, dropout)
        # adaptive gates
        self.gate_a1 = AdaptiveInfoGate(hidden_dim)
        self.gate_a2 = AdaptiveInfoGate(hidden_dim)
        # primary -> aux cross-attention
        self.ca_p_to_a1 = IBGuidedMultiHeadAttention(hidden_dim, num_heads, dropout)
        self.ca_p_to_a2 = IBGuidedMultiHeadAttention(hidden_dim, num_heads, dropout)

        self.ln_p1 = nn.LayerNorm(hidden_dim)
        self.ln_p2 = nn.LayerNorm(hidden_dim)
        self.ln_a1 = nn.LayerNorm(hidden_dim)
        self.ln_a1_ff = nn.LayerNorm(hidden_dim)
        self.ln_a2 = nn.LayerNorm(hidden_dim)
        self.ln_a2_ff = nn.LayerNorm(hidden_dim)

        self.ffn_p = PositionwiseFFN(hidden_dim, dropout=dropout)
        self.ffn_a1 = PositionwiseFFN(hidden_dim, dropout=dropout)
        self.ffn_a2 = PositionwiseFFN(hidden_dim, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, B_p, conf_p, B_a1, conf_a1, B_a2, conf_a2):
        B_p_n = self.ln_p1(B_p)
        B_a1_n = self.ln_a1(B_a1)
        B_a2_n = self.ln_a2(B_a2)

        # IB-guided cross-attention: auxiliaries -> primary
        ca_a1 = self.ca_a1_to_p(B_p_n, B_a1_n, B_a1_n, conf_a1)
        ca_a2 = self.ca_a2_to_p(B_p_n, B_a2_n, B_a2_n, conf_a2)

        # IB-guided self-attention on primary
        sa_p = self.sa_p(B_p_n, B_p_n, B_p_n, conf_p)
        B_p_up = B_p + self.dropout(sa_p)

        # Adaptive gating (replaces H_p + CA_a1 + CA_a2 in PCCA)
        gated_a1 = self.gate_a1(B_p_up, ca_a1)
        gated_a2 = self.gate_a2(B_p_up, ca_a2)
        B_p_fused = B_p_up + self.dropout(gated_a1) + self.dropout(gated_a2)

        # Bidirectional: primary -> auxiliaries
        B_p_fn = self.ln_p2(B_p_fused)
        ca_p_a1 = self.ca_p_to_a1(B_a1_n, B_p_fn, B_p_fn, conf_p)
        ca_p_a2 = self.ca_p_to_a2(B_a2_n, B_p_fn, B_p_fn, conf_p)

        # FFN + skip
        B_a1_out = B_a1 + self.dropout(ca_p_a1)
        B_a1_out = B_a1_out + self.ffn_a1(self.ln_a1_ff(B_a1_out))

        B_a2_out = B_a2 + self.dropout(ca_p_a2)
        B_a2_out = B_a2_out + self.ffn_a2(self.ln_a2_ff(B_a2_out))

        B_p_out = B_p_fused + self.ffn_p(self.ln_p2(B_p_fused))

        return B_p_out, B_a1_out, B_a2_out


class InfoGateModule(nn.Module):
    """Multi-layer stacked InfoGate cross-attention."""
    def __init__(self, hidden_dim, num_layers=3, num_heads=4, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            InfoGateLayer(hidden_dim, num_heads, dropout) for _ in range(num_layers)
        ])
        self.final_ln = nn.LayerNorm(hidden_dim)

    def forward(self, B_p, conf_p, B_a1, conf_a1, B_a2, conf_a2):
        for layer in self.layers:
            B_p, B_a1, B_a2 = layer(B_p, conf_p, B_a1, conf_a1, B_a2, conf_a2)
        return self.final_ln(B_p)


# ============================================================
# MSelector (from MODS)
# ============================================================

class MSelector(nn.Module):
    """Dynamic primary modality selector (from MODS, AAAI 2026)."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.W_a = nn.Linear(hidden_dim, 1)
        self.W_l = nn.Linear(hidden_dim, 1)
        self.W_v = nn.Linear(hidden_dim, 1)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 3),
        )

    def adaptive_aggregate(self, H, W_proj):
        scores = W_proj(H) / math.sqrt(self.hidden_dim)
        attn = F.softmax(scores, dim=1)
        return torch.bmm(attn.transpose(1, 2), H).squeeze(1)

    def forward(self, H_a, H_l, H_v):
        """
        Args:
            H_a, H_l, H_v: [B, T, D]
        Returns:
            weighted features, weights [w_a, w_l, w_v], primary_idx (0=a,1=l,2=v)
        """
        h_a = self.adaptive_aggregate(H_a, self.W_a)
        h_l = self.adaptive_aggregate(H_l, self.W_l)
        h_v = self.adaptive_aggregate(H_v, self.W_v)

        logits = self.mlp(torch.cat([h_a, h_l, h_v], dim=-1))
        weights = F.softmax(logits, dim=-1)
        primary_idx = torch.argmax(weights, dim=-1)

        w_a = weights[:, 0:1].unsqueeze(-1)
        w_l = weights[:, 1:2].unsqueeze(-1)
        w_v = weights[:, 2:3].unsqueeze(-1)

        return w_a * H_a, w_l * H_l, w_v * H_v, weights, primary_idx


# ============================================================
# Main InfoGate Module
# ============================================================

class InfoGate(nn.Module):
    """
    InfoGate: Information Bottleneck-Guided Adaptive Cross-Attention
    for Robust Multimodal Fusion.

    Pipeline:
        text / acoustic / visual
          -> Unimodal Projectors  (unified space)
          -> IB Encoders          (bottleneck + confidence)
          -> MSelector            (dynamic primary selection)
          -> InfoGate Module      (IB-guided cross-attention with adaptive gates)
          -> Aggregation + MLP    (sentiment prediction)

    Also computes:
        - Cyclic token-level IB loss  (L_tib)
        - Cyclic translation loss     (L_tran, stage 2 only)
        - Exports nce_extras for InfoNCE computation in the training loop
    """

    def __init__(self, args):
        super().__init__()
        self.modalities = ('t', 'a', 'v')
        self.modality_to_idx = {'t': 0, 'a': 1, 'v': 2}

        text_dim = args.get('text_dim', 768)
        acoustic_dim = args.get('acoustic_dim', 74)
        visual_dim = args.get('visual_dim', 47)
        unified_dim = args.get('unified_dim', 256)
        ib_hidden = args.get('ib_hidden_dim', 256)
        bn_dim = args.get('bottleneck_dim', 128)
        num_heads = args.get('num_heads', 4)
        num_layers = args.get('num_infogate_layers', 3)
        dropout = args.get('dropout_prob', 0.1)
        cra_layers = args.get('cra_layers', 8)
        cra_dims = args.get('cra_dims', [64, 32, 16])

        self.beta_ib = args.get('beta_ib', 32)
        self.gamma_cyc = args.get('gamma_cyc', 1.0)
        self.alpha_ib = args.get('alpha_ib', 0.01)
        self.bottleneck_dim = bn_dim

        # --- 1. Unimodal projectors ---
        self.proj_t = nn.Sequential(
            nn.Linear(text_dim, unified_dim), nn.ReLU(), nn.Dropout(dropout))
        self.proj_a = nn.Sequential(
            nn.Linear(acoustic_dim, unified_dim), nn.ReLU(), nn.Dropout(dropout))
        self.proj_v = nn.Sequential(
            nn.Linear(visual_dim, unified_dim), nn.ReLU(), nn.Dropout(dropout))

        # --- 2. IB encoders ---
        self.ib_enc_t = IBEncoder(unified_dim, ib_hidden, bn_dim, dropout)
        self.ib_enc_a = IBEncoder(unified_dim, ib_hidden, bn_dim, dropout)
        self.ib_enc_v = IBEncoder(unified_dim, ib_hidden, bn_dim, dropout)

        # --- 3. IB decoders (9 = 3 intra + 6 inter) for cyclic IB loss ---
        self.decoders = nn.ModuleDict({
            f'{s}_{t}': IBDecoder(bn_dim, ib_hidden, unified_dim, dropout)
            for s in self.modalities for t in self.modalities
        })

        # --- 4. CRA translators (3 shared bi-directional) ---
        self.translators = nn.ModuleDict({
            'a_t': CRA(bn_dim, cra_layers, cra_dims),
            't_v': CRA(bn_dim, cra_layers, cra_dims),
            'a_v': CRA(bn_dim, cra_layers, cra_dims),
        })

        # --- 5. MSelector ---
        self.mselector = MSelector(bn_dim)

        # --- 6. InfoGate cross-attention ---
        self.infogate = InfoGateModule(bn_dim, num_layers, num_heads, dropout)

        # --- 7. Aggregation ---
        self.agg_proj = nn.Linear(bn_dim, 1)

        # --- 8. InfoNCE reverse projections ---
        self.reverse_proj_a = nn.Linear(bn_dim, bn_dim)
        self.reverse_proj_l = nn.Linear(bn_dim, bn_dim)
        self.reverse_proj_v = nn.Linear(bn_dim, bn_dim)

        # --- 9. Label-level IB predictors (per-modality, for L_lib) ---
        self.label_preds = nn.ModuleDict({
            't': nn.Linear(bn_dim, 1),
            'a': nn.Linear(bn_dim, 1),
            'v': nn.Linear(bn_dim, 1),
        })

        # --- 10. ITHP-style residual fusion + prediction ---
        # expand bottleneck back to text_dim, add as residual to DeBERTa output
        self.expand = nn.Linear(bn_dim, text_dim)
        self.fuse_ln = nn.LayerNorm(text_dim)
        self.fuse_dropout = nn.Dropout(dropout)
        self.beta_shift = 1.0
        self.pooler_dense = nn.Linear(text_dim, text_dim)
        self.cls_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(text_dim, 1)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def adaptive_aggregate(self, H):
        scores = self.agg_proj(H) / math.sqrt(self.bottleneck_dim)
        attn = F.softmax(scores, dim=1)
        return torch.bmm(attn.transpose(1, 2), H).squeeze(1)

    @staticmethod
    def _masked_mean(tensor, mask=None):
        if tensor.dim() == 2:
            return tensor
        if mask is None:
            return tensor.mean(dim=1)
        m = mask.unsqueeze(-1).type_as(tensor)
        return (tensor * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)

    # ------------------------------------------------------------------
    # IB loss computation
    # ------------------------------------------------------------------

    def _compute_kl(self, mu, logvar, mask=None):
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        if mask is None or mu.dim() == 2:
            return kl.mean()
        m = mask.unsqueeze(-1).type_as(kl)
        return (kl * m).sum() / (m.sum().clamp_min(1.0) * mu.size(-1))

    def _compute_rec(self, pred, target, mask=None):
        if mask is None or pred.dim() == 2:
            return F.mse_loss(pred, target)
        diff = (pred - target).pow(2)
        m = mask.unsqueeze(-1).type_as(diff)
        return (diff * m).sum() / (m.sum().clamp_min(1.0) * pred.size(-1))

    def _token_ib(self, B_s, mu_s, lv_s, F_t, decoder, mask):
        kl = self._compute_kl(mu_s, lv_s, mask)
        rec = self._compute_rec(decoder(B_s), F_t, mask)
        return kl + self.beta_ib * rec

    def _cyclic_tib(self, F_dict, B, mu, lv, mask):
        loss = torch.tensor(0.0, device=B['t'].device)
        for m in self.modalities:
            loss = loss + self._token_ib(
                B[m], mu[m], lv[m], F_dict[m], self.decoders[f'{m}_{m}'], mask)
        for s, t in [('t', 'a'), ('t', 'v'), ('a', 'v')]:
            l_st = self._token_ib(B[s], mu[s], lv[s], F_dict[t], self.decoders[f'{s}_{t}'], mask)
            l_ts = self._token_ib(B[t], mu[t], lv[t], F_dict[s], self.decoders[f'{t}_{s}'], mask)
            loss = loss + 0.5 * (l_st + l_ts)
        return loss

    def _label_ib(self, B_pooled, mu_pooled, lv_pooled, labels):
        """Label-level IB: each modality's bottleneck should predict the label."""
        labels = labels.view(-1)
        total = labels.new_tensor(0.0)
        for m in self.modalities:
            kl = self._compute_kl(mu_pooled[m], lv_pooled[m])
            y_pred = self.label_preds[m](B_pooled[m]).squeeze(-1)
            pred_loss = F.l1_loss(y_pred, labels)
            total = total + kl + self.beta_ib * pred_loss
        return total / 3.0

    # ------------------------------------------------------------------
    # Translation loss
    # ------------------------------------------------------------------

    def _translate(self, src, tgt, bottleneck):
        key = '_'.join(sorted((src, tgt)))
        return self.translators[key](bottleneck)

    def _translation_loss(self, B_pooled):
        total = B_pooled['t'].new_tensor(0.0)
        rec_t = B_pooled['t'].new_tensor(0.0)
        cyc_t = B_pooled['t'].new_tensor(0.0)
        for s, t in [('t', 'a'), ('a', 't'), ('t', 'v'), ('v', 't'), ('a', 'v'), ('v', 'a')]:
            t_rec = self._translate(s, t, B_pooled[s])
            s_cyc = self._translate(t, s, t_rec)
            rec = F.mse_loss(t_rec, B_pooled[t])
            cyc = F.mse_loss(s_cyc, B_pooled[s])
            total = total + rec + cyc
            rec_t = rec_t + rec
            cyc_t = cyc_t + cyc
        return total, rec_t, cyc_t

    # ------------------------------------------------------------------
    # Missing-modality supplementation
    # ------------------------------------------------------------------

    def _supplement_missing(self, B_seq, B_pooled, modality_mask):
        """Replace missing modality sequences with CRA-translated pooled vectors."""
        supplemented = {}
        for tgt in self.modalities:
            t_idx = self.modality_to_idx[tgt]
            present = modality_mask[:, t_idx].view(-1, 1, 1)  # [B, 1, 1]
            if present.all():
                supplemented[tgt] = B_seq[tgt]
                continue

            rec = torch.zeros_like(B_pooled[tgt])
            n = modality_mask[:, t_idx:t_idx+1].new_zeros(modality_mask.size(0), 1)
            for src in self.modalities:
                if src == tgt:
                    continue
                s_idx = self.modality_to_idx[src]
                s_present = modality_mask[:, s_idx:s_idx+1]
                rec = rec + self._translate(src, tgt, B_pooled[src]) * s_present
                n = n + s_present
            rec = torch.where(n > 0, rec / n.clamp_min(1), B_pooled[tgt])
            rec_seq = rec.unsqueeze(1).expand_as(B_seq[tgt])
            supplemented[tgt] = present * B_seq[tgt] + (1 - present) * rec_seq
        return supplemented

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def _route_by_primary(self, B_a_w, B_l_w, B_v_w, conf, weights, primary_idx):
        """Route weighted bottlenecks and confidences by primary selection."""
        Bs = primary_idx.size(0)
        dev = primary_idx.device
        idx = torch.arange(Bs, device=dev)

        # order: acoustic=0, language=1, visual=2 (matches MSelector)
        all_B = torch.stack([B_a_w, B_l_w, B_v_w], dim=1)          # [B,3,T,D]
        all_conf = torch.stack([conf['a'], conf['t'], conf['v']], dim=1)

        B_p = all_B[idx, primary_idx]
        conf_p = all_conf[idx, primary_idx]

        mask = torch.ones(Bs, 3, device=dev, dtype=torch.bool)
        mask[idx, primary_idx] = False
        rem_w = weights.masked_select(mask).view(Bs, 2)
        rem_i = torch.arange(3, device=dev).unsqueeze(0).expand(Bs, -1)
        rem_i = rem_i.masked_select(mask).view(Bs, 2)
        order = rem_w.argsort(dim=1, descending=True)
        sorted_i = rem_i.gather(1, order)

        B_a1 = all_B[idx, sorted_i[:, 0]]
        B_a2 = all_B[idx, sorted_i[:, 1]]
        conf_a1 = all_conf[idx, sorted_i[:, 0]]
        conf_a2 = all_conf[idx, sorted_i[:, 1]]
        return B_p, conf_p, B_a1, conf_a1, B_a2, conf_a2

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, text, acoustic, visual, labels=None, stage=1,
                attention_mask=None, modality_mask=None):
        """
        Args:
            text:     [B, T, text_dim]   from DeBERTa
            acoustic: [B, T, acoustic_dim]
            visual:   [B, T, visual_dim]
            labels:   [B]  (unused here, reserved for label-level IB extension)
            stage:    1 = IB only; 2 = IB + cyclic translation
            attention_mask: [B, T] valid-token mask
            modality_mask:  [B, 3] modality presence (t / a / v)
        Returns:
            logits:     [B, 1]
            ib_loss:    scalar
            loss_dict:  dict of detailed losses
            nce_extras: dict for InfoNCE (training) or None
        """
        Bs, T = text.size(0), text.size(1)
        device = text.device

        if attention_mask is None:
            tok_mask = torch.ones(Bs, T, device=device, dtype=text.dtype)
        else:
            tok_mask = attention_mask.float()
        if modality_mask is None:
            modality_mask = torch.ones(Bs, 3, device=device, dtype=text.dtype)

        # 1. Project --------------------------------------------------
        F_t = self.proj_t(text)
        F_a = self.proj_a(acoustic)
        F_v = self.proj_v(visual)
        F_dict = {'t': F_t, 'a': F_a, 'v': F_v}

        # 2. IB encode ------------------------------------------------
        B_t, mu_t, lv_t, conf_t = self.ib_enc_t(F_t)
        B_a, mu_a, lv_a, conf_a = self.ib_enc_a(F_a)
        B_v, mu_v, lv_v, conf_v = self.ib_enc_v(F_v)

        B = {'t': B_t, 'a': B_a, 'v': B_v}
        mu = {'t': mu_t, 'a': mu_a, 'v': mu_v}
        lv = {'t': lv_t, 'a': lv_a, 'v': lv_v}
        conf = {'t': conf_t, 'a': conf_a, 'v': conf_v}

        # 3. Pool for loss computation --------------------------------
        B_pooled = {m: self._masked_mean(B[m], tok_mask) for m in self.modalities}
        mu_pooled = {m: self._masked_mean(mu[m], tok_mask) for m in self.modalities}
        lv_pooled = {m: self._masked_mean(lv[m], tok_mask) for m in self.modalities}

        # 4. Cyclic token-level IB loss --------------------------------
        L_tib = self._cyclic_tib(F_dict, B, mu, lv, tok_mask)

        # 5. Label-level IB loss (task-aware bottleneck supervision) ----
        if labels is not None:
            L_lib = self._label_ib(B_pooled, mu_pooled, lv_pooled, labels)
        else:
            L_lib = torch.tensor(0.0, device=device)

        # 6. Translation loss (stage 2 only, full modalities) ----------
        all_present = bool(torch.all(modality_mask > 0.5).item())
        if stage == 2 and all_present:
            L_tran, L_rec, L_cyc = self._translation_loss(B_pooled)
        else:
            zero = torch.tensor(0.0, device=device)
            L_tran, L_rec, L_cyc = zero, zero, zero

        # 6. Supplement missing modalities -----------------------------
        if not all_present:
            B_seq = self._supplement_missing(B, B_pooled, modality_mask)
            for m in self.modalities:
                m_idx = self.modality_to_idx[m]
                present = modality_mask[:, m_idx].view(-1, 1, 1)
                conf[m] = present * conf[m] + (1 - present) * 0.5
            B = B_seq

        # 7. MSelector (order: acoustic, language, visual) -------------
        B_a_w, B_l_w, B_v_w, weights, primary_idx = self.mselector(
            B['a'], B['t'], B['v'])

        # 8. Route by primary with confidence -------------------------
        B_p, conf_p, B_a1, conf_a1, B_a2, conf_a2 = self._route_by_primary(
            B_a_w, B_l_w, B_v_w, conf, weights, primary_idx)

        # 9. InfoGate cross-attention ---------------------------------
        B_p_enhanced = self.infogate(B_p, conf_p, B_a1, conf_a1, B_a2, conf_a2)

        # 10. ITHP-style residual fusion + prediction -------------------
        # expand bottleneck [B, T, bn_dim] -> [B, T, text_dim]
        h_expand = self.expand(B_p_enhanced)
        # residual: DeBERTa signal + InfoGate multimodal signal
        fused_seq = self.fuse_dropout(
            self.fuse_ln(self.beta_shift * h_expand + text))
        # pool [CLS] token like BertPooler
        pooled = torch.tanh(self.pooler_dense(fused_seq[:, 0, :]))
        logits = self.classifier(self.cls_dropout(pooled))

        # aggregate bottleneck for NCE (not for prediction)
        h_p = self.adaptive_aggregate(B_p_enhanced)

        # 12. Combine IB losses ----------------------------------------
        ib_loss = self.alpha_ib * (L_tib + L_lib)
        if stage == 2:
            ib_loss = ib_loss + self.gamma_cyc * L_tran

        loss_dict = {
            'L_tib': L_tib.item() if torch.is_tensor(L_tib) else L_tib,
            'L_lib': L_lib.item() if torch.is_tensor(L_lib) else L_lib,
            'L_tran': L_tran.item() if torch.is_tensor(L_tran) else L_tran,
            'L_rec': L_rec.item() if torch.is_tensor(L_rec) else L_rec,
            'L_cyc': L_cyc.item() if torch.is_tensor(L_cyc) else L_cyc,
        }

        return logits, ib_loss, loss_dict, h_p
