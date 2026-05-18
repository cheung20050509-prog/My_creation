"""Helpers for per-sample VTB confidence (``IBEncoder`` ``conf = sigmoid(-logvar)``)."""

from __future__ import annotations

import types
from typing import Any, Callable

import torch

from infogate_modules import masked_sequence_mean


def per_token_ib_conf_mean_over_bottleneck(
    conf: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """Token-level VTB confidence: mean over bottleneck dim ``D`` for each position.

    ``conf`` is ``[B, T, D]`` (same tensor as captured for ``ib_conf_*`` hooks).
    ``attention_mask`` is ``[B, T]`` (1 = valid token). Padded positions are set to 0.

    Returns ``[B, T]`` — one scalar per token summarizing ``D`` (same reduction as
    ``per_sample_ib_conf_mean`` uses before the temporal mean).
    """
    if conf.dim() != 3:
        raise ValueError(f"expected conf [B,T,D], got {tuple(conf.shape)}")
    m = attention_mask.float()
    per_tok = conf.mean(dim=-1)
    return per_tok * m


def per_sample_ib_conf_mean(conf: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean VTB confidence per sample: mask-mean over time, then over bottleneck dim.

    ``conf`` is ``[B, T, D]``; ``attention_mask`` is ``[B, T]`` (1 = valid).  Matches the
    spirit of ``loss_dict['conf_t'] = conf['t'].mean()`` in ``InfoGate.forward``, but
    restricted to valid tokens and reported **per row** in ``[B]``.
    """
    m = attention_mask.float()
    return masked_sequence_mean(conf, m).mean(dim=-1)


def per_sample_dpr_entropy(weights: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Shannon entropy (nats) of DPR softmax weights per row. ``weights`` is ``[B, K]``."""
    p = weights.clamp_min(eps)
    return -(p * p.log()).sum(dim=-1)


def install_infogate_ib_trace(ig: Any) -> tuple[Callable[[], None], dict[str, torch.Tensor]]:
    """Capture each forward's ``conf_{t,a,v,(h)}`` and ``conf_p`` on ``InfoGate`` ``ig``.

    Returns ``(cleanup, store)``.  During ``model(...)``, hooks fill ``store`` with
    tensors ``[B,T,D]``; keys ``t,a,v`` always, ``h`` when ``use_hcf``, ``conf_p`` after
    routing (same object used for ``fusion_conf`` in training, i.e. primary mixed conf).

    Call ``cleanup()`` once at the end to remove hooks and restore ``_route_by_primary``.
    """
    store: dict[str, torch.Tensor] = {}
    handles: list[Any] = []

    def _mk(slot: str):
        def _hook(_mod, _inp, out):
            store[slot] = out[3]

        return _hook

    handles.append(ig.ib_enc_t.register_forward_hook(_mk("t")))
    handles.append(ig.ib_enc_a.register_forward_hook(_mk("a")))
    handles.append(ig.ib_enc_v.register_forward_hook(_mk("v")))
    if getattr(ig, "use_hcf", False) and hasattr(ig, "ib_enc_h"):
        handles.append(ig.ib_enc_h.register_forward_hook(_mk("h")))

    _orig_route = ig._route_by_primary

    def _route_wrap(self, B_all_list, conf_all_list, weights, primary_onehot, primary_idx):
        out = _orig_route(B_all_list, conf_all_list, weights, primary_onehot, primary_idx)
        store["conf_p"] = out[1]
        return out

    ig._route_by_primary = types.MethodType(_route_wrap, ig)

    def _cleanup() -> None:
        for h in handles:
            h.remove()
        ig._route_by_primary = _orig_route
        store.clear()

    return _cleanup, store
