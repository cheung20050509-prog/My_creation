"""
InfoGate + DeBERTa integration module.
Uses DeBERTa-v3-base as the text encoder; prediction is handled by InfoGate's
internal MLP head.
"""

import os

from transformers.models.deberta_v2.modeling_deberta_v2 import (
    DebertaV2PreTrainedModel, DebertaV2Model,
)
from infogate_modules import InfoGate
import global_configs
from global_configs import DEVICE

_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deberta-v3-base")


def _resolve_dims(config, mc):
    text_dim = getattr(mc, 'text_dim', None) or global_configs.TEXT_DIM
    acoustic_dim = getattr(mc, 'acoustic_dim', None) or global_configs.ACOUSTIC_DIM
    visual_dim = getattr(mc, 'visual_dim', None) or global_configs.VISUAL_DIM

    ds = getattr(mc, 'dataset', None)
    if ds and (text_dim <= 0 or acoustic_dim <= 0 or visual_dim <= 0):
        global_configs.set_dataset_config(ds)
        text_dim = getattr(mc, 'text_dim', None) or global_configs.TEXT_DIM
        acoustic_dim = getattr(mc, 'acoustic_dim', None) or global_configs.ACOUSTIC_DIM
        visual_dim = getattr(mc, 'visual_dim', None) or global_configs.VISUAL_DIM

    if text_dim <= 0:
        text_dim = config.hidden_size
    if acoustic_dim <= 0 or visual_dim <= 0:
        raise ValueError(
            "Acoustic / visual dims not configured. "
            "Call global_configs.set_dataset_config(...) before model creation."
        )
    return text_dim, acoustic_dim, visual_dim


class InfoGate_DebertaModel(DebertaV2PreTrainedModel):
    def __init__(self, config, multimodal_config):
        super().__init__(config)
        TEXT_DIM, ACOUSTIC_DIM, VISUAL_DIM = _resolve_dims(config, multimodal_config)
        self.config = config

        model = DebertaV2Model.from_pretrained(_MODEL_DIR)
        self.model = model.to(DEVICE)

        ig_args = {
            'text_dim': TEXT_DIM,
            'acoustic_dim': ACOUSTIC_DIM,
            'visual_dim': VISUAL_DIM,
            'unified_dim': getattr(multimodal_config, 'unified_dim', 256),
            'ib_hidden_dim': getattr(multimodal_config, 'ib_hidden_dim', 256),
            'bottleneck_dim': getattr(multimodal_config, 'bottleneck_dim', 128),
            'num_heads': getattr(multimodal_config, 'num_heads', 4),
            'num_infogate_layers': getattr(multimodal_config, 'num_infogate_layers', 3),
            'dropout_prob': getattr(multimodal_config, 'dropout_prob', 0.1),
            'beta_ib': getattr(multimodal_config, 'beta_ib', 32),
            'gamma_cyc': getattr(multimodal_config, 'gamma_cyc', 1.0),
            'alpha_ib': getattr(multimodal_config, 'alpha_ib', 0.01),
            'cra_layers': getattr(multimodal_config, 'cra_layers', 8),
            'cra_dims': getattr(multimodal_config, 'cra_dims', [64, 32, 16]),
        }

        self.infogate = InfoGate(ig_args)
        self.init_weights()

    def forward(self, input_ids, visual, acoustic,
                labels=None, stage=1):
        pad_id = self.config.pad_token_id if self.config.pad_token_id is not None else 0
        attention_mask = input_ids.ne(pad_id).long()

        text_features = self.model(
            input_ids=input_ids, attention_mask=attention_mask
        )[0]  # [B, T, 768]

        logits, ib_loss, loss_dict, nce_extras = self.infogate(
            text_features, acoustic, visual,
            labels=labels, stage=stage,
            attention_mask=attention_mask,
        )
        return logits, ib_loss, loss_dict, nce_extras


class InfoGate_DeBertaForSequenceClassification(DebertaV2PreTrainedModel):
    def __init__(self, config, multimodal_config):
        super().__init__(config)
        self.dberta = InfoGate_DebertaModel(config, multimodal_config)

    def forward(self, input_ids, visual, acoustic,
                labels=None, stage=1):
        return self.dberta(
            input_ids, visual, acoustic,
            labels=labels, stage=stage,
        )
