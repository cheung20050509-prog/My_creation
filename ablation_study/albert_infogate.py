"""
InfoGate + ALBERT integration module (HKT-aligned classification path).

Uses ALBERT-base-v2 as the text encoder so that the MHD (UR-FUNNY) and MSD
(MUStARD) tasks line up with the HKT family
(https://github.com/matalvepu/HKT) and MOAC's reported setup. Prediction is
handled by InfoGate's internal MLP head. HCF is routed as a 4th modality into
InfoGate (see `infogate_modules.InfoGate` with `hcf_dim > 0`).
"""

import os

from transformers.models.albert.modeling_albert import (
    AlbertPreTrainedModel, AlbertModel,
)
from infogate_modules import InfoGate
import global_configs
from global_configs import DEVICE

_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "albert-base-v2")


def _resolve_dims(config, mc):
    text_dim = getattr(mc, 'text_dim', None) or global_configs.TEXT_DIM
    acoustic_dim = getattr(mc, 'acoustic_dim', None) or global_configs.ACOUSTIC_DIM
    visual_dim = getattr(mc, 'visual_dim', None) or global_configs.VISUAL_DIM
    hcf_dim = getattr(mc, 'hcf_dim', None)
    if hcf_dim is None:
        hcf_dim = global_configs.HCF_DIM

    ds = getattr(mc, 'dataset', None)
    if ds and (text_dim <= 0 or acoustic_dim <= 0 or visual_dim <= 0):
        global_configs.set_dataset_config(ds)
        text_dim = getattr(mc, 'text_dim', None) or global_configs.TEXT_DIM
        acoustic_dim = getattr(mc, 'acoustic_dim', None) or global_configs.ACOUSTIC_DIM
        visual_dim = getattr(mc, 'visual_dim', None) or global_configs.VISUAL_DIM
        hcf_dim = getattr(mc, 'hcf_dim', None)
        if hcf_dim is None:
            hcf_dim = global_configs.HCF_DIM

    if text_dim <= 0:
        text_dim = config.hidden_size
    if acoustic_dim <= 0 or visual_dim <= 0:
        raise ValueError(
            "Acoustic / visual dims not configured. "
            "Call global_configs.set_dataset_config(...) before model creation."
        )
    return text_dim, acoustic_dim, visual_dim, hcf_dim


class InfoGate_AlbertModel(AlbertPreTrainedModel):
    def __init__(self, config, multimodal_config):
        super().__init__(config)
        TEXT_DIM, ACOUSTIC_DIM, VISUAL_DIM, HCF_DIM = _resolve_dims(config, multimodal_config)
        self.config = config

        model = AlbertModel.from_pretrained(_MODEL_DIR)
        self.model = model.to(DEVICE)

        ig_args = {
            'text_dim': TEXT_DIM,
            'acoustic_dim': ACOUSTIC_DIM,
            'visual_dim': VISUAL_DIM,
            'hcf_dim': HCF_DIM,
            'unified_dim': getattr(multimodal_config, 'unified_dim', 256),
            'ib_hidden_dim': getattr(multimodal_config, 'ib_hidden_dim', 256),
            'bottleneck_dim': getattr(multimodal_config, 'bottleneck_dim', 128),
            'num_heads': getattr(multimodal_config, 'num_heads', 4),
            'num_infogate_layers': getattr(multimodal_config, 'num_infogate_layers', 3),
            'dropout_prob': getattr(multimodal_config, 'dropout_prob', 0.1),
            'beta_ib': getattr(multimodal_config, 'beta_ib', 32),
            'alpha_ib': getattr(multimodal_config, 'alpha_ib', 0.01),
            'use_l_lib': getattr(multimodal_config, 'use_l_lib', True),
            'use_l_tran': getattr(multimodal_config, 'use_l_tran', True),
            'use_l_rib': getattr(multimodal_config, 'use_l_rib', True),
            'selector_target_temp': getattr(multimodal_config, 'selector_target_temp', 0.35),
            'selector_balance_weight': getattr(multimodal_config, 'selector_balance_weight', 0.0),
            'selector_rib_weight': getattr(multimodal_config, 'selector_rib_weight', 0.05),
            'gumbel_tau': getattr(multimodal_config, 'gumbel_tau_start', 1.0),
            'task_type': getattr(multimodal_config, 'task_type', 'binary'),
            'ablation': getattr(multimodal_config, 'ablation', 'none'),
        }

        self.infogate = InfoGate(ig_args)
        self.init_weights()

    def forward(self, input_ids, visual, acoustic, hcf=None,
                labels=None, stage=1):
        pad_id = self.config.pad_token_id if self.config.pad_token_id is not None else 0
        attention_mask = input_ids.ne(pad_id).long()

        text_features = self.model(
            input_ids=input_ids, attention_mask=attention_mask
        )[0]  # [B, T, 768]

        logits, ib_loss, loss_dict, h_p = self.infogate(
            text_features, acoustic, visual,
            hcf=hcf,
            labels=labels, stage=stage,
            attention_mask=attention_mask,
        )
        return logits, ib_loss, loss_dict, h_p


class InfoGate_AlbertForSequenceClassification(AlbertPreTrainedModel):
    """Matching wrapper name to ``InfoGate_DeBertaForSequenceClassification``.

    The inner attribute is named ``albert`` (rather than ``dberta``) so that
    ``train_classify.py`` / ``test_classify.py`` can distinguish the two
    backbones at the param-group level. The child ``infogate`` module is
    reached via ``model.albert.infogate`` (mirrors the DeBERTa wrapper's
    ``model.dberta.infogate``).
    """

    def __init__(self, config, multimodal_config):
        super().__init__(config)
        self.albert = InfoGate_AlbertModel(config, multimodal_config)

    def forward(self, input_ids, visual, acoustic, hcf=None,
                labels=None, stage=1):
        return self.albert(
            input_ids, visual, acoustic, hcf=hcf,
            labels=labels, stage=stage,
        )
