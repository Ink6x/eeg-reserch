"""Model zoo for eeg99.

Every model implements :class:`BaseModel`. See ``docs/MASTER_PLAN_99.md`` §2.
"""
from __future__ import annotations

from .adapter import FiLMLayer, SubjectAdaptiveFiLM
from .base import BaseModel
from .conformer_hier import ConformerConfig, HierarchicalCausalConformer
from .eegnet_plus import MultiScaleEEGNetConfig, MultiScaleEEGNetPlus
from .hybrid import HybridConfig, HybridModel
from .ssl_bendr import (
    BENDRConfig,
    CausalBENDREncoder,
    CausalBENDRFinetuner,
    MaskedEEGPretrainer,
)
from .tcn_multi import MultiScaleTCN, TCNConfig

__all__ = [
    "BaseModel",
    "SubjectAdaptiveFiLM",
    "FiLMLayer",
    "MultiScaleEEGNetPlus",
    "MultiScaleEEGNetConfig",
    "HierarchicalCausalConformer",
    "ConformerConfig",
    "HybridModel",
    "HybridConfig",
    "MultiScaleTCN",
    "TCNConfig",
    "CausalBENDREncoder",
    "BENDRConfig",
    "MaskedEEGPretrainer",
    "CausalBENDRFinetuner",
]
