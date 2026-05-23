"""Model zoo for eeg99.

Every model implements :class:`BaseModel`. See ``docs/MASTER_PLAN_99.md`` §2.
"""
from __future__ import annotations

from .base import BaseModel
from .adapter import SubjectAdaptiveFiLM, FiLMLayer
from .eegnet_plus import MultiScaleEEGNetPlus, MultiScaleEEGNetConfig
from .conformer_hier import HierarchicalCausalConformer, ConformerConfig
from .hybrid import HybridModel, HybridConfig
from .tcn_multi import MultiScaleTCN, TCNConfig
from .ssl_bendr import (
    CausalBENDREncoder, BENDRConfig,
    MaskedEEGPretrainer, CausalBENDRFinetuner,
)

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
