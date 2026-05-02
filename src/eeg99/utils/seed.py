"""Reproducibility: set every RNG to a deterministic seed.

This is the only utility module that ships with a working implementation in
Phase 0 because every notebook and script needs it before anything else.
"""
from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int = 0) -> None:
    """Set seeds for ``random``, ``numpy``, ``PYTHONHASHSEED``, and torch (if available).

    Parameters
    ----------
    seed : int
        Seed value. Defaults to 0 for ablation reproducibility.

    Notes
    -----
    Determinism is not guaranteed for CUDA kernels that lack a deterministic
    implementation. For full determinism, callers should additionally export
    ``CUBLAS_WORKSPACE_CONFIG=:4096:8`` and call
    ``torch.use_deterministic_algorithms(True)``.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
