"""Cross-cutting utilities: config loader, model registry, seed, logging."""
from __future__ import annotations

from .seed import set_global_seed

__all__ = ["set_global_seed"]
