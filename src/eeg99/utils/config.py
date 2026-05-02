"""YAML config loader with shallow override semantics.

Phase 0 stub. Will be expanded as Phase 1+ components introduce their own
config sections.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Read a YAML file and return its contents as a ``dict``.

    Parameters
    ----------
    path : str or Path
        Path to the YAML file.

    Returns
    -------
    dict[str, Any]
        Parsed YAML.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def merge_overlay(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Shallow-merge ``overlay`` onto ``base``; returns a new dict.

    Nested dicts are merged one level deep; deeper structures are replaced
    wholesale (intentional: avoids fragile deep-merge semantics).
    """
    merged: dict[str, Any] = dict(base)
    for key, value in overlay.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    return merged
