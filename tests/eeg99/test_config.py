"""Test YAML config loader and shallow overlay merge."""
from __future__ import annotations

from pathlib import Path

import pytest

from eeg99.utils.config import load_yaml, merge_overlay

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "src" / "eeg99" / "configs" / "default.yaml"


@pytest.mark.unit
def test_default_config_loads() -> None:
    cfg = load_yaml(DEFAULT_CONFIG)
    assert cfg["data"]["sfreq"] == 500.0
    assert cfg["data"]["n_channels"] == 32
    assert cfg["data"]["n_events"] == 6


@pytest.mark.unit
def test_merge_overlay_shallow() -> None:
    base = {"a": 1, "b": {"x": 1, "y": 2}}
    overlay = {"b": {"x": 99}, "c": 3}
    merged = merge_overlay(base, overlay)
    assert merged == {"a": 1, "b": {"x": 99, "y": 2}, "c": 3}
    # original is not mutated
    assert base == {"a": 1, "b": {"x": 1, "y": 2}}


@pytest.mark.unit
def test_load_yaml_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_yaml(tmp_path / "nonexistent.yaml")
