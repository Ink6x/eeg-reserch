"""Test the only Phase-0 implemented utility: set_global_seed."""
from __future__ import annotations

import random

import numpy as np
import pytest

from eeg99.utils.seed import set_global_seed


@pytest.mark.unit
def test_seed_reproduces_random() -> None:
    set_global_seed(42)
    a = random.random()
    set_global_seed(42)
    b = random.random()
    assert a == b


@pytest.mark.unit
def test_seed_reproduces_numpy() -> None:
    set_global_seed(123)
    a = np.random.rand(8)
    set_global_seed(123)
    b = np.random.rand(8)
    np.testing.assert_array_equal(a, b)


@pytest.mark.unit
def test_seed_changes_with_seed() -> None:
    set_global_seed(1)
    a = np.random.rand(4)
    set_global_seed(2)
    b = np.random.rand(4)
    assert not np.allclose(a, b)
