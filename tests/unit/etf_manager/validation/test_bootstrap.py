"""Unit tests for circular moving-block bootstrap."""

from __future__ import annotations

import pytest

from src.etf_manager.validation.bootstrap import moving_block_bootstrap


@pytest.mark.parametrize("scenario_id", ["VAL-V03-bootstrap-seed"])
def test_val_v03_bootstrap_seed(scenario_id: str) -> None:
    """VAL-V03-bootstrap-seed"""
    first = moving_block_bootstrap((1.0, 2.0, 3.0, 4.0), block_size=2, n_paths=3, seed=7)
    repeat = moving_block_bootstrap((1.0, 2.0, 3.0, 4.0), block_size=2, n_paths=3, seed=7)
    other_seed = moving_block_bootstrap((1.0, 2.0, 3.0, 4.0), block_size=2, n_paths=3, seed=8)

    assert first == repeat
    assert first != other_seed
    assert len(first) == 3
    assert all(len(path) == 4 for path in first)
    assert all(set(path) <= {1.0, 2.0, 3.0, 4.0} for path in first)

    with pytest.raises(ValueError, match="block_size"):
        moving_block_bootstrap((1.0, 2.0, 3.0, 4.0), block_size=0, n_paths=3, seed=7)
