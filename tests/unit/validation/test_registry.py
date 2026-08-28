"""Unit tests for experiment identity hashing."""

from __future__ import annotations

from datetime import date

import pytest

from src.policy.targets import PolicyId
from src.sim.allocation import AllocationConfig
from src.validation.registry import make_experiment


@pytest.mark.parametrize("scenario_id", ["REG-MIX-identity-hash"])
def test_reg_mix_identity_hash(scenario_id: str) -> None:
    """REG-MIX-identity-hash"""
    bare = make_experiment(
        config=AllocationConfig(
            policy=PolicyId.QQQ,
            start=date(2012, 1, 3),
            end=date(2024, 12, 31),
            monthly_contribution_krw=1_000_000.0,
            targets_override={"QQQ": 1.0},
        ),
        manifest_hash="manifest",
        git_commit="deadbeef",
        seed=None,
        metrics={},
    )
    mixed = make_experiment(
        config=AllocationConfig(
            policy=PolicyId.QQQ,
            start=date(2012, 1, 3),
            end=date(2024, 12, 31),
            monthly_contribution_krw=1_000_000.0,
            targets_override={"QQQ": 0.8, "GRID": 0.2},
        ),
        manifest_hash="manifest",
        git_commit="deadbeef",
        seed=None,
        metrics={},
    )
    assert bare.config_hash != mixed.config_hash
    assert bare.experiment_id != mixed.experiment_id
