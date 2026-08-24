"""Unit tests for S8 drawdown-blend diagnostics (reporting only; no adoption gate)."""

from __future__ import annotations

from datetime import date, timedelta
from math import isclose

import pytest

from src.etf_manager.analytics.blends import (
    S8_BLEND_RECIPES,
    compare_s8_blends,
)
from src.etf_manager.analytics.regimes import S8_REGIME_WINDOWS
from src.etf_manager.policy.targets import PolicyId
from src.etf_manager.sim.allocation import AllocationConfig, AllocationResult, AllocationSnapshot

_EXPECTED_IDS: tuple[str, ...] = (
    "s8_qqq",
    "qqq90_vti10",
    "qqq80_vti20",
    "qqq70_vti30",
    "qqq60_vti40",
    "qqq80_ief20",
    "qqq70_ief30",
    "s1_vti",
)


class _FakeRunner:
    """Records configs and returns a fixed number of snapshots per call."""

    def __init__(self, count: int) -> None:
        self._count = count
        self.configs: list[AllocationConfig] = []

    def __call__(self, config: AllocationConfig) -> AllocationResult:
        self.configs.append(config)
        return _result(config, self._count)


class _DivergingRunner:
    """Shortens the path whenever explicit targets_override participate."""

    def __init__(self) -> None:
        self.configs: list[AllocationConfig] = []

    def __call__(self, config: AllocationConfig) -> AllocationResult:
        self.configs.append(config)
        return _result(config, 2 if config.targets_override is not None else 3)


def _snapshot(session: date, contribution: float) -> AllocationSnapshot:
    return AllocationSnapshot(
        session=session,
        cash_krw=contribution,
        cash_usd=0.0,
        shares={},
        mark_krw=contribution,
        contribution_krw=contribution,
        fees_krw=0.0,
    )


def _result(config: AllocationConfig, count: int) -> AllocationResult:
    sessions = [date(2024, 1, 1) + timedelta(days=30 * index) for index in range(count)]
    snapshots = tuple(_snapshot(session, config.monthly_contribution_krw) for session in sessions)
    wealth = float(len(snapshots)) * config.monthly_contribution_krw
    return AllocationResult(
        config=config,
        snapshots=snapshots,
        terminal_wealth_krw=wealth,
        xirr=0.0,
        max_drawdown=0.0,
        terminal_wealth_real_krw=wealth,
        xirr_real=0.0,
    )


@pytest.mark.parametrize("scenario_id", ["BLD-O-recipes"])
def test_bld_o_recipes(scenario_id: str) -> None:
    """BLD-O-recipes"""
    assert len(S8_BLEND_RECIPES) == 8
    assert tuple(recipe_id for recipe_id, _ in S8_BLEND_RECIPES) == _EXPECTED_IDS

    by_id = dict(S8_BLEND_RECIPES)
    assert by_id["s8_qqq"] == {"QQQ": 1.0}
    assert by_id["qqq90_vti10"] == {"QQQ": 0.90, "VTI": 0.10}
    assert by_id["qqq80_vti20"] == {"QQQ": 0.80, "VTI": 0.20}
    assert by_id["qqq70_vti30"] == {"QQQ": 0.70, "VTI": 0.30}
    assert by_id["qqq60_vti40"] == {"QQQ": 0.60, "VTI": 0.40}
    assert by_id["qqq80_ief20"] == {"QQQ": 0.80, "IEF": 0.20}
    assert by_id["qqq70_ief30"] == {"QQQ": 0.70, "IEF": 0.30}
    assert by_id["s1_vti"] == {"VTI": 1.0}

    for _, weights in S8_BLEND_RECIPES:
        assert all(weight >= 0.0 for weight in weights.values())
        assert isclose(sum(weights.values()), 1.0, rel_tol=0.0, abs_tol=1e-6)


@pytest.mark.parametrize("scenario_id", ["BLD-O-equal-cashflow"])
def test_bld_o_equal_cashflow(scenario_id: str) -> None:
    """BLD-O-equal-cashflow"""
    runner = _FakeRunner(count=3)

    comparisons = compare_s8_blends(runner=runner, contribution_krw=1_000_000.0)

    assert len(comparisons) == len(S8_REGIME_WINDOWS) * len(S8_BLEND_RECIPES)
    assert {config.monthly_contribution_krw for config in runner.configs} == {1_000_000.0}

    first_window = comparisons[: len(S8_BLEND_RECIPES)]
    assert [comparison.recipe for comparison in first_window] == list(_EXPECTED_IDS)
    assert all(comparison.name == S8_REGIME_WINDOWS[0][0] for comparison in first_window)
    assert all(comparison.start == S8_REGIME_WINDOWS[0][1] for comparison in first_window)
    assert all(comparison.end == S8_REGIME_WINDOWS[0][2] for comparison in first_window)

    configs = runner.configs[: len(S8_BLEND_RECIPES)]
    s8_config = configs[_EXPECTED_IDS.index("s8_qqq")]
    assert s8_config.policy is PolicyId.S8_US_NASDAQ
    assert s8_config.targets_override is None
    s1_config = configs[_EXPECTED_IDS.index("s1_vti")]
    assert s1_config.policy is PolicyId.S1_US
    assert s1_config.targets_override is None
    mix_config = configs[_EXPECTED_IDS.index("qqq80_vti20")]
    assert mix_config.policy is PolicyId.S8_US_NASDAQ
    assert mix_config.targets_override == {"QQQ": 0.80, "VTI": 0.20}

    with pytest.raises(ValueError, match="contribution"):
        compare_s8_blends(runner=_FakeRunner(count=3), contribution_krw=0.0)

    with pytest.raises(ValueError, match="snapshot"):
        compare_s8_blends(runner=_DivergingRunner(), contribution_krw=1_000_000.0)
