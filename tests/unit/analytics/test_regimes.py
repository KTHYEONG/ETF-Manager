"""Unit tests for QQQ regime-window diagnostics (reporting only; no adoption gate)."""

from __future__ import annotations

from datetime import date, timedelta
from typing import TYPE_CHECKING

import pytest

from src.analytics.regimes import (
    QQQ_REGIME_WINDOWS,
    compare_policy_regimes,
)
from src.policy.targets import PolicyId
from src.sim.allocation import AllocationConfig, AllocationResult, AllocationSnapshot

if TYPE_CHECKING:
    from collections.abc import Mapping


class _FakeRunner:
    """Records configs and returns one result per policy with a fixed snapshot count."""

    def __init__(self, counts: Mapping[PolicyId, int]) -> None:
        self._counts = dict(counts)
        self.configs: list[AllocationConfig] = []

    def __call__(self, config: AllocationConfig) -> AllocationResult:
        self.configs.append(config)
        return _result(config, self._counts[config.policy])


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
    real = wealth / 2.0 if config.policy is PolicyId.QQQ else wealth
    return AllocationResult(
        config=config,
        snapshots=snapshots,
        terminal_wealth_krw=wealth,
        xirr=0.0,
        max_drawdown=0.0,
        terminal_wealth_real_krw=real,
        xirr_real=0.0,
    )


@pytest.mark.parametrize("scenario_id", ["REG-N-qqq-windows"])
def test_reg_n_qqq_windows(scenario_id: str) -> None:
    """REG-N-qqq-windows"""
    assert len(QQQ_REGIME_WINDOWS) == 6

    by_name = {name: (start, end) for name, start, end in QQQ_REGIME_WINDOWS}
    assert by_name["calendar_max"] == (date(2012, 8, 31), date(2024, 8, 31))
    assert by_name["gfc_crisis"] == (date(2007, 10, 1), date(2009, 3, 31))
    assert by_name["pre_ai"] == (date(2010, 1, 4), date(2019, 12, 31))
    assert by_name["shipped_old"] == (date(2014, 1, 3), date(2024, 8, 31))
    assert by_name["bear_2022"] == (date(2022, 1, 3), date(2022, 12, 30))
    assert by_name["recent_2023_2024"] == (date(2023, 1, 3), date(2024, 8, 31))


@pytest.mark.parametrize("scenario_id", ["REG-N-equal-cashflow"])
def test_reg_n_equal_cashflow(scenario_id: str) -> None:
    """REG-N-equal-cashflow"""
    runner = _FakeRunner({PolicyId.VTI: 3, PolicyId.QQQ: 3})

    comparisons = compare_policy_regimes(runner=runner, contribution_krw=1_000_000.0)

    assert len(comparisons) == len(QQQ_REGIME_WINDOWS)
    for (name, start, end), comparison in zip(QQQ_REGIME_WINDOWS, comparisons, strict=True):
        assert comparison.name == name
        assert comparison.start == start
        assert comparison.end == end
    pairs = [runner.configs[index : index + 2] for index in range(0, len(runner.configs), 2)]
    assert len(pairs) == len(QQQ_REGIME_WINDOWS)
    for pair in pairs:
        assert [config.policy for config in pair] == [PolicyId.VTI, PolicyId.QQQ]
        contributions = {config.monthly_contribution_krw for config in pair}
        assert contributions == {1_000_000.0}
        starts = {config.start for config in pair}
        ends = {config.end for config in pair}
        assert len(starts) == 1
        assert len(ends) == 1

    with pytest.raises(ValueError, match="contribution"):
        compare_policy_regimes(runner=_FakeRunner({PolicyId.VTI: 1, PolicyId.QQQ: 1}), contribution_krw=0.0)

    diverging = _FakeRunner({PolicyId.VTI: 3, PolicyId.QQQ: 2})
    with pytest.raises(ValueError, match="snapshot"):
        compare_policy_regimes(runner=diverging, contribution_krw=1_000_000.0)
