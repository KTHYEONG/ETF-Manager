"""Unit tests for S8 decision-cadence comparison diagnostics (reporting only; no adoption gate)."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from src.etf_manager.analytics.cadence import compare_s8_cadence
from src.etf_manager.analytics.regimes import S8_REGIME_WINDOWS
from src.etf_manager.policy.targets import PolicyError, PolicyId
from src.etf_manager.sim.allocation import AllocationConfig, AllocationResult, AllocationSnapshot

_WINDOW: tuple[tuple[str, date, date], ...] = (
    ("calendar_max", date(2006, 10, 31), date(2026, 6, 30)),
)
_SKIP_WINDOWS: tuple[tuple[str, date, date], ...] = (
    _WINDOW[0],
    ("gfc_crisis", date(2007, 10, 1), date(2009, 3, 31)),
)


def _snapshot(session: date, contribution_krw: float) -> AllocationSnapshot:
    return AllocationSnapshot(
        session=session,
        cash_krw=contribution_krw,
        cash_usd=0.0,
        shares={},
        mark_krw=max(contribution_krw, 1.0),
        contribution_krw=contribution_krw,
        fees_krw=0.0,
        reserve_krw=0.0,
    )


def _result(config: AllocationConfig, count: int) -> AllocationResult:
    snapshots = tuple(
        _snapshot(date(2024, 1, 1) + timedelta(days=30 * index), config.monthly_contribution_krw)
        for index in range(count)
    )
    wealth = float(count) * config.monthly_contribution_krw
    return AllocationResult(
        config=config,
        snapshots=snapshots,
        terminal_wealth_krw=wealth,
        xirr=0.0,
        max_drawdown=0.0,
        terminal_wealth_real_krw=wealth,
        xirr_real=0.0,
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
    """Shortens the path whenever the month-open cadence participates."""

    def __init__(self) -> None:
        self.configs: list[AllocationConfig] = []

    def __call__(self, config: AllocationConfig) -> AllocationResult:
        self.configs.append(config)
        return _result(config, 2 if config.cadence == "month_open" else 3)


class _WarmupBlockedRunner:
    """Fails closed with PolicyError on the month-open arm of the first window only."""

    def __init__(self) -> None:
        self.configs: list[AllocationConfig] = []

    def __call__(self, config: AllocationConfig) -> AllocationResult:
        self.configs.append(config)
        if config.cadence == "month_open" and config.start == _SKIP_WINDOWS[0][1]:
            raise PolicyError("calendar_max requires 252 sessions of QQQ warmup")
        return _result(config, 3)


class _AlwaysBlockedRunner:
    """Fails closed with PolicyError on every arm of every window."""

    def __call__(self, config: AllocationConfig) -> AllocationResult:
        raise PolicyError(f"warmup unavailable for {config.policy!r}")


@pytest.mark.parametrize("scenario_id", ["CAD-U-compare-s8"])
def test_cad_u_compare_s8(scenario_id: str) -> None:
    """CAD-U-compare-s8"""
    runner = _FakeRunner(count=3)

    comparisons = compare_s8_cadence(runner=runner, contribution_krw=1_000_000.0, windows=_WINDOW)

    assert len(comparisons) == 1
    comparison = comparisons[0]
    assert comparison.name == _WINDOW[0][0]
    assert comparison.start == _WINDOW[0][1]
    assert comparison.end == _WINDOW[0][2]
    assert len(runner.configs) == 2
    assert runner.configs[0].cadence == "monthly"
    assert runner.configs[1].cadence == "month_open"
    for config in runner.configs:
        assert config.policy is PolicyId.S8_US_NASDAQ
        assert config.monthly_contribution_krw == pytest.approx(1_000_000.0)
        assert config.start == _WINDOW[0][1]
        assert config.end == _WINDOW[0][2]
        assert config.overlay is None
        assert config.reserve is None
        assert config.currency is None
        assert config.mapping is None
        assert config.targets_override is None

    default_comparisons = compare_s8_cadence(runner=_FakeRunner(count=2), contribution_krw=1.0)
    assert len(default_comparisons) == len(S8_REGIME_WINDOWS)

    with pytest.raises(ValueError, match="contribution"):
        compare_s8_cadence(runner=_FakeRunner(count=3), contribution_krw=0.0, windows=_WINDOW)

    with pytest.raises(ValueError, match="snapshot"):
        compare_s8_cadence(runner=_DivergingRunner(), contribution_krw=1_000_000.0, windows=_WINDOW)


@pytest.mark.parametrize("scenario_id", ["CAD-U-compare-s8"])
def test_cad_u_compare_s8_skip_warmup(scenario_id: str) -> None:
    """CAD-U-compare-s8"""
    runner = _WarmupBlockedRunner()

    comparisons = compare_s8_cadence(runner=runner, contribution_krw=1_000_000.0, windows=_SKIP_WINDOWS)

    assert [comparison.name for comparison in comparisons] == [_SKIP_WINDOWS[1][0]]
    assert len(comparisons) == 1
    assert comparisons[0].start == _SKIP_WINDOWS[1][1]
    assert comparisons[0].end == _SKIP_WINDOWS[1][2]

    with pytest.raises(ValueError, match="usable"):
        compare_s8_cadence(runner=_AlwaysBlockedRunner(), contribution_krw=1_000_000.0, windows=_SKIP_WINDOWS)

    with pytest.raises(ValueError, match="snapshot"):
        compare_s8_cadence(runner=_DivergingRunner(), contribution_krw=1_000_000.0, windows=_SKIP_WINDOWS)
