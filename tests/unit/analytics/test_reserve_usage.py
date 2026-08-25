"""Unit tests for S8 reserve usage reconstruction (reporting only; no adoption gate)."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from src.etf_manager.analytics.regimes import S8_REGIME_WINDOWS
from src.etf_manager.analytics.reserve_usage import (
    compare_s8_reserve,
    summarize_reserve_usage,
)
from src.etf_manager.policy.reserve import ReserveConfig
from src.etf_manager.policy.targets import PolicyError, PolicyId
from src.etf_manager.sim.allocation import AllocationConfig, AllocationResult, AllocationSnapshot

_WINDOW: tuple[tuple[str, date, date], ...] = (
    ("calendar_max", date(2006, 10, 31), date(2026, 6, 30)),
)
_SKIP_WINDOWS: tuple[tuple[str, date, date], ...] = (
    _WINDOW[0],
    ("gfc_crisis", date(2007, 10, 1), date(2009, 3, 31)),
)
_RESERVED_LEDGER: tuple[float, ...] = (100_000.0, 150_000.0, 50_000.0)


def _snapshot(session: date, *, contribution_krw: float, reserve_krw: float) -> AllocationSnapshot:
    return AllocationSnapshot(
        session=session,
        cash_krw=contribution_krw,
        cash_usd=0.0,
        shares={},
        mark_krw=max(contribution_krw, 1.0),
        contribution_krw=contribution_krw,
        fees_krw=0.0,
        reserve_krw=reserve_krw,
    )


def _snapshots(reserves: list[float], contribution_krw: float = 1_000_000.0) -> tuple[AllocationSnapshot, ...]:
    base = date(2024, 1, 1)
    return tuple(
        _snapshot(
            base + timedelta(days=30 * index),
            contribution_krw=contribution_krw,
            reserve_krw=reserve,
        )
        for index, reserve in enumerate(reserves)
    )


def _result(config: AllocationConfig, count: int) -> AllocationResult:
    sessions = [date(2024, 1, 1) + timedelta(days=30 * index) for index in range(count)]
    if config.reserve is None:
        reserves = [0.0] * count
    else:
        reserves = [_RESERVED_LEDGER[index % len(_RESERVED_LEDGER)] for index in range(count)]
    snapshots = tuple(
        _snapshot(session, contribution_krw=config.monthly_contribution_krw, reserve_krw=reserve)
        for session, reserve in zip(sessions, reserves, strict=True)
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
    """Shortens the path whenever a reserve config participates."""

    def __init__(self) -> None:
        self.configs: list[AllocationConfig] = []

    def __call__(self, config: AllocationConfig) -> AllocationResult:
        self.configs.append(config)
        return _result(config, 2 if config.reserve is not None else 3)


class _WarmupBlockedRunner:
    """Fails closed with PolicyError on the reserved arm of the first window only."""

    def __init__(self) -> None:
        self.configs: list[AllocationConfig] = []

    def __call__(self, config: AllocationConfig) -> AllocationResult:
        self.configs.append(config)
        if config.reserve is not None and config.start == _SKIP_WINDOWS[0][1]:
            raise PolicyError("calendar_max requires 252 sessions of QQQ warmup")
        return _result(config, 3)


class _AlwaysBlockedRunner:
    """Fails closed with PolicyError on every arm of every window."""

    def __call__(self, config: AllocationConfig) -> AllocationResult:
        raise PolicyError(f"warmup unavailable for {config.policy!r}")


@pytest.mark.parametrize("scenario_id", ["RSV-U-identity-zero"])
def test_rsv_u_identity_zero(scenario_id: str) -> None:
    """RSV-U-identity-zero"""
    usage = summarize_reserve_usage(_snapshots([0.0, 0.0, 0.0]))

    assert usage.withheld_total == 0.0
    assert usage.redeployed_total == 0.0
    assert usage.extra_investment_ratio == 0.0
    assert usage.cash_drag_ratio == 0.0
    assert usage.reserve_idle_months == 0
    assert usage.reserve_deployment_events == 0

    with pytest.raises(ValueError, match="empty"):
        summarize_reserve_usage(())


@pytest.mark.parametrize("scenario_id", ["RSV-U-ledger-identity"])
def test_rsv_u_ledger_identity(scenario_id: str) -> None:
    """RSV-U-ledger-identity"""
    snapshots = _snapshots([100_000.0, 200_000.0, 50_000.0])
    usage = summarize_reserve_usage(snapshots)

    assert usage.withheld_total == pytest.approx(200_000.0)
    assert usage.redeployed_total == pytest.approx(150_000.0)
    reconstructed_final = usage.withheld_total - usage.redeployed_total
    assert abs(reconstructed_final - 50_000.0) <= 1e-9
    contribution_sum = sum(snapshot.contribution_krw for snapshot in snapshots)
    assert usage.extra_investment_ratio == pytest.approx(150_000.0 / contribution_sum, abs=1e-12)
    assert usage.cash_drag_ratio == pytest.approx((0.1 + 0.2 + 0.05) / 3.0)
    assert usage.reserve_idle_months == 2
    assert usage.reserve_deployment_events == 1

    with pytest.raises(ValueError, match="negative"):
        summarize_reserve_usage(_snapshots([100_000.0, -1.0]))


@pytest.mark.parametrize("scenario_id", ["RSV-U-compare-same-length"])
def test_rsv_u_compare_same_length(scenario_id: str) -> None:
    """RSV-U-compare-same-length"""
    runner = _FakeRunner(count=3)

    comparisons = compare_s8_reserve(runner=runner, contribution_krw=1_000_000.0, windows=_WINDOW)

    assert len(comparisons) == 1
    comparison = comparisons[0]
    assert comparison.name == _WINDOW[0][0]
    assert comparison.start == _WINDOW[0][1]
    assert comparison.end == _WINDOW[0][2]
    assert comparison.plain_usage.withheld_total == 0.0
    assert comparison.reserved_usage.withheld_total > 0.0
    assert comparison.reserved_usage.redeployed_total > 0.0
    assert len(runner.configs) == 2
    for config in runner.configs:
        assert config.policy is PolicyId.S8_US_NASDAQ
        assert config.monthly_contribution_krw == pytest.approx(1_000_000.0)
        assert config.start == _WINDOW[0][1]
        assert config.end == _WINDOW[0][2]
    assert runner.configs[0].reserve is None
    assert runner.configs[1].reserve is not None
    assert runner.configs[1].reserve.max_withhold == pytest.approx(0.10)

    default_comparisons = compare_s8_reserve(runner=_FakeRunner(count=2), contribution_krw=1.0)
    assert len(default_comparisons) == len(S8_REGIME_WINDOWS)

    with pytest.raises(ValueError, match="contribution"):
        compare_s8_reserve(runner=_FakeRunner(count=3), contribution_krw=0.0, windows=_WINDOW)

    with pytest.raises(ValueError, match="snapshot"):
        compare_s8_reserve(runner=_DivergingRunner(), contribution_krw=1_000_000.0, windows=_WINDOW)


@pytest.mark.parametrize("scenario_id", ["RSV-U-skip-warmup"])
def test_rsv_u_skip_warmup(scenario_id: str) -> None:
    """RSV-U-skip-warmup"""
    runner = _WarmupBlockedRunner()

    comparisons = compare_s8_reserve(runner=runner, contribution_krw=1_000_000.0, windows=_SKIP_WINDOWS)

    assert [comparison.name for comparison in comparisons] == [_SKIP_WINDOWS[1][0]]
    assert len(comparisons) == 1
    assert comparisons[0].start == _SKIP_WINDOWS[1][1]
    assert comparisons[0].end == _SKIP_WINDOWS[1][2]

    with pytest.raises(ValueError, match="usable"):
        compare_s8_reserve(runner=_AlwaysBlockedRunner(), contribution_krw=1_000_000.0, windows=_SKIP_WINDOWS)

    with pytest.raises(ValueError, match="snapshot"):
        compare_s8_reserve(runner=_DivergingRunner(), contribution_krw=1_000_000.0, windows=_SKIP_WINDOWS)

    with pytest.raises(ValueError, match="contribution"):
        compare_s8_reserve(runner=_AlwaysBlockedRunner(), contribution_krw=-1.0, windows=_SKIP_WINDOWS)



@pytest.mark.parametrize("scenario_id", ["RSV-U-compare-v2"])
def test_rsv_u_compare_v2(scenario_id: str) -> None:
    """RSV-U-compare-v2"""
    runner = _FakeRunner(count=3)
    comparisons = compare_s8_reserve(
        runner=runner,
        contribution_krw=1_000_000.0,
        windows=_WINDOW,
        reserve=ReserveConfig(max_withhold=0.10, schedule="v2"),
    )

    assert len(comparisons) == 1
    assert runner.configs[1].reserve is not None
    assert runner.configs[1].reserve.schedule == "v2"
    assert runner.configs[1].reserve.max_withhold == pytest.approx(0.10)

    default_runner = _FakeRunner(count=3)
    compare_s8_reserve(runner=default_runner, contribution_krw=1_000_000.0, windows=_WINDOW)
    assert default_runner.configs[1].reserve is not None
    assert default_runner.configs[1].reserve.schedule == "v1"
    assert default_runner.configs[1].reserve.max_withhold == pytest.approx(0.10)
