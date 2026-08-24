"""Unit tests for the explicit contribution reserve ledger."""

from __future__ import annotations

from datetime import date, datetime

import polars as pl
import pytest

from src.etf_manager.data.calendar import load_calendar
from src.etf_manager.data.pit import AVAILABLE_AT
from src.etf_manager.policy.reserve import ReserveConfig, apply_reserve_schedule
from src.etf_manager.policy.targets import PolicyError

_CALENDAR = load_calendar("XNYS")
_TICKER = "VTI"
_PANEL_DAYS = _CALENDAR.sessions(date(2022, 12, 1), date(2024, 3, 28))
_SIGNAL_AT = _CALENDAR.close_ts(_PANEL_DAYS[-1])
_CONTRIBUTION_KRW = 1_000_000.0


def _price_panel(closes: list[float]) -> pl.DataFrame:
    """Single-ticker PIT frame whose ``adjusted_close`` row order matches the session list."""
    return pl.DataFrame(
        {
            "ticker": [_TICKER] * len(closes),
            "date": list(_PANEL_DAYS[: len(closes)]),
            "adjusted_close": closes,
            AVAILABLE_AT: [_CALENDAR.close_ts(day) for day in _PANEL_DAYS[: len(closes)]],
        },
        schema={
            "ticker": pl.String,
            "date": pl.Date,
            "adjusted_close": pl.Float64,
            AVAILABLE_AT: pl.Datetime("us", "UTC"),
        },
    )


def _rising_closes() -> list[float]:
    """Monotonic 0.1%/session closes: positive trend, no drawdown."""
    return [100.0 * 1.001**index for index in range(len(_PANEL_DAYS))]


def _crash_closes() -> list[float]:
    """Flat closes that fall 22% below the running peak inside the final window."""
    flat = [100.0] * (len(_PANEL_DAYS) - 5)
    return [*flat, *[78.0] * 5]


@pytest.mark.parametrize("scenario_id", ["RSV-A-identity-and-bounds"])
def test_rsv_a_identity_and_bounds(scenario_id: str) -> None:
    """RSV-A-identity-and-bounds"""
    assert ReserveConfig(max_withhold=0.10).max_withhold == pytest.approx(0.10)
    with pytest.raises(ValueError, match="max_withhold"):
        ReserveConfig(max_withhold=0.0)
    with pytest.raises(ValueError, match="max_withhold"):
        ReserveConfig(max_withhold=0.11)


@pytest.mark.parametrize("scenario_id", ["RSV-A-identity-and-bounds"])
def test_rsv_a_withhold_on_rising_trend(scenario_id: str) -> None:
    """RSV-A-identity-and-bounds"""
    config = ReserveConfig(max_withhold=0.10)
    cap = 0.10 * _CONTRIBUTION_KRW

    withheld = apply_reserve_schedule(
        contribution_krw=_CONTRIBUTION_KRW,
        reserve_krw=500_000.0,
        prices=_price_panel(_rising_closes()),
        ticker=_TICKER,
        signal_at=_SIGNAL_AT,
        config=config,
    )

    assert withheld.investable_krw == pytest.approx(_CONTRIBUTION_KRW * 0.90)
    assert withheld.reserve_krw == pytest.approx(500_000.0 + cap)


@pytest.mark.parametrize("scenario_id", ["RSV-A-identity-and-bounds"])
def test_rsv_a_deploy_on_deep_drawdown(scenario_id: str) -> None:
    """RSV-A-identity-and-bounds"""
    config = ReserveConfig(max_withhold=0.10)
    panel = _price_panel(_crash_closes())
    cap = 0.10 * _CONTRIBUTION_KRW

    full_deploy = apply_reserve_schedule(
        contribution_krw=_CONTRIBUTION_KRW,
        reserve_krw=500_000.0,
        prices=panel,
        ticker=_TICKER,
        signal_at=_SIGNAL_AT,
        config=config,
    )
    assert full_deploy.investable_krw == pytest.approx(_CONTRIBUTION_KRW + min(500_000.0, cap))
    assert full_deploy.reserve_krw == pytest.approx(500_000.0 - min(500_000.0, cap))

    partial_deploy = apply_reserve_schedule(
        contribution_krw=_CONTRIBUTION_KRW,
        reserve_krw=50_000.0,
        prices=panel,
        ticker=_TICKER,
        signal_at=_SIGNAL_AT,
        config=config,
    )
    assert partial_deploy.investable_krw == pytest.approx(_CONTRIBUTION_KRW + 50_000.0)
    assert partial_deploy.reserve_krw == pytest.approx(0.0)

    # Flat closes keep the trend at exactly zero, so neither rule fires.
    passthrough = apply_reserve_schedule(
        contribution_krw=_CONTRIBUTION_KRW,
        reserve_krw=0.0,
        prices=_price_panel([100.0] * len(_PANEL_DAYS)),
        ticker=_TICKER,
        signal_at=_SIGNAL_AT,
        config=config,
    )
    assert passthrough.investable_krw == pytest.approx(_CONTRIBUTION_KRW)
    assert passthrough.reserve_krw == pytest.approx(0.0)


@pytest.mark.parametrize("scenario_id", ["RSV-A-identity-and-bounds"])
def test_rsv_a_fail_closed_guards(scenario_id: str) -> None:
    """RSV-A-identity-and-bounds"""
    config = ReserveConfig(max_withhold=0.10)
    naive_signal = datetime(2024, 3, 28, 21)
    with pytest.raises((ValueError, PolicyError)):
        apply_reserve_schedule(
            contribution_krw=_CONTRIBUTION_KRW,
            reserve_krw=0.0,
            prices=_price_panel(_rising_closes()),
            ticker=_TICKER,
            signal_at=naive_signal,
            config=config,
        )
    short_panel = _price_panel(_rising_closes()[-30:])
    with pytest.raises(PolicyError):
        apply_reserve_schedule(
            contribution_krw=_CONTRIBUTION_KRW,
            reserve_krw=0.0,
            prices=short_panel,
            ticker=_TICKER,
            signal_at=_SIGNAL_AT,
            config=config,
        )
