"""Unit tests for named strategic target-weight policies."""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Final

import polars as pl
import pytest

from src.etf_manager.data.calendar import load_calendar
from src.etf_manager.data.pipeline import ingest
from src.etf_manager.data.pit import TS_DTYPE
from src.etf_manager.data.schema import Dataset, spec_for
from src.etf_manager.policy.targets import (
    PolicyError,
    PolicyId,
    all_policy_tickers,
    policy_sleeves,
    resolve_targets,
)

_CALENDAR = load_calendar("XNYS")
_RETRIEVED_AT = datetime(2024, 4, 1, 5, 0, tzinfo=UTC)
_SIGNAL_AT = datetime(2024, 1, 31, 21, 0, tzinfo=UTC)


@pytest.mark.parametrize(
    ("scenario_id", "policy_value"),
    [
        ("POL-G03-static-sum-one", "s0_global"),
        ("POL-G03-static-sum-one", "s1_us"),
        ("POL-G03-static-sum-one", "s2_regional"),
        ("POL-G03-static-sum-one", "s3_global_bond"),
        ("POL-G03-static-sum-one", "s4_defensive"),
        ("POL-G03-static-sum-one", "s6_us_core_value"),
    ],
)
def test_pol_g03_static_sum_one(scenario_id: str, policy_value: str) -> None:
    """POL-G03-static-sum-one"""
    targets = resolve_targets(PolicyId(policy_value), pl.DataFrame(), _SIGNAL_AT)

    weights = list(targets.values())
    assert abs(sum(weights) - 1.0) <= 1e-6
    assert min(weights) >= 0.0
    if policy_value == "s2_regional":
        assert targets == {"VTI": 0.5, "VEA": 0.3, "VWO": 0.2}
    if policy_value == "s6_us_core_value":
        assert targets == {"VTI": 0.8, "VTV": 0.2}


@pytest.mark.parametrize("scenario_id", ["TGT-W1-sleeve-universe"])
def test_tgt_w1_sleeve_universe(scenario_id: str) -> None:
    """TGT-W1-sleeve-universe"""
    assert all_policy_tickers() == ("BND", "IEF", "TLT", "VEA", "VT", "VTI", "VTV", "VWO")
    union: set[str] = set()
    for member in PolicyId:
        union.update(policy_sleeves(member))
    assert all_policy_tickers() == tuple(sorted(union))


@pytest.mark.parametrize("scenario_id", ["POL-M2-s6-weights"])
def test_pol_m2_s6_weights(scenario_id: str) -> None:
    """POL-M2-s6-weights"""
    targets = resolve_targets(PolicyId.S6_US_CORE_VALUE, pl.DataFrame(), _SIGNAL_AT)

    assert targets == {"VTI": 0.8, "VTV": 0.2}
    weights = list(targets.values())
    assert min(weights) >= 0.0
    assert abs(sum(weights) - 1.0) <= 1e-6


@pytest.mark.parametrize("scenario_id", ["POL-G03-static-sum-one"])
def test_pol_g03_rejects_naive_signal_at(scenario_id: str) -> None:
    """Static policies ignore prices but must still type-check signal_at."""
    with pytest.raises(ValueError, match="timezone-aware"):
        resolve_targets(PolicyId.S2_REGIONAL, pl.DataFrame(), _SIGNAL_AT.replace(tzinfo=None))


_INVVOL_BARS: Final = 64


def _invvol_days() -> tuple[date, ...]:
    return _CALENDAR.sessions(date(2023, 10, 2), date(2024, 1, 31))[:_INVVOL_BARS]


def _invvol_panel() -> pl.DataFrame:
    """VTI vol is twice VEA/VWO vol via alternating +/- amplitude returns."""
    amplitudes = {"VEA": 0.01, "VWO": 0.01, "VTI": 0.02}
    days = _invvol_days()
    spec = spec_for(Dataset.PRICES)
    tickers: list[str] = []
    dates: list[date] = []
    closes: list[float] = []
    for ticker in sorted(amplitudes):
        price = 100.0
        for index, day in enumerate(days):
            sign = 1.0 if index % 2 == 0 else -1.0
            tickers.append(ticker)
            dates.append(day)
            closes.append(price)
            price *= 1.0 + sign * amplitudes[ticker]
    n = len(dates)
    raw = pl.DataFrame(
        {
            "ticker": tickers,
            "date": dates,
            "open": [value * 0.98 for value in closes],
            "high": [value * 1.02 for value in closes],
            "low": [value * 0.97 for value in closes],
            "close": closes,
            "volume": [10_000] * n,
            "adjusted_close": closes,
            "dividend": [0.0] * n,
            "split_factor": [1.0] * n,
            "source": ["synthetic"] * n,
            "retrieved_at": [_RETRIEVED_AT] * n,
        },
        schema=dict(spec.columns),
    )
    return ingest(raw, Dataset.PRICES)


def _delay_last_vti_bar(prices: pl.DataFrame) -> pl.DataFrame:
    last_day = _invvol_days()[-1]
    later = _CALENDAR.close_ts(_CALENDAR.next_session(last_day))
    return prices.with_columns(
        pl.when((pl.col("ticker") == "VTI") & (pl.col("date") == last_day))
        .then(pl.lit(later, dtype=TS_DTYPE))
        .otherwise(pl.col("available_at"))
        .alias("available_at")
    )


@pytest.mark.parametrize("scenario_id", ["POL-G04-invvol-pit"])
def test_pol_g04_invvol_pit(scenario_id: str) -> None:
    """POL-G04-invvol-pit"""
    signal_at = _CALENDAR.close_ts(_invvol_days()[-1])
    prices = _invvol_panel()

    targets = resolve_targets(PolicyId.S5_INVVOL, prices, signal_at)

    assert abs(sum(targets.values()) - 1.0) <= 1e-6
    assert targets["VEA"] == pytest.approx(targets["VWO"])
    assert targets["VTI"] == pytest.approx(0.5 * targets["VEA"])

    with pytest.raises(PolicyError):
        resolve_targets(PolicyId.S5_INVVOL, _delay_last_vti_bar(prices), signal_at)
