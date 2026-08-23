"""Unit tests for bounded KRW conversion policy."""

from __future__ import annotations

from datetime import date, datetime

import polars as pl
import pytest

from src.etf_manager.data.calendar import load_calendar
from src.etf_manager.policy.currency import (
    CurrencyConfig,
    conversion_fraction,
    economic_currency,
    trading_currency,
)
from src.etf_manager.policy.targets import PolicyError

_CALENDAR = load_calendar("XNYS")


def _fx_frame(rates: list[float]) -> pl.DataFrame:
    days = _CALENDAR.sessions(date(2024, 1, 2), date(2024, 6, 28))[: len(rates)]
    return pl.DataFrame(
        {
            "date": list(days),
            "usdkrw": rates,
            "available_at": [_CALENDAR.close_ts(day) for day in days],
        },
        schema={
            "date": pl.Date,
            "usdkrw": pl.Float64,
            "available_at": pl.Datetime("us", "UTC"),
        },
    )


@pytest.mark.parametrize("scenario_id", ["POL-K02-conversion-bounds"])
def test_pol_k02_conversion_bounds(scenario_id: str) -> None:
    """POL-K02-conversion-bounds"""
    rising = _fx_frame([1200.0, 1250.0, 1300.0, 1350.0, 1400.0])
    signal_at = _CALENDAR.close_ts(rising.item(-1, "date"))

    fraction = conversion_fraction(
        rising,
        signal_at,
        CurrencyConfig(max_defer=0.4, expensive_percentile=0.5, percentile_window=5),
    )

    # p=(4+0.5)/5=0.9 above pi=0.5: f = 1 - 0.4*(0.9-0.5)/(1-0.5).
    assert fraction == pytest.approx(0.68, rel=1e-9)

    flat = _fx_frame([1300.0] * 5)
    assert conversion_fraction(
        flat, signal_at, CurrencyConfig(max_defer=0.4, percentile_window=5)
    ) == pytest.approx(1.0)

    with pytest.raises(ValueError, match="max_defer"):
        CurrencyConfig(max_defer=0.0)
    with pytest.raises(ValueError, match="max_defer"):
        CurrencyConfig(max_defer=1.1)


@pytest.mark.parametrize("scenario_id", ["POL-K02-conversion-bounds"])
def test_pol_k02_fail_closed(scenario_id: str) -> None:
    """POL-K02-conversion-bounds"""
    rising = _fx_frame([1200.0, 1250.0, 1300.0, 1350.0, 1400.0])
    config = CurrencyConfig(max_defer=0.4, percentile_window=5)
    with pytest.raises(ValueError, match="timezone-aware"):
        conversion_fraction(rising, datetime(2024, 1, 20), config)
    # Only the first print is visible at its own close, so window=5 fails closed.
    with pytest.raises(PolicyError):
        conversion_fraction(rising, _CALENDAR.close_ts(rising.item(0, "date")), config)


@pytest.mark.parametrize("scenario_id", ["POL-K03-currency-labels"])
def test_pol_k03_currency_labels(scenario_id: str) -> None:
    """POL-K03-currency-labels"""
    assert trading_currency("VEA") == "USD"
    assert economic_currency("VEA") == "DEV"
    assert economic_currency("VT") == "MULTI"
    assert economic_currency("VTI") == "USD"
    for ticker in ("TLT", "IEF", "BND"):
        assert economic_currency(ticker) == "USD"
    assert economic_currency("UNKNOWN") == "USD"
