"""Unit tests for PIT trailing FX percentile."""

from __future__ import annotations

from datetime import date, datetime

import polars as pl
import pytest

from src.etf_manager.data.calendar import load_calendar
from src.etf_manager.features.fx import trailing_fx_percentile

_CALENDAR = load_calendar("XNYS")


def _fx_frame(rates: list[float], *, hidden_last: float | None = None) -> pl.DataFrame:
    """Consecutive-session FX prints; the optional extra row publishes after the window."""
    all_rates = [*rates, *([] if hidden_last is None else [hidden_last])]
    days = _CALENDAR.sessions(date(2024, 1, 2), date(2024, 6, 28))[: len(all_rates)]
    return pl.DataFrame(
        {
            "date": list(days),
            "usdkrw": all_rates,
            "available_at": [_CALENDAR.close_ts(day) for day in days],
        },
        schema={
            "date": pl.Date,
            "usdkrw": pl.Float64,
            "available_at": pl.Datetime("us", "UTC"),
        },
    )


@pytest.mark.parametrize("scenario_id", ["FEAT-K01-fx-percentile-pit"])
def test_feat_k01_fx_percentile_pit(scenario_id: str) -> None:
    """FEAT-K01-fx-percentile-pit"""
    frame = _fx_frame([1200.0, 1400.0], hidden_last=1600.0)
    as_of = _CALENDAR.close_ts(frame.item(1, "date"))

    percentile = trailing_fx_percentile(frame, as_of_ts=as_of, window=2)

    # Midrank over the visible window including the last print: (1 + 0.5) / 2.
    assert percentile == pytest.approx(0.75)

    # The third print is published after as_of, so window=3 cannot be filled PIT.
    with pytest.raises(ValueError, match="finite positive"):
        trailing_fx_percentile(frame, as_of_ts=as_of, window=3)


@pytest.mark.parametrize("scenario_id", ["FEAT-K01-fx-percentile-pit"])
def test_feat_k01_guards(scenario_id: str) -> None:
    """FEAT-K01-fx-percentile-pit"""
    frame = _fx_frame([1200.0, 1400.0])
    with pytest.raises(ValueError, match="timezone-aware"):
        trailing_fx_percentile(frame, as_of_ts=datetime(2024, 1, 3), window=2)
    aware = _CALENDAR.close_ts(date(2024, 1, 3))
    with pytest.raises(ValueError, match="window"):
        trailing_fx_percentile(frame, as_of_ts=aware, window=0)
