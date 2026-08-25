"""Unit tests for PIT trailing compound returns."""

from __future__ import annotations

from datetime import date, datetime

import polars as pl
import pytest

from src.etf_manager.data.calendar import load_calendar
from src.etf_manager.features.momentum import trailing_compound_return

_CALENDAR = load_calendar("XNYS")


def _returns_frame(values: list[float], *, hidden_last: float | None = None) -> pl.DataFrame:
    """Consecutive-session returns; the optional extra row is published after the window."""
    all_values = [*values, *([] if hidden_last is None else [hidden_last])]
    days = _CALENDAR.sessions(date(2024, 1, 2), date(2024, 6, 28))[: len(all_values) + 1][1:]
    return pl.DataFrame(
        {
            "date": list(days),
            "ticker": ["TEST"] * len(all_values),
            "simple_return": all_values,
            "available_at": [_CALENDAR.close_ts(day) for day in days],
        },
        schema={
            "date": pl.Date,
            "ticker": pl.String,
            "simple_return": pl.Float64,
            "available_at": pl.Datetime("us", "UTC"),
        },
    )


@pytest.mark.parametrize("scenario_id", ["FEAT-J01-compound-return-pit"])
def test_feat_j01_compound_return_pit(scenario_id: str) -> None:
    """FEAT-J01-compound-return-pit"""
    frame = _returns_frame([0.10, -0.10], hidden_last=0.50)
    as_of = _CALENDAR.close_ts(frame.item(1, "date"))

    compound = trailing_compound_return(frame, as_of_ts=as_of, window=2)

    assert compound == pytest.approx(-0.01)
    with pytest.raises(ValueError, match="finite"):
        trailing_compound_return(frame, as_of_ts=as_of, window=3)


@pytest.mark.parametrize("scenario_id", ["FEAT-J01-compound-return-pit"])
def test_feat_j01_guards(scenario_id: str) -> None:
    """FEAT-J01-compound-return-pit"""
    frame = _returns_frame([0.10, -0.10])
    with pytest.raises(ValueError, match="timezone-aware"):
        trailing_compound_return(frame, as_of_ts=datetime(2024, 1, 10), window=2)
    aware = _CALENDAR.close_ts(date(2024, 1, 10))
    with pytest.raises(ValueError, match="window"):
        trailing_compound_return(frame, as_of_ts=aware, window=0)
