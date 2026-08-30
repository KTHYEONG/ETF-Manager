"""Unit tests for the exchange-calendar wrapper and fill-delay rule."""

from __future__ import annotations

from datetime import date

import pytest

from src.data.calendar import clamp_inclusive_session_range, load_calendar, next_execution_session


def test_cal_a08_holiday_fill_delay() -> None:
    """CAL-A08-holiday-fill-delay"""
    calendar = load_calendar("XNYS")
    assert next_execution_session(calendar, date(2024, 12, 24), 1) == date(2024, 12, 26)
    assert next_execution_session(calendar, date(2024, 7, 3), 1) == date(2024, 7, 5)
    assert calendar.is_session(date(2024, 12, 25)) is False
    # Early close means an earlier close time-of-day (13:00 ET vs 16:00 ET); absolute instants span different days.
    assert calendar.close_ts(date(2024, 12, 24)).time() < calendar.close_ts(date(2024, 12, 23)).time()


def test_cal_a09_fill_delay_lower_bound() -> None:
    """CAL-A09-fill-delay-lower-bound"""
    calendar = load_calendar("XNYS")
    for invalid_delay in (0, -1):
        with pytest.raises(ValueError, match="fill_delay_sessions"):
            next_execution_session(calendar, date(2024, 1, 31), invalid_delay)
    assert next_execution_session(calendar, date(2024, 1, 31), 2) == date(2024, 2, 2)
    assert load_calendar("XNYS") is load_calendar("XNYS")


@pytest.mark.parametrize("scenario_id", ["CAL-L-month-start"])
def test_cal_l_month_start_sessions(scenario_id: str) -> None:
    """CAL-L-month-start"""
    calendar = load_calendar("XNYS")

    starts = calendar.month_start_sessions(date(2024, 1, 1), date(2024, 12, 31))

    assert len(starts) == 12
    assert starts == tuple(sorted(starts))
    for start in starts:
        month_sessions = calendar.sessions(start.replace(day=1), date(start.year, start.month, 28))
        assert start == month_sessions[0]
    assert {(start.year, start.month) for start in starts} == {(2024, month) for month in range(1, 13)}


def test_cal_clamp_inclusive_session_range() -> None:
    """Non-session catalog dates clamp to the nearest in-range sessions."""
    calendar = load_calendar("XNYS")
    start, end = clamp_inclusive_session_range(calendar, date(2006, 8, 24), date(2026, 8, 21))
    assert start == calendar._cal.first_session.date()
    assert calendar.is_session(start)
    assert calendar.is_session(end)
    assert end <= date(2026, 8, 21)
