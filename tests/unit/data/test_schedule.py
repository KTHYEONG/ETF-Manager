"""Unit tests for the monthly decision schedule."""

from __future__ import annotations

from datetime import date

import pytest

from src.etf_manager.data.calendar import load_calendar
from src.etf_manager.data.schedule import DecisionPoint, build_decision_schedule


def test_sch_a10_monthly_schedule_no_same_bar() -> None:
    """SCH-A10-monthly-schedule-no-same-bar"""
    points = build_decision_schedule(date(2024, 1, 1), date(2024, 12, 31))
    assert len(points) == 12
    calendar = load_calendar("XNYS")
    previous: DecisionPoint | None = None
    month_keys = set()
    for point in points:
        assert point.execution_session > point.signal_session
        assert point.signal_at == calendar.close_ts(point.signal_session)
        assert calendar.is_session(point.signal_session) is True
        sessions_of_month = calendar.sessions(
            point.signal_session.replace(day=1),
            point.signal_session,
        )
        assert point.signal_session == sessions_of_month[-1]
        month_keys.add((point.signal_session.year, point.signal_session.month))
        if previous is not None:
            assert point.signal_session > previous.signal_session
        previous = point
    assert len(month_keys) == 12
    with pytest.raises(ValueError, match="fill_delay_sessions"):
        build_decision_schedule(date(2024, 1, 1), date(2024, 12, 31), fill_delay_sessions=0)


@pytest.mark.parametrize("scenario_id", ["SCH-L-month-open"])
def test_sch_l_month_open_schedule(scenario_id: str) -> None:
    """SCH-L-month-open"""
    calendar = load_calendar("XNYS")

    monthly = build_decision_schedule(date(2024, 1, 1), date(2024, 12, 31), frequency="monthly")
    month_open = build_decision_schedule(date(2024, 1, 1), date(2024, 12, 31), frequency="month_open")

    assert len(monthly) == len(month_open) == 12
    month_keys = set()
    previous: DecisionPoint | None = None
    for point in month_open:
        month_sessions = calendar.sessions(point.signal_session.replace(day=1), point.signal_session)
        assert point.signal_session == month_sessions[0]
        assert point.execution_session > point.signal_session
        month_keys.add((point.signal_session.year, point.signal_session.month))
        if previous is not None:
            assert point.signal_session > previous.signal_session
        previous = point
    assert len(month_keys) == 12

    with pytest.raises(ValueError, match="frequency"):
        build_decision_schedule(date(2024, 1, 1), date(2024, 12, 31), frequency="weekly")
