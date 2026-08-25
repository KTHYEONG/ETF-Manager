"""Unit tests for the monthly decision schedule."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from src.data.calendar import load_calendar
from src.data.schedule import DecisionPoint, build_decision_schedule, contribution_krw_for_point


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


@pytest.mark.parametrize("scenario_id", ["SCH-T-twice-monthly"])
def test_sch_t_twice_monthly(scenario_id: str) -> None:
    """SCH-T-twice-monthly"""
    calendar = load_calendar("XNYS")
    points = build_decision_schedule(date(2024, 1, 1), date(2024, 12, 31), frequency="twice_monthly")

    assert len(points) == 24
    by_month: dict[tuple[int, int], list[DecisionPoint]] = {}
    previous: DecisionPoint | None = None
    for point in points:
        assert point.execution_session > point.signal_session
        assert point.signal_at == calendar.close_ts(point.signal_session)
        assert calendar.is_session(point.signal_session) is True
        by_month.setdefault((point.signal_session.year, point.signal_session.month), []).append(point)
        if previous is not None:
            assert point.signal_session > previous.signal_session
        previous = point
    assert len(by_month) == 12
    for (year, month), month_points in by_month.items():
        next_year, next_month = (year + 1, 1) if month == 12 else (year, month + 1)
        last_day = date(next_year, next_month, 1) - timedelta(days=1)
        month_sessions = calendar.sessions(date(year, month, 1), last_day)
        assert [point.signal_session for point in month_points] == [
            month_sessions[0],
            month_sessions[-1],
        ]

    for point in points:
        credit = contribution_krw_for_point(
            monthly_contribution_krw=1_000_000.0, point=point, schedule=points
        )
        assert credit == pytest.approx(500_000.0)
    for month_points in by_month.values():
        total = sum(
            contribution_krw_for_point(monthly_contribution_krw=1_000_000.0, point=point, schedule=points)
            for point in month_points
        )
        assert total == pytest.approx(1_000_000.0)

    monthly = build_decision_schedule(date(2024, 1, 1), date(2024, 12, 31), frequency="monthly")
    for point in monthly:
        assert contribution_krw_for_point(
            monthly_contribution_krw=1_000_000.0, point=point, schedule=monthly
        ) == pytest.approx(1_000_000.0)

    with pytest.raises(ValueError, match="frequency"):
        build_decision_schedule(date(2024, 1, 1), date(2024, 12, 31), frequency="weekly")
    with pytest.raises(ValueError, match="fill_delay_sessions"):
        build_decision_schedule(
            date(2024, 1, 1), date(2024, 12, 31), frequency="twice_monthly", fill_delay_sessions=0
        )
    with pytest.raises(ValueError, match="monthly_contribution_krw"):
        contribution_krw_for_point(monthly_contribution_krw=0.0, point=points[0], schedule=points)
