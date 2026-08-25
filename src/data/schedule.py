"""Decision schedule seam: pairs each signal session with a strictly later fill session."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime
from typing import Literal

from src.data.calendar import DEFAULT_CALENDAR_NAME, load_calendar, next_execution_session

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class DecisionPoint:
    """A signal instant bound to the session on which its orders may fill.

    Attributes:
        signal_session: Exchange session whose close produced the signal.
        signal_at: UTC close timestamp of ``signal_session``.
        execution_session: Strictly later session on which orders fill.
    """

    signal_session: date
    signal_at: datetime
    execution_session: date


def build_decision_schedule(
    start: date,
    end: date,
    *,
    calendar_name: str = DEFAULT_CALENDAR_NAME,
    frequency: Literal["monthly", "month_open", "twice_monthly"] = "monthly",
    fill_delay_sessions: int = 1,
) -> tuple[DecisionPoint, ...]:
    """Build the signal/execution session pairs for a rebalancing cadence.

    Args:
        start: Inclusive first calendar date of the schedule.
        end: Inclusive last calendar date of the schedule.
        calendar_name: Exchange calendar code.
        frequency: ``monthly`` signals on each month-end session; ``month_open``
            signals on each month's first session; both emit one point per calendar
            month. ``twice_monthly`` signals on the sorted unique union of in-range
            month-start and month-end sessions (up to two points per calendar month).
        fill_delay_sessions: Sessions between signal and fill; must be at least 1.

    Returns:
        Ordered decision points whose execution session is strictly later than the signal session.

    Raises:
        ValueError: If ``fill_delay_sessions`` is below 1, which would permit same-bar execution,
            or ``frequency`` is not a supported cadence.
    """
    if frequency not in ("monthly", "month_open", "twice_monthly"):
        raise ValueError(f"unsupported decision frequency {frequency!r}")
    calendar = load_calendar(calendar_name)
    if frequency == "monthly":
        sessions: tuple[date, ...] = calendar.month_end_sessions(start, end)
    elif frequency == "month_open":
        sessions = calendar.month_start_sessions(start, end)
    else:
        sessions = tuple(
            sorted(
                set(calendar.month_start_sessions(start, end))
                | set(calendar.month_end_sessions(start, end))
            )
        )
    points = tuple(
        DecisionPoint(
            signal_session=session,
            signal_at=calendar.close_ts(session),
            execution_session=next_execution_session(calendar, session, fill_delay_sessions),
        )
        for session in sessions
    )
    logger.info("[DATA] event=schedule_built points=%d fill_delay=%d", len(points), fill_delay_sessions)
    return points


def contribution_krw_for_point(
    *,
    monthly_contribution_krw: float,
    point: DecisionPoint,
    schedule: tuple[DecisionPoint, ...],
) -> float:
    """Split each calendar month's credit equally across that month's decision points.

    Keeps the Σ external KRW per calendar-month invariant: a twice_monthly month with
    two in-range points splits 50/50, while single-point months (``monthly`` /
    ``month_open`` schedules) receive the full credit.

    Raises:
        ValueError: When ``monthly_contribution_krw`` is not positive or the point's
            month is absent from the schedule.
    """
    if monthly_contribution_krw <= 0:
        raise ValueError(f"monthly_contribution_krw must be positive, got {monthly_contribution_krw!r}")
    month_key = (point.signal_session.year, point.signal_session.month)
    same_month_count = sum(
        1
        for other in schedule
        if (other.signal_session.year, other.signal_session.month) == month_key
    )
    if same_month_count == 0:
        raise ValueError("decision point's calendar month is absent from the schedule")
    return monthly_contribution_krw / same_month_count
