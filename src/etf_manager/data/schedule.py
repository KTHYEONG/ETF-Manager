"""Decision schedule seam: pairs each signal session with a strictly later fill session."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime
from typing import Literal

from src.etf_manager.data.calendar import DEFAULT_CALENDAR_NAME, load_calendar, next_execution_session

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
    frequency: Literal["monthly"] = "monthly",
    fill_delay_sessions: int = 1,
) -> tuple[DecisionPoint, ...]:
    """Build the signal/execution session pairs for a rebalancing frequency.

    Args:
        start: Inclusive first calendar date of the schedule.
        end: Inclusive last calendar date of the schedule.
        calendar_name: Exchange calendar code.
        frequency: Decision frequency; only monthly is supported.
        fill_delay_sessions: Sessions between signal and fill; must be at least 1.

    Returns:
        Ordered decision points whose execution session is strictly later than the signal session.

    Raises:
        ValueError: If ``fill_delay_sessions`` is below 1, which would permit same-bar execution.
    """
    if frequency != "monthly":
        raise ValueError(f"unsupported decision frequency {frequency!r}")
    calendar = load_calendar(calendar_name)
    points = tuple(
        DecisionPoint(
            signal_session=session,
            signal_at=calendar.close_ts(session),
            execution_session=next_execution_session(calendar, session, fill_delay_sessions),
        )
        for session in calendar.month_end_sessions(start, end)
    )
    logger.info("[DATA] event=schedule_built points=%d fill_delay=%d", len(points), fill_delay_sessions)
    return points
