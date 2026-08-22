"""Exchange-session calendar wrapper: real sessions, holidays, and early closes."""

from __future__ import annotations

from datetime import date, datetime
from functools import cache
from typing import Final

import exchange_calendars as xcals
import pandas as pd

DEFAULT_CALENDAR_NAME: Final[str] = "XNYS"


class TradingCalendar:
    """Session-level operations over one exchange calendar, UTC-normalized closes."""

    __slots__ = ("_cal", "_close_cache", "name")

    def __init__(self, name: str, cal: xcals.ExchangeCalendar) -> None:
        self.name = name
        self._cal = cal
        self._close_cache: dict[date, datetime] = {}

    def is_session(self, day: date) -> bool:
        """Whether ``day`` is an exchange session."""
        return bool(self._cal.is_session(pd.Timestamp(day)))

    def sessions(self, start: date, end: date) -> tuple[date, ...]:
        """All sessions in the inclusive range, ascending."""
        index = self._cal.sessions_in_range(pd.Timestamp(start), pd.Timestamp(end))
        return tuple(ts.date() for ts in index)

    def close_ts(self, session: date) -> datetime:
        """Actual session close (early closes honored), converted to UTC."""
        cached = self._close_cache.get(session)
        if cached is None:
            close = self._cal.session_close(pd.Timestamp(session))
            cached = close.tz_convert("UTC").to_pydatetime()
            self._close_cache[session] = cached
        return cached

    def next_session(self, day: date, offset: int = 1) -> date:
        """The session ``offset`` exchange sessions after ``day``; never a weekday offset."""
        if offset < 1:
            raise ValueError(f"offset must be at least 1, got {offset}")
        current = pd.Timestamp(day)
        for _ in range(offset):
            current = self._cal.next_session(current)
        resolved: date = current.date()
        return resolved

    def month_end_sessions(self, start: date, end: date) -> tuple[date, ...]:
        """Last session of each month within the inclusive range, ascending."""
        last_per_month: dict[tuple[int, int], date] = {}
        for session in self.sessions(start, end):
            last_per_month[(session.year, session.month)] = session
        return tuple(sorted(last_per_month.values()))


@cache
def load_calendar(calendar_name: str = DEFAULT_CALENDAR_NAME) -> TradingCalendar:
    """Memoized per-name wrapper around ``exchange_calendars``."""
    return TradingCalendar(calendar_name, xcals.get_calendar(calendar_name))


def next_execution_session(
    calendar: TradingCalendar,
    signal_session: date,
    fill_delay_sessions: int = 1,
) -> date:
    """Resolve the fill session strictly after the signal session.

    Counts exchange sessions only, so holidays and weekends cannot collapse the
    delay to same-bar execution.

    Raises:
        ValueError: If ``fill_delay_sessions`` is below 1.
    """
    if fill_delay_sessions < 1:
        raise ValueError(f"fill_delay_sessions must be >= 1 to ban same-bar fills, got {fill_delay_sessions}")
    execution = signal_session
    for _ in range(fill_delay_sessions):
        execution = calendar.next_session(execution)
    if execution <= signal_session:
        raise ValueError("execution session must be strictly later than the signal session")
    return execution
