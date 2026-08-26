"""Time-ordered walk-forward folds and rolling start cohorts."""

from __future__ import annotations

import calendar
from datetime import date, timedelta

__all__ = ["add_calendar_months", "rolling_cohorts", "walk_forward_windows"]


def add_calendar_months(day: date, months: int) -> date:
    """Shift ``day`` by ``months`` calendar months (not a session count)."""
    total = day.month - 1 + months
    year = day.year + total // 12
    month = total % 12 + 1
    last_day = calendar.monthrange(year, month)[1]
    return date(year, month, min(day.day, last_day))


def _inclusive_end(start: date, months: int) -> date:
    """Last covered day of a ``months`` span starting at ``start`` (exclusive bound minus one day)."""
    return add_calendar_months(start, months) - timedelta(days=1)


def walk_forward_windows(
    start: date,
    end: date,
    *,
    train_months: int,
    test_months: int,
    embargo_months: int = 0,
    require_full_test: bool = True,
) -> tuple[tuple[date, date, date, date], ...]:
    """Emit time-ordered (train_start, train_end, test_start, test_end) folds.

    Ends are inclusive dates and even ``embargo_months=0`` advances one full
    calendar month so train and test never share a contribution month. Folds roll
    forward by ``test_months``. With ``require_full_test=True`` (default) a fold is
    emitted only when its full ``test_months`` span fits before ``end``, so partial
    clips are omitted rather than padded; ``require_full_test=False`` restores the
    legacy clip-to-end behavior for the final partial window.

    Raises:
        ValueError: When month counts are invalid or ``start`` is after ``end``.
    """
    if start > end:
        raise ValueError(f"start {start.isoformat()} is after end {end.isoformat()}")
    if train_months < 1 or test_months < 1 or embargo_months < 0:
        raise ValueError(
            "train_months and test_months must be >= 1 and embargo_months >= 0, got "
            f"train_months={train_months}, test_months={test_months}, embargo_months={embargo_months}"
        )
    gap_months = max(embargo_months, 0) + 1
    folds: list[tuple[date, date, date, date]] = []
    offset = 0
    while True:
        train_start = add_calendar_months(start, offset)
        train_end = _inclusive_end(train_start, train_months)
        test_start = add_calendar_months(train_end, gap_months)
        if test_start > end:
            break
        full_test_end = _inclusive_end(test_start, test_months)
        if require_full_test and full_test_end > end:
            break
        folds.append((train_start, train_end, test_start, min(full_test_end, end)))
        offset += test_months
    return tuple(folds)


def rolling_cohorts(
    start: date,
    end: date,
    *,
    horizon_months: int,
    step_months: int,
) -> tuple[tuple[date, date], ...]:
    """Emit (cohort_start, cohort_end) windows with a fixed inclusive horizon.

    A cohort starting Jan 1 with a 12-month horizon covers through Dec 31 of the
    same year; cohorts whose horizon extends past ``end`` are dropped entirely.

    Raises:
        ValueError: When month counts are below 1 or ``start`` is after ``end``.
    """
    if start > end:
        raise ValueError(f"start {start.isoformat()} is after end {end.isoformat()}")
    if horizon_months < 1 or step_months < 1:
        raise ValueError(
            f"horizon_months and step_months must be >= 1, got horizon_months={horizon_months}, "
            f"step_months={step_months}"
        )
    cohorts: list[tuple[date, date]] = []
    offset = 0
    while True:
        c_start = add_calendar_months(start, offset)
        c_end = _inclusive_end(c_start, horizon_months)
        if c_end > end:
            break
        cohorts.append((c_start, c_end))
        offset += step_months
    return tuple(cohorts)
