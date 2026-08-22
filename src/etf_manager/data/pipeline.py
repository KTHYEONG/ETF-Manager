"""Write-path seam: normalize a provider frame into a point-in-time stamped frame."""

from __future__ import annotations

import logging

import polars as pl

from src.etf_manager.data.calendar import DEFAULT_CALENDAR_NAME, TradingCalendar, load_calendar
from src.etf_manager.data.pit import stamp_availability
from src.etf_manager.data.schema import AvailabilityKind, Dataset, spec_for

logger = logging.getLogger(__name__)


def ingest(raw: pl.DataFrame, dataset: Dataset, *, calendar_name: str = DEFAULT_CALENDAR_NAME) -> pl.DataFrame:
    """Attach point-in-time availability to a normalized provider frame.

    Args:
        raw: Normalized provider frame matching the dataset spec columns.
        dataset: Dataset identity resolved through the spec registry.
        calendar_name: Exchange calendar code used for session-close availability.

    Returns:
        Frame with an ``available_at`` column in UTC microsecond precision.

    Raises:
        ValueError: If the frame misses spec columns or carries naive timestamps.
    """
    spec = spec_for(dataset)
    if spec.observation_column not in raw.columns:
        raise ValueError(f"frame for {dataset!r} is missing observation column {spec.observation_column!r}")
    calendar: TradingCalendar | None = None
    if spec.availability.kind is AvailabilityKind.SESSION_CLOSE:
        calendar = load_calendar(calendar_name)
    stamped = stamp_availability(raw, spec, calendar)
    logger.info("[DATA] event=ingest dataset=%s rows=%d", str(dataset), stamped.height)
    return stamped
