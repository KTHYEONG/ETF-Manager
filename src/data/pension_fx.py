"""Merged USD/KRW quote series with an interior-gap fallback source.

The primary ECOS base rate publishes on Korean business days only, so Korean
holidays that coincide with US sessions leave holes. A fallback source
(FRED DEXKOUS) fills only interior publication gaps; it never overwrites a
primary quote and never extends past the primary's first/last quote dates.
"""

from __future__ import annotations

import bisect
import logging
import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from enum import StrEnum
from typing import Any, cast

import polars as pl

logger = logging.getLogger(__name__)

__all__ = [
    "FxBasisStats",
    "FxFallbackStatus",
    "KrwFxSeries",
    "build_krw_fx_series",
]

_BPS: float = 10_000.0


class FxFallbackStatus(StrEnum):
    """Whether a fallback USD/KRW source contributed to the series."""

    APPLIED = "APPLIED"
    NOT_NEEDED = "NOT_NEEDED"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass(frozen=True, slots=True)
class FxBasisStats:
    """Same-day disagreement between primary and fallback quotes on dates both publish.

    Values are absolute relative differences in basis points of the primary quote.
    """

    overlap_count: int
    mean_abs_bps: float
    p99_abs_bps: float
    max_abs_bps: float


@dataclass(frozen=True, slots=True)
class KrwFxSeries:
    """Merged USD/KRW quote series with per-row source labels.

    ``frame`` has columns ``date`` (Date), ``usdkrw`` (Float64), ``source`` (Utf8),
    ``available_at`` (Datetime UTC), unique ascending dates, all quotes finite and
    positive. The frame is directly usable as the general after-tax engine's ``fx``
    input: primary rows keep the primary stamp and admitted fallback rows keep the
    fallback stamp.
    """

    frame: pl.DataFrame
    primary_source: str
    fallback_source: str | None
    status: FxFallbackStatus
    fallback_row_count: int
    rejected_late_count: int
    basis: FxBasisStats | None

    def fallback_session_dates(self, session_dates: Sequence[date]) -> tuple[date, ...]:
        """Return the sessions whose applicable quote (latest quote dated on or before the session) came from the fallback source.

        Sessions before the first quote count as not relying on fallback; the engine's
        staleness/missing-quote guard owns that failure.
        """
        admitted = _admitted_dates(self.frame, self.fallback_source)
        if not admitted:
            return ()
        quote_dates: list[date] = self.frame.get_column("date").to_list()
        out: list[date] = []
        for session in session_dates:
            idx = bisect.bisect_right(quote_dates, session) - 1
            if idx < 0:
                continue
            if quote_dates[idx] in admitted:
                out.append(session)
        return tuple(out)

    def provenance(self, session_dates: Sequence[date]) -> dict[str, object]:
        """Return a JSON-serializable disclosure of how the series was built.

        Keys: ``primary_source``, ``fallback_source``, ``fallback_status``,
        ``fallback_row_count``, ``fallback_first_date``, ``fallback_last_date`` (ISO or
        ``None``), ``rejected_late_count``, ``fallback_session_count``,
        ``fallback_session_share`` (0.0 when ``session_dates`` is empty),
        ``basis_overlap_count``, ``basis_mean_abs_bps``, ``basis_p99_abs_bps``,
        ``basis_max_abs_bps`` (basis values ``None`` when there is no overlap or no fallback).
        """
        admitted = sorted(_admitted_dates(self.frame, self.fallback_source)) if self.fallback_row_count else []
        sessions = list(session_dates)
        fallback_sessions = self.fallback_session_dates(sessions)
        count = len(fallback_sessions)
        share = (count / len(sessions)) if sessions else 0.0
        if self.basis is None:
            overlap, mean, p99, maximum = 0, None, None, None
        else:
            overlap, mean, p99, maximum = (
                self.basis.overlap_count,
                float(self.basis.mean_abs_bps),
                float(self.basis.p99_abs_bps),
                float(self.basis.max_abs_bps),
            )
        return {
            "primary_source": self.primary_source,
            "fallback_source": self.fallback_source,
            "fallback_status": str(self.status.value),
            "fallback_row_count": int(self.fallback_row_count),
            "fallback_first_date": admitted[0].isoformat() if admitted else None,
            "fallback_last_date": admitted[-1].isoformat() if admitted else None,
            "rejected_late_count": int(self.rejected_late_count),
            "fallback_session_count": int(count),
            "fallback_session_share": float(share),
            "basis_overlap_count": int(overlap),
            "basis_mean_abs_bps": mean,
            "basis_p99_abs_bps": p99,
            "basis_max_abs_bps": maximum,
        }


def _mode_label(values: list[str]) -> str:
    counts = Counter(values)
    top = max(counts.values())
    return sorted(label for label, n in counts.items() if n == top)[0]


def _admitted_dates(frame: pl.DataFrame, fallback_source: str | None) -> set[date]:
    if fallback_source is None:
        return set()
    sources: list[str] = frame.get_column("source").to_list()
    dates: list[date] = frame.get_column("date").to_list()
    return {day for day, src in zip(dates, sources, strict=True) if src == fallback_source}


def _require_columns(frame: pl.DataFrame, required: tuple[str, ...], label: str) -> None:
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"{label} is missing required column(s) {missing}")


def _quote_maps(
    frame: pl.DataFrame, label: str
) -> tuple[dict[date, float], dict[date, str], dict[date, datetime]]:
    dates: list[Any] = frame.get_column("date").to_list()
    quotes: list[Any] = frame.get_column("usdkrw").to_list()
    sources: list[Any] = frame.get_column("source").to_list()
    stamps: list[Any] = frame.get_column("available_at").to_list()
    prices: dict[date, float] = {}
    labels: dict[date, str] = {}
    published: dict[date, datetime] = {}
    for day, quote, source, stamp in zip(dates, quotes, sources, stamps, strict=True):
        if not isinstance(day, date) or isinstance(day, datetime):
            raise ValueError(f"{label} carries a non-date value {day!r}")
        if quote is None:
            continue
        value = float(cast("float", quote))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{label} carries a non-positive or non-finite quote on {day.isoformat()}")
        if not isinstance(source, str) or not source:
            raise ValueError(f"{label} carries a missing source on {day.isoformat()}")
        if stamp is None or not isinstance(stamp, datetime):
            raise ValueError(f"{label} carries a missing available_at on {day.isoformat()}")
        aware = stamp if stamp.tzinfo is not None else stamp.replace(tzinfo=UTC)
        assert isinstance(day, date)
        prices[day] = value
        labels[day] = source
        published[day] = aware.astimezone(UTC)
    return prices, labels, published


def _p99_linear(values: list[float]) -> float:
    ordered = sorted(values)
    count = len(ordered)
    if count == 1:
        return float(ordered[0])
    rank = 0.99 * (count - 1)
    low = math.floor(rank)
    high = math.ceil(rank)
    frac = rank - low
    return float(ordered[low] * (1.0 - frac) + ordered[high] * frac)


def build_krw_fx_series(primary: pl.DataFrame, fallback: pl.DataFrame | None) -> KrwFxSeries:
    """Merge a primary USD/KRW series with a fallback that fills only interior publication gaps.

    The primary (ECOS 매매기준율) publishes on Korean business days, so Korean holidays
    that coincide with US sessions leave holes that would otherwise force either a stale
    quote or a wider staleness limit. The fallback (FRED DEXKOUS) is admitted solely on
    dates the primary lacks and that lie strictly between the primary's first and last
    quote dates: a stale or truncated primary must remain visible to downstream
    staleness guards instead of being masked by a different source. Fallback quotes must
    have been published within their own UTC calendar day (point-in-time). The two
    sources measure the rate at different times of day, so the basis on overlapping
    dates is measured and returned for disclosure.

    Args:
        primary: Columns ``date``, ``usdkrw``, ``source``, ``available_at`` (tz-aware).
            Null quotes are treated as unpublished dates.
        fallback: ``None`` when the fallback dataset is unavailable; otherwise columns
            ``date``, ``usdkrw``, ``source``, ``available_at`` (tz-aware). Null quotes
            (vendor holiday gap rows) are treated as unpublished dates.

    Returns:
        The merged series and its disclosure fields; status is ``UNAVAILABLE`` when
        ``fallback is None``, ``NOT_NEEDED`` when no fallback row was admitted, else ``APPLIED``.
        The output ``frame`` carries ``available_at`` per row (primary rows keep the
        primary stamp; admitted fallback rows keep the fallback stamp).

    Raises:
        ValueError: A required column is missing; an ``available_at`` column is not
            tz-aware; a stamp is missing; a non-null quote is non-finite or
            non-positive; a source has duplicate dates; the primary has no quote; or the
            primary and fallback share a ``source`` label.
    """
    _require_columns(primary, ("date", "usdkrw", "source", "available_at"), "primary")
    if fallback is not None:
        _require_columns(fallback, ("date", "usdkrw", "source", "available_at"), "fallback")
    for label, frame in (("primary", primary), ("fallback", fallback)) if fallback is not None else (("primary", primary),):
        dates = frame.get_column("date").to_list()
        if len(set(dates)) != len(dates):
            raise ValueError(f"{label} carries duplicate dates")
        if not isinstance(frame.schema["available_at"], pl.Datetime) or frame.schema["available_at"].time_zone is None:
            raise ValueError(f"{label} available_at must be a tz-aware timestamp column")

    primary_prices, primary_labels, primary_stamps = _quote_maps(primary, "primary")
    if not primary_prices:
        raise ValueError("primary carries no quote")
    primary_source = _mode_label(list(primary_labels.values()))
    primary_dates = set(primary_prices)
    first = min(primary_dates)
    last = max(primary_dates)

    if fallback is None:
        ordered = sorted(primary_dates)
        frame = pl.DataFrame(
            {
                "date": ordered,
                "usdkrw": [float(primary_prices[d]) for d in ordered],
                "source": [primary_labels[d] for d in ordered],
                "available_at": [primary_stamps[d] for d in ordered],
            },
            schema={"date": pl.Date, "usdkrw": pl.Float64, "source": pl.String, "available_at": pl.Datetime("us", "UTC")},
        )
        logger.info("[DATA] event=krw_fx_series status=%s fallback_rows=%d rejected_late=%d", "UNAVAILABLE", 0, 0)
        return KrwFxSeries(
            frame=frame,
            primary_source=primary_source,
            fallback_source=None,
            status=FxFallbackStatus.UNAVAILABLE,
            fallback_row_count=0,
            rejected_late_count=0,
            basis=None,
        )

    _, fallback_labels, fallback_stamps = _quote_maps(fallback, "fallback")
    primary_label_set = set(primary_labels.values())
    fallback_label_set_all = {str(v) for v in fallback.get_column("source").drop_nulls().to_list()}
    if primary_label_set & fallback_label_set_all:
        raise ValueError(f"primary and fallback share a source label {sorted(primary_label_set & fallback_label_set_all)!r}")
    fallback_source: str | None = _mode_label(list(fallback_labels.values())) if fallback_labels else None
    if fallback_source is None and not fallback.is_empty():
        null_dropped = fallback.get_column("source").drop_nulls()
        if null_dropped.len():
            fallback_source = _mode_label([str(v) for v in null_dropped.to_list()])

    fallback_dates: list[Any] = fallback.get_column("date").to_list()
    fallback_quotes: list[Any] = fallback.get_column("usdkrw").to_list()
    on_time: dict[date, float] = {}
    rejected_late_count = 0
    for day, quote in zip(fallback_dates, fallback_quotes, strict=True):
        if quote is None:
            continue
        assert isinstance(day, date)
        assert not isinstance(day, datetime)
        stamp_utc = fallback_stamps[day]
        deadline = datetime(day.year, day.month, day.day, tzinfo=UTC) + timedelta(days=1)
        if stamp_utc >= deadline:
            rejected_late_count += 1
            continue
        on_time[day] = float(cast("float", quote))

    admitted = sorted(d for d in on_time if d not in primary_dates and first < d < last)

    overlap_values: list[float] = [
        abs(on_time[day] / primary_prices[day] - 1.0) * _BPS for day in primary_dates if day in on_time
    ]
    if overlap_values:
        mean_abs = float(sum(overlap_values) / len(overlap_values))
        basis = FxBasisStats(
            overlap_count=len(overlap_values),
            mean_abs_bps=mean_abs,
            p99_abs_bps=_p99_linear(overlap_values),
            max_abs_bps=float(max(overlap_values)),
        )
    else:
        basis = None

    merged_dates = sorted(primary_dates | set(admitted))
    merged_prices: list[float] = []
    merged_sources: list[str] = []
    merged_stamps: list[datetime] = []
    for day in merged_dates:
        if day in primary_prices:
            merged_prices.append(float(primary_prices[day]))
            merged_sources.append(primary_labels[day])
            merged_stamps.append(primary_stamps[day])
        else:
            merged_prices.append(float(on_time[day]))
            merged_sources.append(fallback_labels[day])
            merged_stamps.append(fallback_stamps[day])
    frame = pl.DataFrame(
        {"date": merged_dates, "usdkrw": merged_prices, "source": merged_sources, "available_at": merged_stamps},
        schema={"date": pl.Date, "usdkrw": pl.Float64, "source": pl.String, "available_at": pl.Datetime("us", "UTC")},
    )
    status = FxFallbackStatus.APPLIED if admitted else FxFallbackStatus.NOT_NEEDED
    logger.info(
        "[DATA] event=krw_fx_series status=%s fallback_rows=%d rejected_late=%d",
        str(status.value),
        len(admitted),
        rejected_late_count,
    )
    return KrwFxSeries(
        frame=frame,
        primary_source=primary_source,
        fallback_source=fallback_source,
        status=status,
        fallback_row_count=len(admitted),
        rejected_late_count=rejected_late_count,
        basis=basis,
    )
