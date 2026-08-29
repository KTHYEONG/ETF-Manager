"""Point-in-time temporal core: availability stamping, vintage resolution, look-ahead guard."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Final

import polars as pl

from src.data.calendar import TradingCalendar
from src.data.schema import AvailabilityKind, DatasetSpec

logger = logging.getLogger(__name__)

AVAILABLE_AT: Final[str] = "available_at"
TS_DTYPE: Final[pl.Datetime] = pl.Datetime("us", "UTC")
_UTC = UTC


class LookAheadError(RuntimeError):
    """Raised when a frame exposes rows that were not yet available at decision time."""


def _ensure_utc(ts: datetime, argument: str) -> datetime:
    if ts.tzinfo is None:
        raise ValueError(f"{argument} must be timezone-aware, got naive datetime {ts!r}")
    return ts.astimezone(_UTC)


def _reject_naive_timestamp_columns(frame: pl.DataFrame) -> None:
    for name, dtype in frame.schema.items():
        if isinstance(dtype, pl.Datetime) and dtype.time_zone is None:
            raise ValueError(f"column {name!r} is a naive timestamp; tz-aware UTC is required")


def _observation_dates(frame: pl.DataFrame, spec: DatasetSpec) -> pl.Series:
    column = frame.get_column(spec.observation_column)
    if column.dtype == TS_DTYPE or isinstance(column.dtype, pl.Datetime):
        return column.dt.date()
    return column.cast(pl.Date)


def _stamp_session_close(frame: pl.DataFrame, spec: DatasetSpec, calendar: TradingCalendar) -> pl.DataFrame:
    dates = _observation_dates(frame, spec).alias("_obs_date")
    distinct = dates.unique().sort().to_list()
    closes = [calendar.close_ts(day) for day in distinct]
    mapping = pl.DataFrame(
        {"_obs_date": distinct, AVAILABLE_AT: closes},
        schema={"_obs_date": pl.Date, AVAILABLE_AT: TS_DTYPE},
    )
    return (
        frame.with_columns(dates)
        .join(mapping, on="_obs_date", how="left")
        .drop("_obs_date")
        .select(*frame.columns, AVAILABLE_AT)
    )


def _stamp_release_column(frame: pl.DataFrame, spec: DatasetSpec) -> pl.DataFrame:
    rule = spec.availability
    if rule.release_column is None:
        raise ValueError(f"RELEASE_COLUMN availability for dataset {spec.dataset!r} requires release_column")
    release = frame.get_column(rule.release_column)
    if isinstance(release.dtype, pl.Datetime) and release.dtype.time_zone is None:
        raise ValueError(f"release column {rule.release_column!r} is naive; tz-aware UTC is required")
    return frame.with_columns(pl.col(rule.release_column).cast(TS_DTYPE).alias(AVAILABLE_AT))


def _stamp_fixed_lag(frame: pl.DataFrame, spec: DatasetSpec) -> pl.DataFrame:
    rule = spec.availability
    if rule.lag is None or rule.lag <= timedelta(0):
        raise ValueError(f"FIXED_LAG availability for dataset {spec.dataset!r} requires lag > 0")
    observation = frame.get_column(spec.observation_column)
    if isinstance(observation.dtype, pl.Datetime):
        available = pl.col(spec.observation_column)
    else:
        available = pl.col(spec.observation_column).cast(pl.Datetime("us")).dt.replace_time_zone("UTC")
    return frame.with_columns((available + rule.lag).cast(TS_DTYPE).alias(AVAILABLE_AT))


def stamp_availability(
    frame: pl.DataFrame,
    spec: DatasetSpec,
    calendar: TradingCalendar | None = None,
) -> pl.DataFrame:
    """Attach the ``available_at`` column according to the spec's availability rule.

    Dispatches exactly on ``spec.availability.kind``; rows are never synthesized.

    Raises:
        ValueError: On kind/rule mismatches, naive timestamps, or missing columns.
    """
    _reject_naive_timestamp_columns(frame)
    if AVAILABLE_AT in frame.columns:
        raise ValueError(f"frame already carries an {AVAILABLE_AT!r} column")
    kind = spec.availability.kind
    if kind is AvailabilityKind.SESSION_CLOSE:
        if calendar is None:
            raise ValueError(f"SESSION_CLOSE availability for dataset {spec.dataset!r} requires a calendar")
        stamped = _stamp_session_close(frame, spec, calendar)
    elif kind is AvailabilityKind.RELEASE_COLUMN:
        stamped = _stamp_release_column(frame, spec)
    elif kind is AvailabilityKind.FIXED_LAG:
        stamped = _stamp_fixed_lag(frame, spec)
    else:
        raise ValueError(f"unsupported availability kind {kind!r}")
    logger.info(
        "[DATA] event=availability_stamped dataset=%s kind=%s rows=%d",
        str(spec.dataset),
        str(kind),
        stamped.height,
    )
    return stamped


def as_of(frame: pl.DataFrame, spec: DatasetSpec, as_of_ts: datetime) -> pl.DataFrame:
    """Keep only rows visible at ``as_of_ts``; revisable specs collapse to the latest vintage.

    Semantics: filter ``available_at <= as_of_ts``; when ``spec.revisable``, keep per key
    group the row with maximum ``available_at`` (ties broken by the last sorted row).

    Raises:
        ValueError: If ``as_of_ts`` is naive.
    """
    cutoff = _ensure_utc(as_of_ts, "as_of_ts")
    visible = frame.filter(pl.col(AVAILABLE_AT) <= pl.lit(cutoff, dtype=TS_DTYPE))
    if not spec.revisable or visible.is_empty():
        return visible
    # Holdings amendment consolidation: within (etf_ticker, report_date, holding_id) keep max filing_date
    dedup_keys = list(spec.key)
    try:
        from src.data.schema import Dataset as _Dataset

        if spec.dataset == _Dataset.ETF_HOLDINGS:
            dedup_keys = ["etf_ticker", "report_date", "holding_id"]
    except Exception:  # noqa: S110
        pass
    ordered = visible.sort([*dedup_keys, AVAILABLE_AT], maintain_order=True)
    return ordered.filter(pl.struct(dedup_keys).is_last_distinct())


def assert_no_lookahead(frame: pl.DataFrame, decision_ts: datetime) -> None:
    """Fail closed when any row becomes available after the decision instant.

    Never filters or repairs; a violation is an error, not a data-quality warning.

    Raises:
        LookAheadError: If any ``available_at`` exceeds ``decision_ts``.
        ValueError: On missing ``available_at`` column or a naive ``decision_ts``.
    """
    cutoff = _ensure_utc(decision_ts, "decision_ts")
    if AVAILABLE_AT not in frame.columns:
        raise ValueError(f"missing required column {AVAILABLE_AT!r}")
    violations = frame.filter(pl.col(AVAILABLE_AT) > pl.lit(cutoff, dtype=TS_DTYPE))
    if violations.height > 0:
        worst = violations.get_column(AVAILABLE_AT).max()
        msg = f"look-ahead detected: {violations.height} row(s) available at {worst!r} > decision {cutoff.isoformat()}"
        logger.error("[DATA] event=lookahead_violation rows=%d decision=%s", violations.height, cutoff.isoformat())
        raise LookAheadError(msg)
