"""Modern-tier pre-inception splice over declared research proxy blends."""

from __future__ import annotations

import calendar as _calendar
import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Final

import polars as pl

from src.sim.pension_monthly import MonthlyReturnPanel, panel_from_prices

logger = logging.getLogger(__name__)

__all__ = [
    "SpliceRecord",
    "SpliceRule",
    "splice_modern_panel",
]

_WEIGHT_SUM_TOLERANCE: Final[float] = 1e-9
_MONTHS_PER_YEAR: Final[int] = 12


@dataclass(frozen=True, slots=True)
class SpliceRule:
    """Pre-inception proxy for one modern-tier sleeve.

    Months strictly before ``etf_first_month`` take the weighted blend of research series in
    ``proxy_weights``; months from ``etf_first_month`` on take the ETF's own month-end total
    return. ``etf_first_month`` is the first month whose prior month-end close exists, so no
    partial listing month is ever used.

    Attributes:
        sleeve: ETF ticker used as the modern sleeve id (e.g. ``SCHD``).
        proxy_weights: Research series id to weight; a simplex.
        etf_first_month: Month-end of the first ETF-sourced return.

    Raises:
        ValueError: If weights are not a finite non-negative simplex within 1e-9, the mapping
            is empty, the sleeve id is blank, or ``etf_first_month`` is not a month-end.
    """

    sleeve: str
    proxy_weights: Mapping[str, float]
    etf_first_month: date

    def __post_init__(self) -> None:
        sleeve = self.sleeve
        if not isinstance(sleeve, str) or not sleeve.strip():
            raise ValueError(f"pension splice sleeve must be a non-blank string, got {sleeve!r}")
        weights = self.proxy_weights
        if not isinstance(weights, Mapping) or not weights:
            raise ValueError("pension splice proxy_weights must be a non-empty mapping")
        cleaned: dict[str, float] = {}
        total = 0.0
        for series_id, weight in weights.items():
            if not isinstance(series_id, str) or not series_id.strip():
                raise ValueError(f"pension splice proxy_weights carries an invalid series id {series_id!r}")
            if isinstance(weight, bool) or not isinstance(weight, float | int) or not math.isfinite(float(weight)):
                raise ValueError(f"pension splice weight for {series_id!r} must be finite")
            value = float(weight)
            if value < 0.0:
                raise ValueError(f"pension splice weight for {series_id!r} must be non-negative")
            total += value
            cleaned[series_id] = value
        if abs(total - 1.0) > _WEIGHT_SUM_TOLERANCE:
            raise ValueError(f"pension splice proxy_weights must sum to 1, got {total!r}")
        first_month = self.etf_first_month
        if not isinstance(first_month, date) or isinstance(first_month, datetime):
            raise ValueError(f"pension splice etf_first_month must be a date, got {first_month!r}")
        if first_month.day != _calendar.monthrange(first_month.year, first_month.month)[1]:
            raise ValueError(f"pension splice etf_first_month {first_month.isoformat()} is not a month-end")
        object.__setattr__(self, "sleeve", sleeve.strip())
        object.__setattr__(self, "proxy_weights", cleaned)


@dataclass(frozen=True, slots=True)
class SpliceRecord:
    """Audit trail of one applied splice: proxy months and the ETF-sourced remainder."""

    sleeve: str
    proxy_first_month: date
    proxy_last_month: date
    etf_first_month: date
    proxy_weights: Mapping[str, float]


def splice_modern_panel(
    prices: pl.DataFrame,
    research: pl.DataFrame,
    sleeves: Sequence[str],
    splices: Mapping[str, SpliceRule],
    as_of: datetime,
    start: date,
    end: date,
) -> tuple[MonthlyReturnPanel, tuple[SpliceRecord, ...]]:
    """Build the modern tier where late-listed sleeves are prefixed by a research proxy.

    Unspliced sleeves are built exactly as ``panel_from_prices`` over ``[start, end]``.
    Spliced sleeves take ETF returns over ``[etf_first_month, end]`` from
    ``panel_from_prices`` and blended research returns for the earlier months. Returns stay in
    USD, preserving the paired-ratio FX invariance of the modern tier.

    Args:
        prices: PRICES rows (snapshot-pinned by the caller).
        research: RESEARCH_MONTHLY rows (snapshot-pinned by the caller).
        sleeves: All modern sleeve ids required by candidates and controls.
        splices: Sleeve id to rule; keys must be a subset of ``sleeves``.
        as_of: Timezone-aware decision instant; later availability is invisible.
        start: First modern month-end.
        end: Last modern month-end.

    Returns:
        The modern panel (tier ``modern``) and one record per applied splice, sorted by sleeve.

    Raises:
        ValueError: If a splice key is not a requested sleeve, ``etf_first_month`` is not
            inside ``(start, end]``, any proxy series lacks a visible row for a pre-inception
            month, a proxy series has duplicate month-ends, or ``panel_from_prices`` fails for
            any sleeve segment.
    """
    cutoff = _require_as_of(as_of)
    names = tuple(sleeves)
    if not names or len(set(names)) != len(names):
        raise ValueError("pension sleeves must be non-empty and unique")
    known = set(names)
    for key in splices:
        if key not in known:
            raise ValueError(f"pension splice key {key!r} is not a requested sleeve")
    if not splices:
        return panel_from_prices(prices, names, cutoff, start, end), ()
    grid = _targets_in_range(start, end)
    for sleeve, rule in splices.items():
        if sleeve != rule.sleeve:
            raise ValueError(f"pension splice key {sleeve!r} differs from rule sleeve {rule.sleeve!r}")
        if not (start < rule.etf_first_month <= end):
            raise ValueError(
                f"pension splice etf_first_month {rule.etf_first_month.isoformat()} for {sleeve!r}"
                f" must lie in ({start.isoformat()}, {end.isoformat()}]"
            )
    cells = _visible_proxy_cells(research, cutoff, {series for rule in splices.values() for series in rule.proxy_weights})
    aligned: dict[str, tuple[float, ...]] = {}
    plain = [sleeve for sleeve in names if sleeve not in splices]
    if plain:
        aligned.update(dict(panel_from_prices(prices, plain, cutoff, start, end).returns))
    records: list[SpliceRecord] = []
    for sleeve in sorted(splices):
        rule = splices[sleeve]
        proxy_months = [month for month in grid if month < rule.etf_first_month]
        if not proxy_months:
            raise ValueError(
                f"pension splice for {sleeve!r} covers no proxy month before {rule.etf_first_month.isoformat()}"
            )
        blended: list[float] = []
        for month in proxy_months:
            month_total = 0.0
            for series_id, weight in rule.proxy_weights.items():
                cell = cells.get((series_id, month))
                if cell is None:
                    raise ValueError(
                        f"pension splice proxy series {series_id!r} lacks a visible row for {month.isoformat()}"
                    )
                month_total += weight * cell
            blended.append(month_total)
        segment = panel_from_prices(prices, [sleeve], cutoff, rule.etf_first_month, end)
        aligned[sleeve] = tuple(blended) + segment.returns[sleeve]
        records.append(
            SpliceRecord(
                sleeve=sleeve,
                proxy_first_month=proxy_months[0],
                proxy_last_month=proxy_months[-1],
                etf_first_month=rule.etf_first_month,
                proxy_weights=dict(rule.proxy_weights),
            )
        )
        logger.info(
            "[DATA] event=pension_modern_splice sleeve=%s proxy_months=%d etf_first_month=%s",
            sleeve,
            len(proxy_months),
            rule.etf_first_month.isoformat(),
        )
    return MonthlyReturnPanel(tier="modern", months=grid, returns=aligned), tuple(records)


def _require_as_of(as_of: datetime) -> datetime:
    if not isinstance(as_of, datetime) or as_of.tzinfo is None or as_of.utcoffset() is None:
        raise ValueError(f"pension as_of must be timezone-aware, got {as_of!r}")
    return as_of.astimezone(UTC)


def _shift_month(base: date, offset: int) -> date:
    total = base.year * _MONTHS_PER_YEAR + (base.month - 1) + offset
    year, month0 = divmod(total, _MONTHS_PER_YEAR)
    return date(year, month0 + 1, _calendar.monthrange(year, month0 + 1)[1])


def _targets_in_range(start: date, end: date) -> tuple[date, ...]:
    if start > end:
        raise ValueError(f"pension window start {start.isoformat()} is after end {end.isoformat()}")
    targets: list[date] = []
    cursor = date(start.year, start.month, 1)
    while True:
        month_end = date(cursor.year, cursor.month, _calendar.monthrange(cursor.year, cursor.month)[1])
        if month_end > end:
            break
        if month_end >= start:
            targets.append(month_end)
        cursor = _shift_month(cursor, 1)
    if not targets:
        raise ValueError(f"pension window [{start.isoformat()}, {end.isoformat()}] covers no month-end")
    return tuple(targets)


def _visible_proxy_cells(
    research: pl.DataFrame, cutoff: datetime, needed: set[str]
) -> dict[tuple[str, date], float]:
    for column in ("series_id", "period_end", "simple_return", "available_at"):
        if column not in research.columns:
            raise ValueError(f"pension research frame misses required column {column!r}")
    scoped = research.filter(
        (pl.col("series_id").is_in(sorted(needed))) & (pl.col("available_at") <= pl.lit(cutoff))
    )
    grouped = scoped.group_by(["series_id", "period_end"]).agg(
        pl.len().alias("count"), pl.col("simple_return").first().alias("simple_return")
    )
    overfull = grouped.filter(pl.col("count") > 1)
    if not overfull.is_empty():
        guilty = overfull.get_column("series_id").to_list()[0]
        raise ValueError(f"pension splice proxy series {guilty!r} carries duplicate month-ends")
    return {
        (str(series_id), month): float(value)
        for series_id, month, value in zip(
            grouped.get_column("series_id").to_list(),
            grouped.get_column("period_end").to_list(),
            grouped.get_column("simple_return").to_list(),
            strict=True,
        )
    }
