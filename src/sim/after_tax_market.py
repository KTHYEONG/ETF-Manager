"""PIT-visible market indexes for the after-tax engine."""

from __future__ import annotations

import bisect
import math
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from datetime import date, datetime

__all__ = [
    "AfterTaxDataError",
    "_CpiIndex",
    "_FxIndex",
    "_PriceIndex",
    "_RateIndex",
]


class AfterTaxDataError(RuntimeError):
    """Missing or stale raw price, base-rate FX, CPI, or rate at a required instant; never skipped."""


class _PriceIndex:
    """Index pinned tradable prices and corporate actions for causal execution marks.

    Args:
        frame: PIT-visible PRICES rows.

    Raises:
        AfterTaxDataError: If an execution mark is missing or invalid.
    """

    def __init__(self, frame: pl.DataFrame) -> None:
        missing = [column for column in ("ticker", "date", "close", "adjusted_close", "available_at") if column not in frame.columns]
        if missing:
            raise AfterTaxDataError(f"prices frame lacks columns {missing}")
        self._rows = {
            (row["ticker"], row["date"]): (row["close"], row["adjusted_close"], row["available_at"])
            for row in frame.iter_rows(named=True)
        }

    def price(self, ticker: str, day: date, instant: datetime, *, adjusted: bool) -> float:
        """Close visible at ``instant``; fail-closed on gaps and invalid marks."""
        row = self._rows.get((ticker, day))
        if row is None or row[2] > instant:
            raise AfterTaxDataError(f"missing {ticker!r} price row on {day.isoformat()} at its execution close")
        value = row[1] if adjusted else row[0]
        if value is None or not isinstance(value, int | float) or not math.isfinite(value) or value <= 0.0:
            raise AfterTaxDataError(f"non-positive close for {ticker!r} on {day.isoformat()}")
        return float(value)


class _FxIndex:
    """Once-per-run sorted base-rate series resolved as-of with a staleness bound."""

    def __init__(self, frame: pl.DataFrame) -> None:
        missing = [column for column in ("date", "usdkrw", "available_at") if column not in frame.columns]
        if missing:
            raise AfterTaxDataError(f"fx frame lacks columns {missing}")
        rows = sorted(
            ((row["date"], row["usdkrw"], row["available_at"]) for row in frame.iter_rows(named=True)),
            key=lambda item: item[0],
        )
        self._dates = [item[0] for item in rows]
        self._rows = rows

    def resolve(self, day: date, instant: datetime, max_staleness_days: int) -> float:
        """Latest visible non-null rate with ``date <= day`` inside the staleness bound."""
        position = bisect.bisect_right(self._dates, day) - 1
        while position >= 0:
            rate_date, value, available_at = self._rows[position]
            position -= 1
            if available_at > instant:
                continue
            if value is None or not isinstance(value, int | float) or not math.isfinite(value) or value <= 0.0:
                continue
            if (day - rate_date).days > max_staleness_days:
                break
            return float(value)
        raise AfterTaxDataError(f"missing usdkrw row on {day.isoformat()} within staleness bound")


class _CpiIndex:
    """Once-per-run CPI levels; latest positive visible period_end wins."""

    def __init__(self, frame: pl.DataFrame) -> None:
        missing = [column for column in ("period_end", "value", "available_at") if column not in frame.columns]
        if missing:
            raise AfterTaxDataError(f"cpi frame lacks columns {missing}")
        self._rows = sorted(
            ((row["period_end"], row["value"], row["available_at"]) for row in frame.iter_rows(named=True)),
            key=lambda item: item[0],
        )

    def resolve(self, instant: datetime) -> float:
        """Latest positive finite level visible at ``instant``."""
        level: float | None = None
        for _period_end, value, available_at in self._rows:
            if available_at > instant:
                continue
            if value is None or not isinstance(value, int | float) or not math.isfinite(value) or value <= 0.0:
                continue
            level = float(value)
        if level is None:
            raise AfterTaxDataError("missing positive CPI row at execution close")
        return level


class _RateIndex:
    """Once-per-run RATES observations for the cash-sleeve accrual."""

    def __init__(self, frame: pl.DataFrame | None) -> None:
        if frame is None:
            self._rows: dict[str, list[tuple[date, float, datetime]]] = {}
            self._loaded = False
            return
        missing = [column for column in ("series_id", "observation_date", "value", "available_at") if column not in frame.columns]
        if missing:
            raise AfterTaxDataError(f"rates frame lacks columns {missing}")
        rows: dict[str, list[tuple[date, float, datetime]]] = {}
        for row in frame.iter_rows(named=True):
            value = row["value"]
            if value is None or not isinstance(value, int | float) or not math.isfinite(value):
                continue
            rows.setdefault(row["series_id"], []).append((row["observation_date"], float(value), row["available_at"]))
        self._rows = {series: sorted(entries) for series, entries in rows.items()}
        self._loaded = True

    def resolve(self, series_id: str, instant: datetime) -> float:
        """Latest non-null observation visible at ``instant``."""
        if not self._loaded:
            raise AfterTaxDataError(f"rates series {series_id!r} required but no RATES frame was passed")
        latest: float | None = None
        for _observation_date, value, available_at in self._rows.get(series_id, []):
            if available_at > instant:
                continue
            latest = value
        if latest is None:
            raise AfterTaxDataError(f"missing {series_id!r} rate at its required instant")
        return latest
