"""Pre-indexed point-in-time view over PRICES and RATES for weight rules."""

from __future__ import annotations

import math
from datetime import date, datetime

import polars as pl

__all__ = ["PitMarket", "PitMarketError"]


class PitMarketError(ValueError):
    """A rule asked for history that is not visible or not long enough at the signal instant."""


class PitMarket:
    """Pre-indexed point-in-time view over PRICES and RATES for weight rules.

    Built once per run from loaded frames so rule evaluation is O(1)/O(window) per
    decision instead of re-filtering polars frames. Every accessor takes the decision
    instant and only returns rows whose ``available_at`` is at or before it; signals use
    total-return (adjusted) closes so dividends and splits do not create false breaks.
    """

    def __init__(self, prices: pl.DataFrame, rates: pl.DataFrame | None) -> None:
        """Index adjusted closes per ticker and rate observations per series.

        Bars are sorted by date once; month ends are the last bar of each calendar
        month. Frames must carry ``available_at``; RATES ``value`` gaps stay skippable.
        """
        bars: dict[str, list[tuple[date, float, datetime]]] = {}
        for row in prices.iter_rows(named=True):
            bars.setdefault(row["ticker"], []).append((row["date"], row["adjusted_close"], row["available_at"]))
        self._bars = {ticker: sorted(entries) for ticker, entries in bars.items()}
        self._month_ends: dict[str, list[tuple[date, float, datetime]]] = {}
        for ticker, entries in self._bars.items():
            last_per_month: dict[tuple[int, int], tuple[date, float, datetime]] = {}
            for entry in entries:
                last_per_month[(entry[0].year, entry[0].month)] = entry
            self._month_ends[ticker] = sorted(last_per_month.values())
        observations: dict[str, list[tuple[date, float, datetime]]] = {}
        if rates is not None:
            for row in rates.iter_rows(named=True):
                value = row["value"]
                if value is None or not isinstance(value, int | float) or not math.isfinite(value):
                    continue
                observations.setdefault(row["series_id"], []).append(
                    (row["observation_date"], float(value), row["available_at"])
                )
        self._rates = {series: sorted(entries) for series, entries in observations.items()}
        self._rates_loaded = rates is not None

    def month_end_adjusted_closes(self, ticker: str, as_of: datetime, count: int) -> tuple[float, ...]:
        """Last ``count`` month-end adjusted closes visible at ``as_of`` (oldest first).

        Raises:
            PitMarketError: When fewer than ``count`` visible month ends exist or ``as_of`` is naive.
        """
        _require_aware(as_of)
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise PitMarketError(f"count must be a positive integer, got {count!r}")
        visible = [entry for entry in self._month_ends.get(ticker, []) if entry[2] <= as_of]
        if len(visible) < count:
            raise PitMarketError(f"only {len(visible)} visible month ends for {ticker!r}, asked {count}")
        return tuple(entry[1] for entry in visible[-count:])

    def daily_adjusted_closes(self, ticker: str, as_of: datetime, count: int) -> tuple[float, ...]:
        """Last ``count`` session adjusted closes visible at ``as_of`` (oldest first).

        Raises:
            PitMarketError: When history is too short or ``as_of`` is naive.
        """
        _require_aware(as_of)
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise PitMarketError(f"count must be a positive integer, got {count!r}")
        visible = [entry for entry in self._bars.get(ticker, []) if entry[2] <= as_of]
        if len(visible) < count:
            raise PitMarketError(f"only {len(visible)} visible sessions for {ticker!r}, asked {count}")
        return tuple(entry[1] for entry in visible[-count:])

    def rate_percent(self, series_id: str, as_of: datetime) -> float:
        """Latest non-null RATES value (percent) visible at ``as_of``.

        Raises:
            PitMarketError: When no visible value exists or rates were not loaded.
        """
        _require_aware(as_of)
        if not self._rates_loaded:
            raise PitMarketError("rates were not loaded for this market")
        visible = [entry for entry in self._rates.get(series_id, []) if entry[2] <= as_of]
        if not visible:
            raise PitMarketError(f"no visible {series_id!r} value at {as_of.isoformat()}")
        return visible[-1][1]


def _require_aware(as_of: datetime) -> None:
    """Reject naive decision instants; PIT comparisons need an absolute clock."""
    if not isinstance(as_of, datetime) or as_of.tzinfo is None:
        raise PitMarketError(f"as_of must be timezone-aware, got {as_of!r}")
