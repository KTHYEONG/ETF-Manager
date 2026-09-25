"""Source-labeled KRW mark construction for the pension engine."""

from __future__ import annotations

import bisect
import math
from datetime import date
from typing import cast

import polars as pl

__all__ = [
    "PensionDataError",
    "_live_marks",
    "_proxy_marks",
    "proxy_krw_marks",
]


class PensionDataError(RuntimeError):
    """Missing sessions, prices, FX, or payout rows; the affected arm aborts, never fills forward."""


def _proxy_marks(
    prices: pl.DataFrame, fx: pl.DataFrame | None, max_fx_age_days: int, withholding_rate: float
) -> tuple[dict[tuple[str, date], float], dict[tuple[str, date], float]]:
    """Build net-of-withholding KRW marks and per-unit withheld tax for US proxies."""
    if fx is None:
        raise PensionDataError("US_PROXY mode requires an as-of USD/KRW frame")
    for column in ("ticker", "date", "close", "adjusted_close", "dividend"):
        if column not in prices.columns:
            raise PensionDataError(f"US_PROXY prices miss required column {column!r}")
    for column in ("date", "usdkrw"):
        if column not in fx.columns:
            raise PensionDataError(f"US_PROXY fx misses required column {column!r}")
    fx_rows = sorted(fx.select("date", "usdkrw").to_dicts(), key=lambda row: row["date"])
    if not fx_rows:
        raise PensionDataError("US_PROXY fx frame is empty")
    if any(row["usdkrw"] is None or row["usdkrw"] <= 0 for row in fx_rows):
        raise PensionDataError("US_PROXY fx carries a missing or nonpositive quote")
    fx_dates = [row["date"] for row in fx_rows]
    fx_values = [float(row["usdkrw"]) for row in fx_rows]

    def _fx_at(day: date) -> float:
        position = bisect.bisect_right(fx_dates, day) - 1
        if position < 0:
            raise PensionDataError(f"US_PROXY fx is missing on or before {day.isoformat()}")
        if (day - fx_dates[position]).days > max_fx_age_days:
            raise PensionDataError(
                f"US_PROXY fx quote on {fx_dates[position].isoformat()} is stale for {day.isoformat()}"
            )
        return fx_values[position]

    marks: dict[tuple[str, date], float] = {}
    withheld: dict[tuple[str, date], float] = {}
    rows = prices.select("ticker", "date", "close", "adjusted_close", "dividend").to_dicts()
    by_ticker: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_ticker.setdefault(str(row["ticker"]), []).append(row)
    for ticker, ticker_rows in by_ticker.items():
        ordered = sorted(ticker_rows, key=lambda row: row["date"])  # type: ignore[arg-type,return-value]
        index: float | None = None
        prev_close = 0.0
        prev_index = 0.0
        prev_quote = 0.0
        for row in ordered:
            day = cast("date", row["date"])
            close = row["close"]
            quote = row["adjusted_close"]
            dividend = row["dividend"]
            if isinstance(close, bool) or not isinstance(close, float | int):
                raise PensionDataError(f"US_PROXY close for {ticker!r} on {day!r} is missing or zero")
            if not math.isfinite(close) or close <= 0:
                raise PensionDataError(f"US_PROXY close for {ticker!r} on {day!r} is missing or zero")
            if isinstance(quote, bool) or not isinstance(quote, float | int):
                raise PensionDataError(f"US_PROXY price for {ticker!r} on {day!r} is missing or zero")
            if not math.isfinite(quote) or quote <= 0:
                raise PensionDataError(f"US_PROXY price for {ticker!r} on {day!r} is missing or zero")
            if isinstance(dividend, bool) or not isinstance(dividend, float | int):
                raise PensionDataError(f"US_PROXY dividend for {ticker!r} on {day!r} is missing")
            if not math.isfinite(dividend) or dividend < 0:
                raise PensionDataError(f"US_PROXY dividend for {ticker!r} on {day!r} is missing")
            fx_rate = _fx_at(day)
            if index is None:
                index = float(quote)
            else:
                ratio = float(quote) / prev_quote
                drag = withholding_rate * float(dividend) / prev_close
                index = prev_index * (ratio - drag)
                if float(dividend) > 0:
                    withheld[(ticker, day)] = (
                        withholding_rate * float(dividend) / prev_close * prev_index * fx_rate
                    )
            marks[(ticker, day)] = index * fx_rate
            prev_quote = float(quote)
            prev_close = float(close)
            prev_index = index
    return marks, withheld


def proxy_krw_marks(
    prices: pl.DataFrame,
    fx: pl.DataFrame,
    *,
    max_fx_age_days: int,
    withholding_rate: float,
) -> dict[tuple[str, date], float]:
    """Build source-labeled KRW proxy marks from certified USD research prices and causal FX.

    Args:
        prices: Existing proxy price input.
        fx: Available-at FX input.
        max_fx_age_days: Existing causal FX staleness limit.
        withholding_rate: Foreign dividend withholding assumption.

    Returns:
        Existing net-of-withholding KRW mark mapping.

    Raises:
        PensionDataError: If prices or FX cannot cover required sessions causally.
    """
    marks, _ = _proxy_marks(prices, fx, max_fx_age_days, withholding_rate)
    return marks


def _live_marks(
    prices: pl.DataFrame,
) -> tuple[dict[tuple[str, date], float], dict[tuple[str, date], float], dict[tuple[str, date], date | None], dict[tuple[str, date], float]]:
    for column in ("ticker", "date", "close_krw", "distribution_krw", "distribution_pay_date", "split_factor"):
        if column not in prices.columns:
            raise PensionDataError(f"KR_LIVE prices miss required column {column!r}")
    closes: dict[tuple[str, date], float] = {}
    distributions: dict[tuple[str, date], float] = {}
    pay_dates: dict[tuple[str, date], date | None] = {}
    splits: dict[tuple[str, date], float] = {}
    rows = prices.select("ticker", "date", "close_krw", "distribution_krw", "distribution_pay_date", "split_factor").to_dicts()
    for row in rows:
        close = row["close_krw"]
        if close is None or close <= 0:
            raise PensionDataError(f"KR_LIVE close for {row['ticker']!r} on {row['date']!r} is missing or zero")
        factor = row["split_factor"]
        if factor is None or factor <= 0:
            raise PensionDataError(f"KR_LIVE split factor for {row['ticker']!r} on {row['date']!r} is missing or zero")
        dist = row["distribution_krw"]
        if dist is None or dist < 0:
            raise PensionDataError(f"KR_LIVE distribution for {row['ticker']!r} on {row['date']!r} is missing")
        key = (str(row["ticker"]), row["date"])
        closes[key] = float(close)
        distributions[key] = float(dist)
        pay_dates[key] = row["distribution_pay_date"]
        splits[key] = float(factor)
    return closes, distributions, pay_dates, splits
