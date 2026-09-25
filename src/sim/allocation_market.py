"""PIT-visible market reads for the allocation engine."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import polars as pl

from src.data.query import load_as_of
from src.data.schema import Dataset

if TYPE_CHECKING:
    from datetime import date, datetime

__all__ = [
    "AllocationDataError",
    "visible_close",
    "visible_cpi",
    "visible_fx",
]


class AllocationDataError(RuntimeError):
    """Missing PIT price, FX, or CPI at an execution close; never skipped silently."""


def visible_close(
    prices: pl.DataFrame,
    ticker: str,
    session: date,
    close_ts: datetime,
    *,
    adjusted: bool = True,
) -> float:
    """Resolve one causal execution close from a pinned tradable price frame.

    Args:
        prices: PIT-visible tradable price rows.
        ticker: Execution instrument.
        session: Execution session.
        close_ts: Exchange close instant.
        adjusted: Use the split/dividend-adjusted close when true; otherwise use raw close.

    Returns:
        Positive finite executable close.

    Raises:
        AllocationDataError: If the close is missing, invalid, or unavailable.
    """
    price_field = "adjusted_close" if adjusted else "close"
    if "ticker" not in prices.columns or price_field not in prices.columns:
        raise AllocationDataError(
            f"price frame lacks {price_field} for {ticker!r} on {session.isoformat()}"
        )
    visible = load_as_of(prices, Dataset.PRICES, close_ts)
    rows = visible.filter((pl.col("ticker") == ticker) & (pl.col("date") == session))
    if rows.is_empty():
        raise AllocationDataError(f"missing {ticker!r} price row on {session.isoformat()} at its execution close")
    value = rows.item(0, price_field)
    if value is None or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0.0:
        raise AllocationDataError(f"non-positive {price_field} for {ticker!r} on {session.isoformat()}")
    return float(value)


def visible_fx(fx: pl.DataFrame, session: date, close_ts: datetime) -> float:
    """Resolve one causal USD/KRW rate from a pinned FX frame.

    Args:
        fx: PIT-visible FX rows.
        session: Execution session.
        close_ts: Exchange close instant.

    Returns:
        Positive finite USD/KRW rate.

    Raises:
        AllocationDataError: If the rate is missing, invalid, or unavailable.
    """
    visible = load_as_of(fx, Dataset.FX, close_ts)
    rows = visible.filter(pl.col("date") == session)
    if rows.is_empty():
        raise AllocationDataError(f"missing usdkrw row on {session.isoformat()} at its execution close")
    value = rows.item(0, "usdkrw")
    if value is None or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0.0:
        raise AllocationDataError(f"null or non-positive usdkrw on {session.isoformat()}")
    return float(value)


def visible_cpi(cpi: pl.DataFrame, session: date, close_ts: datetime) -> float:
    """Resolve one causal CPI level from a pinned CPI frame.

    Args:
        cpi: PIT-visible CPI rows.
        session: Execution session.
        close_ts: Exchange close instant.

    Returns:
        Positive finite CPI level.

    Raises:
        AllocationDataError: If the level is missing, invalid, or unavailable.
    """
    visible = load_as_of(cpi, Dataset.CPI, close_ts)
    rows = visible.filter(pl.col("value").is_finite() & (pl.col("value") > 0.0)).sort("period_end")
    if rows.is_empty():
        raise AllocationDataError(f"missing positive CPI row on {session.isoformat()} at its execution close")
    return float(rows.item(rows.height - 1, "value"))
