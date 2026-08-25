"""PIT trailing price-path drawdown."""

from __future__ import annotations

from datetime import UTC
from typing import TYPE_CHECKING

import polars as pl

from src.analytics.metrics import max_drawdown
from src.data.pit import AVAILABLE_AT, TS_DTYPE

if TYPE_CHECKING:
    from datetime import datetime

__all__ = ["trailing_price_drawdown"]


def trailing_price_drawdown(prices: pl.DataFrame, *, ticker: str, as_of_ts: datetime, window: int = 252) -> float:
    """Peak-to-trough decline (non-positive fraction) of the last ``window`` visible sessions.

    Uses only PIT-visible ``adjusted_close`` rows for ``ticker`` whose
    ``available_at`` stamp is at or before ``as_of_ts``; semantics match
    ``analytics.max_drawdown``.

    Raises:
        ValueError: When ``as_of_ts`` is naive, ``window`` is below 1, or fewer
            than ``window`` visible sessions remain.
    """
    if as_of_ts.tzinfo is None:
        raise ValueError(f"as_of_ts must be timezone-aware, got naive datetime {as_of_ts!r}")
    if window < 1:
        raise ValueError(f"window must be at least 1, got {window}")
    cutoff = as_of_ts.astimezone(UTC)
    visible = prices.filter(pl.col("ticker") == ticker).sort("date").filter(
        pl.col(AVAILABLE_AT) <= pl.lit(cutoff, dtype=TS_DTYPE)
    )
    closes = visible.get_column("adjusted_close").tail(window)
    if closes.null_count() > 0 or closes.len() < window or not bool(closes.is_finite().all()):
        raise ValueError(
            f"trailing drawdown requires {window} visible sessions for {ticker!r}, found {closes.len()} usable"
        )
    return max_drawdown([float(value) for value in closes])
