"""Session total-return features."""

from __future__ import annotations

from typing import Final

import polars as pl

from src.data.pit import AVAILABLE_AT

_OUTPUT_COLUMNS: Final[tuple[str, ...]] = ("date", "ticker", "simple_return", AVAILABLE_AT)

__all__ = ["session_returns"]


def session_returns(prices: pl.DataFrame, *, ticker: str) -> pl.DataFrame:
    """Per-session simple returns of ``ticker`` from availability-stamped adjusted closes.

    Each return is stamped with the ``available_at`` of its own (later) bar, so a
    return becomes visible only once its closing bar is visible.

    Raises:
        ValueError: When fewer than 2 rows remain for ``ticker``.
    """
    bars = prices.filter(pl.col("ticker") == ticker).sort("date")
    if bars.height < 2:
        raise ValueError(f"session_returns requires at least 2 rows for {ticker!r}, got {bars.height}")
    return (
        bars.with_columns(
            (pl.col("adjusted_close") / pl.col("adjusted_close").shift(1) - 1.0).alias("simple_return")
        )
        .slice(1)
        .select(_OUTPUT_COLUMNS)
    )
