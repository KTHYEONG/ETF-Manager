"""Trailing risk features."""

from __future__ import annotations

import math
from datetime import UTC
from typing import TYPE_CHECKING

import polars as pl

from src.data.pit import AVAILABLE_AT, TS_DTYPE

if TYPE_CHECKING:
    from datetime import datetime

__all__ = ["trailing_simple_vol"]


def trailing_simple_vol(returns: pl.DataFrame, *, as_of_ts: datetime, window: int = 63) -> float:
    """Sample stdev (ddof=1) of the last ``window`` simple returns visible at ``as_of_ts``.

    Only rows whose ``available_at`` stamp is at or before ``as_of_ts`` may enter
    the window, so a bar published after the decision instant can never tighten it.

    Raises:
        ValueError: When ``as_of_ts`` is naive or ``window`` admits no sample stdev,
            when fewer than ``window`` finite returns are visible, or when the
            sample stdev is not strictly positive.
    """
    if as_of_ts.tzinfo is None:
        raise ValueError(f"as_of_ts must be timezone-aware, got naive datetime {as_of_ts!r}")
    if window < 2:
        raise ValueError(f"window must admit a sample stdev (>= 2 observations), got {window}")
    cutoff = as_of_ts.astimezone(UTC)
    visible = returns.filter(pl.col(AVAILABLE_AT) <= pl.lit(cutoff, dtype=TS_DTYPE))
    recent = visible.sort("date").get_column("simple_return").tail(window)
    if recent.null_count() > 0 or recent.len() < window or not bool(recent.is_finite().all()):
        raise ValueError(f"trailing vol requires {window} finite visible returns, found {recent.len()} usable")
    stdev = recent.std(ddof=1)
    if not isinstance(stdev, float) or not math.isfinite(stdev) or stdev <= 0.0:
        raise ValueError(f"trailing vol requires a strictly positive finite stdev, got {stdev!r}")
    return stdev
