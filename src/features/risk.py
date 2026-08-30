"""Trailing risk features."""

from __future__ import annotations

import math
from datetime import UTC
from typing import TYPE_CHECKING

import polars as pl

from src.data.pit import AVAILABLE_AT, TS_DTYPE

if TYPE_CHECKING:
    from datetime import datetime

__all__ = ["trailing_simple_corr", "trailing_simple_vol"]


def trailing_simple_corr(
    returns_a: pl.DataFrame, returns_b: pl.DataFrame, *, as_of_ts: datetime, window: int = 63
) -> float:
    """Sample correlation (ddof=1) of the last ``window`` overlapping returns visible at ``as_of_ts``.

    Inner-joins on ``date`` after filtering both frames to ``available_at <= as_of_ts``,
    then takes the tail ``window`` rows. Fails closed when the window would admit
    no correlation.
    """
    if as_of_ts.tzinfo is None:
        raise ValueError(f"as_of_ts must be timezone-aware, got naive datetime {as_of_ts!r}")
    if window < 2:
        raise ValueError(f"window must admit a sample correlation (>= 2 observations), got {window}")
    cutoff = as_of_ts.astimezone(UTC)
    visible_a = returns_a.filter(pl.col(AVAILABLE_AT) <= pl.lit(cutoff, dtype=TS_DTYPE))
    visible_b = returns_b.filter(pl.col(AVAILABLE_AT) <= pl.lit(cutoff, dtype=TS_DTYPE))
    a_sel = visible_a.select(["date", "simple_return"]).sort("date")
    b_sel = visible_b.select(["date", "simple_return"]).sort("date")
    joined = a_sel.join(b_sel, on="date", how="inner", suffix="_b")
    if joined.height < window:
        raise ValueError(f"trailing corr requires {window} overlapping finite returns, found {joined.height}")
    tail = joined.tail(window)
    # check finite and not null
    if any(c.null_count() > 0 for c in tail.get_columns()):
        raise ValueError("trailing corr requires finite overlapping returns with no nulls")
    # Use Series for checks
    a_vals = tail.get_column("simple_return")
    b_vals = tail.get_column("simple_return_b")
    if not bool(a_vals.is_finite().all()) or not bool(b_vals.is_finite().all()):
        raise ValueError("trailing corr requires finite overlapping returns")
    if a_vals.len() != window or b_vals.len() != window:
        raise ValueError(f"trailing corr requires {window} overlapping rows, found {a_vals.len()}")
    a_std = a_vals.std(ddof=1)
    b_std = b_vals.std(ddof=1)
    if not isinstance(a_std, float) or not math.isfinite(a_std) or a_std <= 0.0:
        raise ValueError(f"trailing corr requires strictly positive finite stdev for a, got {a_std!r}")
    if not isinstance(b_std, float) or not math.isfinite(b_std) or b_std <= 0.0:
        raise ValueError(f"trailing corr requires strictly positive finite stdev for b, got {b_std!r}")
    # compute sample correlation
    # convert to python lists for numeric stability
    a_list = a_vals.to_list()
    b_list = b_vals.to_list()
    n = len(a_list)
    mean_a = sum(a_list) / n
    mean_b = sum(b_list) / n
    cov = sum((x - mean_a) * (y - mean_b) for x, y in zip(a_list, b_list, strict=True)) / (n - 1)
    corr = cov / (a_std * b_std)
    if not math.isfinite(corr):
        raise ValueError(f"trailing corr produced non-finite correlation {corr!r}")
    # clip only for float overflow slightly beyond 1
    if corr > 1.0:
        corr = 1.0
    elif corr < -1.0:
        corr = -1.0
    if not math.isfinite(corr):
        raise ValueError(f"trailing corr non-finite after clipping {corr!r}")
    return float(corr)


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
