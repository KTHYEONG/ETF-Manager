"""PIT trailing compound returns."""

from __future__ import annotations

import math
from datetime import UTC
from typing import TYPE_CHECKING

import polars as pl

from src.data.pit import AVAILABLE_AT, TS_DTYPE

if TYPE_CHECKING:
    from datetime import datetime

__all__ = ["trailing_compound_return"]


def trailing_compound_return(returns: pl.DataFrame, *, as_of_ts: datetime, window: int = 126) -> float:
    """Compound growth minus one over the last ``window`` returns visible at ``as_of_ts``.

    Only rows whose ``available_at`` stamp is at or before ``as_of_ts`` may enter
    the window, so a bar published after the decision instant can never inflate it.

    Raises:
        ValueError: When ``as_of_ts`` is naive, ``window`` is below 1, or fewer
            than ``window`` finite returns are visible.
    """
    if as_of_ts.tzinfo is None:
        raise ValueError(f"as_of_ts must be timezone-aware, got naive datetime {as_of_ts!r}")
    if window < 1:
        raise ValueError(f"window must be at least 1, got {window}")
    cutoff = as_of_ts.astimezone(UTC)
    visible = returns.filter(pl.col(AVAILABLE_AT) <= pl.lit(cutoff, dtype=TS_DTYPE))
    recent = visible.sort("date").get_column("simple_return").tail(window)
    if recent.null_count() > 0 or recent.len() < window or not bool(recent.is_finite().all()):
        raise ValueError(
            f"trailing compound return requires {window} finite visible returns, found {recent.len()} usable"
        )
    growth = 1.0
    for value in recent:
        growth *= 1.0 + float(value)
    compound = growth - 1.0
    if not math.isfinite(compound):
        raise ValueError(f"trailing compound return overflowed to a non-finite value {compound!r}")
    return compound
