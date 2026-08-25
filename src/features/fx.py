"""PIT trailing FX features."""

from __future__ import annotations

from datetime import UTC
from typing import TYPE_CHECKING

import polars as pl

from src.data.pit import AVAILABLE_AT, TS_DTYPE

if TYPE_CHECKING:
    from datetime import datetime

__all__ = ["trailing_fx_percentile"]


def trailing_fx_percentile(fx: pl.DataFrame, *, as_of_ts: datetime, window: int = 252) -> float:
    """Midrank percentile of the last ``window`` PIT-visible positive ``usdkrw`` prints.

    Only rows with ``available_at <= as_of_ts`` and finite positive rates may enter
    the window, so a print published after the decision instant can never move it.
    The last visible print is ranked inside its own window:
    ``p = (#(x < last) + 0.5 * #(x == last)) / window`` in ``(0, 1]``; a high ``p``
    marks an expensive USD versus the trailing window.

    Raises:
        ValueError: When ``as_of_ts`` is naive, ``window`` is below 1, or fewer
            than ``window`` finite positive rates are visible at ``as_of_ts``.
    """
    if as_of_ts.tzinfo is None:
        raise ValueError(f"as_of_ts must be timezone-aware, got naive datetime {as_of_ts!r}")
    if window < 1:
        raise ValueError(f"window must be at least 1, got {window}")
    cutoff = as_of_ts.astimezone(UTC)
    visible = fx.filter(
        (pl.col(AVAILABLE_AT) <= pl.lit(cutoff, dtype=TS_DTYPE))
        & pl.col("usdkrw").is_finite()
        & (pl.col("usdkrw") > 0.0)
    ).sort("date")
    values = visible.get_column("usdkrw").tail(window)
    if values.len() < window:
        raise ValueError(
            f"trailing fx percentile requires {window} finite positive visible rates, "
            f"found {values.len()} usable"
        )
    last = float(values[-1])
    below = int((values < last).sum())
    equal = int((values == last).sum())
    return (below + 0.5 * equal) / float(window)
