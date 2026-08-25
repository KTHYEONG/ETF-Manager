"""PIT hard filters and implementation-quality ETF score."""

from __future__ import annotations

import math
from datetime import UTC, date
from typing import TYPE_CHECKING, Final

import polars as pl

from src.data.pit import AVAILABLE_AT, TS_DTYPE
from src.features.momentum import trailing_compound_return
from src.features.returns import session_returns
from src.policy.targets import PolicyError

if TYPE_CHECKING:
    from collections.abc import Mapping
    from datetime import datetime

    from src.etf.mapping import MappingConfig

__all__ = ["etf_score", "latest_metadata_row", "passes_hard_filters"]

_REQUIRED_NUMERICS: Final[tuple[str, ...]] = ("expense_ratio", "aum_usd", "avg_dollar_volume")


def latest_metadata_row(
    metadata: pl.DataFrame,
    *,
    ticker: str,
    signal_at: datetime,
) -> Mapping[str, object]:
    """Latest ETF_METADATA row of ``ticker`` visible at ``signal_at``.

    Keeps rows with ``available_at <= signal_at`` and takes the maximum
    ``(effective_date, available_at)``, so a filing published after the
    decision instant can never change the score.

    Raises:
        ValueError: When ``signal_at`` is naive.
        PolicyError: When no visible row exists for ``ticker``.
    """
    if signal_at.tzinfo is None:
        raise ValueError(f"signal_at must be timezone-aware, got naive datetime {signal_at!r}")
    cutoff = signal_at.astimezone(UTC)
    visible = metadata.filter(
        (pl.col("ticker") == ticker)
        & (pl.col(AVAILABLE_AT) <= pl.lit(cutoff, dtype=TS_DTYPE))
    ).sort(["effective_date", AVAILABLE_AT])
    if visible.is_empty():
        raise PolicyError(f"no ETF_METADATA row for {ticker!r} visible at the signal instant")
    return visible.row(visible.height - 1, named=True)


def _finite(value: object) -> float | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
        return None
    return float(value)


def passes_hard_filters(
    row: Mapping[str, object],
    *,
    sleeve: str,
    signal_at: datetime,
    mapping: MappingConfig,
) -> bool:
    """Return whether ``row`` may be an implementation of ``sleeve`` at ``signal_at``.

    A row fails when leveraged or inverse, below AUM or dollar-volume floors,
    with a track record shorter than the minimum, on a sleeve mismatch, or when
    its ticker is not a listed candidate for the economic sleeve. Null inputs
    fail closed.
    """
    ticker = row.get("ticker")
    if not isinstance(ticker, str) or ticker not in mapping.candidates.get(sleeve, ()):
        return False
    if row.get("sleeve") != sleeve:
        return False
    leveraged = row.get("is_leveraged")
    inverse = row.get("is_inverse")
    if not isinstance(leveraged, int) or not isinstance(inverse, int):
        return False
    if leveraged != 0 or inverse != 0:
        return False
    aum_usd = _finite(row.get("aum_usd"))
    dollar_volume = _finite(row.get("avg_dollar_volume"))
    if aum_usd is None or dollar_volume is None:
        return False
    if aum_usd < mapping.min_aum_usd or dollar_volume < mapping.min_dollar_volume:
        return False
    inception = row.get("inception_date")
    if not isinstance(inception, date):
        return False
    return (signal_at.date() - inception).days >= mapping.min_track_record_days


def _fit_and_td(
    prices: pl.DataFrame,
    *,
    ticker: str,
    sleeve: str,
    signal_at: datetime,
    mapping: MappingConfig,
) -> tuple[float, float]:
    """PIT beta fit and trailing tracking difference; ``ValueError`` fails closed."""
    candidate_returns = session_returns(prices, ticker=ticker).filter(
        pl.col(AVAILABLE_AT) <= pl.lit(signal_at, dtype=TS_DTYPE)
    )
    sleeve_rets = session_returns(prices, ticker=sleeve).filter(
        pl.col(AVAILABLE_AT) <= pl.lit(signal_at, dtype=TS_DTYPE)
    )
    overlap = (
        candidate_returns.join(sleeve_rets, on="date", how="inner", suffix="_sleeve")
        .sort("date")
        .tail(mapping.fit_window)
    )
    if overlap.height < mapping.fit_window:
        raise ValueError(
            f"fit requires {mapping.fit_window} overlapping PIT sessions, found {overlap.height}"
        )
    candidate_values = overlap.get_column("simple_return").to_list()
    sleeve_values = overlap.get_column("simple_return_sleeve").to_list()
    mean_c = sum(candidate_values) / len(candidate_values)
    mean_s = sum(sleeve_values) / len(sleeve_values)
    cov_cs = (
        sum((c - mean_c) * (s - mean_s) for c, s in zip(candidate_values, sleeve_values, strict=True))
        / len(candidate_values)
    )
    var_s = sum((s - mean_s) ** 2 for s in sleeve_values) / len(sleeve_values)
    if var_s == 0.0:
        if candidate_values != sleeve_values:
            raise ValueError("zero sleeve variance with divergent candidate returns; fit fails closed")
        beta_fit = 1.0
    else:
        beta_fit = max(0.0, 1.0 - abs(cov_cs / var_s - 1.0))
    td_candidate = trailing_compound_return(candidate_returns, as_of_ts=signal_at, window=mapping.td_window)
    td_sleeve = trailing_compound_return(sleeve_rets, as_of_ts=signal_at, window=mapping.td_window)
    return beta_fit, td_candidate - td_sleeve


def etf_score(
    prices: pl.DataFrame,
    metadata: pl.DataFrame,
    *,
    ticker: str,
    sleeve: str,
    signal_at: datetime,
    mapping: MappingConfig,
) -> float:
    """Implementation score of ``ticker`` for economic ``sleeve`` at ``signal_at``.

    ``score = fit - expense_weight*expense_ratio - td_weight*|td| -
    spread_weight/sqrt(avg_dollar_volume)`` using only the metadata vintage and
    price bars visible at ``signal_at``. Higher is better; the score never chases
    trailing performance beyond its bounded tracking-difference penalty.

    Raises:
        ValueError: When ``signal_at`` is naive.
        PolicyError: When metadata or return windows fail closed.
    """
    if signal_at.tzinfo is None:
        raise ValueError(f"signal_at must be timezone-aware, got naive datetime {signal_at!r}")
    row = latest_metadata_row(metadata, ticker=ticker, signal_at=signal_at)
    if not passes_hard_filters(row, sleeve=sleeve, signal_at=signal_at, mapping=mapping):
        raise PolicyError(f"{ticker!r} fails hard filters as an implementation of {sleeve!r}")
    numerics = {name: _finite(row.get(name)) for name in _REQUIRED_NUMERICS}
    if any(value is None for value in numerics.values()):
        missing = sorted(name for name, value in numerics.items() if value is None)
        raise PolicyError(f"ETF_METADATA row for {ticker!r} lacks finite {', '.join(missing)}")
    dollar_volume = numerics["avg_dollar_volume"]
    if dollar_volume is None or dollar_volume <= 0.0:
        raise PolicyError(f"non-positive avg_dollar_volume for {ticker!r}")
    assert numerics["expense_ratio"] is not None
    expense_ratio: float = numerics["expense_ratio"]
    try:
        if ticker == sleeve:
            # Self-mapped sleeves need no regression; fit is exact by construction.
            beta_fit, tracking_difference = 1.0, 0.0
        else:
            beta_fit, tracking_difference = _fit_and_td(
                prices, ticker=ticker, sleeve=sleeve, signal_at=signal_at, mapping=mapping
            )
    except ValueError as exc:
        raise PolicyError(f"score feature failed closed for {ticker!r}: {exc}") from exc
    spread = 1.0 / math.sqrt(dollar_volume)
    score = (
        beta_fit
        - mapping.expense_weight * expense_ratio
        - mapping.td_weight * abs(tracking_difference)
        - mapping.spread_weight * spread
    )
    if not math.isfinite(score):
        raise PolicyError(f"non-finite ETF score for {ticker!r}")
    return score
