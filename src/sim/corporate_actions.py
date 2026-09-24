"""Vendor corporate-action normalization: exact split ratios and cash dividends."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from fractions import Fraction
from typing import TYPE_CHECKING, Final

import polars as pl

if TYPE_CHECKING:
    from collections.abc import Collection

__all__ = [
    "CorporateAction",
    "CorporateActionError",
    "CorporateActionKind",
    "corporate_actions_from_prices",
    "normalize_split_factor",
]

_SPLIT_REL_TOL: Final[float] = 1e-4


class CorporateActionError(ValueError):
    """Vendor corporate-action data cannot be mapped to an exact share ratio."""


class CorporateActionKind(StrEnum):
    """Split versus cash-dividend ex-date events."""

    SPLIT = "split"
    DIVIDEND = "dividend"


@dataclass(frozen=True, slots=True)
class CorporateAction:
    """One ex-date event: SPLIT carries an exact share ratio, DIVIDEND a USD cash amount per share."""

    ticker: str
    ex_date: date
    kind: CorporateActionKind
    split_ratio: Fraction | None
    dividend_usd: float | None


def normalize_split_factor(raw_factor: float) -> Fraction:
    """Map a vendor split factor onto an exact n:1 or 1:n ratio.

    Accepts ``raw_factor`` within a relative tolerance of 1e-4 of an integer ``n >= 2``
    (forward split) or of ``1/n`` (reverse split).

    Raises:
        CorporateActionError: When the factor is non-finite, non-positive, equal to one,
            or not within tolerance of an exact ratio.
    """
    if isinstance(raw_factor, bool) or not isinstance(raw_factor, int | float):
        raise CorporateActionError(f"split factor must be a number, got {raw_factor!r}")
    if not math.isfinite(raw_factor) or raw_factor <= 0:
        raise CorporateActionError(f"split factor must be finite and positive, got {raw_factor!r}")
    if raw_factor == 1:
        raise CorporateActionError("split factor of one carries no corporate action")
    forward = round(raw_factor)
    if forward >= 2 and abs(raw_factor - forward) / forward <= _SPLIT_REL_TOL:
        return Fraction(forward)
    reverse = round(1.0 / raw_factor)
    if reverse >= 2 and abs(raw_factor - 1.0 / reverse) / (1.0 / reverse) <= _SPLIT_REL_TOL:
        return Fraction(1, reverse)
    raise CorporateActionError(f"split factor {raw_factor!r} is not within tolerance of an exact ratio")


def corporate_actions_from_prices(
    prices: pl.DataFrame, *, tickers: Collection[str], start: date, end: date
) -> tuple[CorporateAction, ...]:
    """Extract split and dividend events for ``tickers`` with ex-dates in ``[start, end]``.

    Reads the raw ``dividend`` and ``split_factor`` columns of Dataset.PRICES; events are
    ordered by (ex_date, ticker, SPLIT before DIVIDEND) so a same-day dividend is paid on
    the post-split share count, matching vendor per-share amounts on the ex-date.

    Raises:
        CorporateActionError: On any unmappable split factor or a negative dividend.
    """
    wanted = set(tickers)
    events: list[CorporateAction] = []
    for row in prices.iter_rows(named=True):
        ticker = row["ticker"]
        ex_date = row["date"]
        if ticker not in wanted or ex_date < start or ex_date > end:
            continue
        split_factor = row["split_factor"]
        if split_factor != 1.0:
            events.append(
                CorporateAction(
                    ticker=ticker,
                    ex_date=ex_date,
                    kind=CorporateActionKind.SPLIT,
                    split_ratio=normalize_split_factor(split_factor),
                    dividend_usd=None,
                )
            )
        dividend = row["dividend"]
        if dividend != 0.0:
            if not isinstance(dividend, int | float) or not math.isfinite(dividend) or dividend < 0:
                raise CorporateActionError(f"dividend for {ticker!r} on {ex_date!r} is not a valid cash amount")
            events.append(
                CorporateAction(
                    ticker=ticker,
                    ex_date=ex_date,
                    kind=CorporateActionKind.DIVIDEND,
                    split_ratio=None,
                    dividend_usd=float(dividend),
                )
            )
    events.sort(key=lambda event: (event.ex_date, event.ticker, 0 if event.kind is CorporateActionKind.SPLIT else 1))
    return tuple(events)
