"""Integer-lot buy fill that recycles prior USD dust into the next ticket."""

from __future__ import annotations

import math
from typing import Final

__all__ = ["fill_integer_buys"]

_BPS: Final[float] = 10_000.0
_WEIGHT_SUM_TOLERANCE: Final[float] = 1e-6
_RESIDUAL_TOLERANCE: Final[float] = 1e-9


def fill_integer_buys(
    *,
    cash_usd: float,
    sleeve_budget_krw: float,
    fx_gross: float,
    weights: dict[str, float],
    prices: dict[str, float],
    commission_bps: float,
) -> tuple[dict[str, int], float, float]:
    """Fill integer-lot buys from one ticket funded by recycled dust plus fresh KRW.

    The trade notional is ``cash_usd + sleeve_budget_krw / fx_gross``; commission
    accrues on that whole notional. Net proceeds split across positive-weight
    tickers proportionally and round down to whole lots, so the returned residual
    stays below the sum of traded prices and never stacks month over month.
    FX spread costs stay with the caller because they apply only to newly
    converted KRW.

    Returns:
        Per-ticker integer lots (zero lots included), the new ``cash_usd``
        residual, and the commission cost in KRW at the gross rate.

    Raises:
        ValueError: When ``cash_usd``, ``sleeve_budget_krw``, or ``commission_bps``
            is negative or non-finite, ``fx_gross`` is non-positive, ``weights``
            is not a nonnegative simplex, or a weighted ticker's price is missing,
            non-finite, or non-positive.
    """
    if not math.isfinite(cash_usd) or cash_usd < 0.0:
        raise ValueError(f"cash_usd must be finite and nonnegative, got {cash_usd!r}")
    if not math.isfinite(sleeve_budget_krw) or sleeve_budget_krw < 0.0:
        raise ValueError(f"sleeve_budget_krw must be finite and nonnegative, got {sleeve_budget_krw!r}")
    if not math.isfinite(fx_gross) or fx_gross <= 0.0:
        raise ValueError(f"fx_gross must be finite and positive, got {fx_gross!r}")
    if not math.isfinite(commission_bps) or commission_bps < 0.0:
        raise ValueError(f"commission_bps must be finite and nonnegative, got {commission_bps!r}")
    total_weight = sum(weights.values())
    if any(not math.isfinite(weight) or weight < 0.0 for weight in weights.values()) or abs(
        total_weight - 1.0
    ) > _WEIGHT_SUM_TOLERANCE:
        raise ValueError(f"weights must form a nonnegative simplex, got sum={total_weight!r}")

    trade_usd = cash_usd + sleeve_budget_krw / fx_gross
    fee_usd = trade_usd * commission_bps / _BPS
    net_usd = trade_usd - fee_usd
    buys: dict[str, int] = {}
    spent_usd = 0.0
    for ticker, weight in weights.items():
        if weight <= 0.0:
            buys[ticker] = 0
            continue
        price = prices.get(ticker)
        if price is None or not math.isfinite(price) or price <= 0.0:
            raise ValueError(f"price for weighted ticker {ticker!r} must be finite and positive")
        lot = math.floor(net_usd * weight / price)
        buys[ticker] = lot
        spent_usd += lot * price
    residual_usd = net_usd - spent_usd
    if residual_usd < -_RESIDUAL_TOLERANCE * max(trade_usd, 1.0):
        raise ValueError("negative residual after integer-lot fill")
    return buys, max(residual_usd, 0.0), fee_usd * fx_gross
