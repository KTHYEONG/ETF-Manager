"""Bounded KRW→USD conversion policy (defer only)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import polars as pl

from src.etf_manager.features.fx import trailing_fx_percentile
from src.etf_manager.policy.targets import PolicyError

if TYPE_CHECKING:
    from datetime import datetime

__all__ = [
    "CurrencyConfig",
    "conversion_fraction",
    "economic_currency",
    "trading_currency",
]

_MAX_DEFER_CEILING: Final[float] = 1.0
# Every current catalog sleeve lists on a US exchange, so trading labels share one currency.
_TRADING_CURRENCY: Final[str] = "USD"


@dataclass(frozen=True, slots=True)
class CurrencyConfig:
    """Defer parameters; conversion fraction stays in [1-max_defer, 1]."""

    max_defer: float
    percentile_window: int = 252
    expensive_percentile: float = 0.80

    def __post_init__(self) -> None:
        if not 0.0 < self.max_defer <= _MAX_DEFER_CEILING:
            raise ValueError(f"max_defer must lie in (0, {_MAX_DEFER_CEILING}], got {self.max_defer!r}")
        if not 0.0 < self.expensive_percentile < 1.0:
            raise ValueError(
                f"expensive_percentile must lie in (0, 1), got {self.expensive_percentile!r}"
            )


def conversion_fraction(fx: pl.DataFrame, signal_at: datetime, currency: CurrencyConfig) -> float:
    """Share of investable KRW to convert at ``signal_at``; never exceeds 1.

    A midrank FX percentile at or below ``expensive_percentile`` converts in full;
    above it the deferred remainder grows linearly up to ``max_defer``. Feature
    failures fail closed with ``PolicyError``.

    Raises:
        ValueError: When ``signal_at`` is naive.
        PolicyError: When the trailing FX feature fails closed.
    """
    if signal_at.tzinfo is None:
        raise ValueError(f"signal_at must be timezone-aware, got naive datetime {signal_at!r}")
    try:
        percentile = trailing_fx_percentile(fx, as_of_ts=signal_at, window=currency.percentile_window)
    except ValueError as exc:
        raise PolicyError(f"currency feature failed closed: {exc}") from exc
    threshold = currency.expensive_percentile
    if percentile <= threshold:
        return 1.0
    return 1.0 - currency.max_defer * (percentile - threshold) / (1.0 - threshold)


_ECONOMIC_BY_TICKER: Final[dict[str, str]] = {
    "VT": "MULTI",
    "VTI": "USD",
    "VEA": "DEV",
    "VWO": "EM",
    "TLT": "USD",
    "IEF": "USD",
    "BND": "USD",
}


def trading_currency(ticker: str) -> str:
    """Listed trading currency of ``ticker`` (US-listed sleeves are USD)."""
    return _TRADING_CURRENCY


def economic_currency(ticker: str) -> str:
    """Underlying economic currency bucket; unknown tickers default to USD."""
    return _ECONOMIC_BY_TICKER.get(ticker, "USD")
