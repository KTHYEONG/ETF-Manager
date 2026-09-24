"""Lot-level KRW cost-basis book for Korean overseas-equity tax attribution."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from fractions import Fraction
from typing import Final

from src.sim.tax import BasisMethod

__all__ = [
    "RealizedDisposal",
    "TaxLotBook",
    "TaxLotError",
]

_SHORT_TOLERANCE: Final[float] = 1e-9
_GAIN_BUDGET_TOLERANCE: Final[float] = 1e-6


class TaxLotError(ValueError):
    """Invalid lot operation (short sale, non-finite quantity, unknown ticker)."""


@dataclass(frozen=True, slots=True)
class RealizedDisposal:
    """One settled disposal translated into KRW for Korean tax attribution.

    ``tax_year`` is the settlement session's calendar year; ``gain_krw`` is
    ``proceeds_krw - basis_krw`` and may be negative.
    """

    ticker: str
    trade_session: date
    settle_session: date
    quantity: float
    proceeds_krw: float
    basis_krw: float
    gain_krw: float

    @property
    def tax_year(self) -> int:
        """Settlement session's calendar year, which attributes the taxable year."""
        return self.settle_session.year


class TaxLotBook:
    """Per-ticker acquisition lots with KRW cost basis for Korean overseas-equity tax.

    Each acquisition stores quantity and KRW cost per share (USD gross cost including
    buy commission, translated at the settlement-date base rate). FIFO consumes the
    oldest lots first; MOVING_AVERAGE keeps one pooled lot per ticker. Quantities are
    floats so fractional-share research runs share the same book; integer-lot callers
    pass whole numbers.
    """

    def __init__(self, method: BasisMethod) -> None:
        """Bind the book to one basis identification method for its whole life."""
        self._method = method
        self._lots: dict[str, list[list[float]]] = {}

    def buy(
        self,
        ticker: str,
        quantity: float,
        gross_cost_usd: float,
        settle_fx: float,
        *,
        trade_session: date,
        settle_session: date,
    ) -> None:
        """Record an acquisition; KRW basis = ``gross_cost_usd * settle_fx``.

        Raises:
            TaxLotError: When quantity or cost is non-finite or non-positive, or ``settle_fx <= 0``.
        """
        _require_positive(quantity, "quantity")
        _require_positive(gross_cost_usd, "gross_cost_usd")
        _require_positive(settle_fx, "settle_fx")
        cost_per_share = gross_cost_usd * settle_fx / quantity
        lots = self._lots.setdefault(ticker, [])
        if self._method is BasisMethod.MOVING_AVERAGE and lots:
            held_quantity, held_cost = lots[0]
            pooled_quantity = held_quantity + quantity
            lots[0] = [pooled_quantity, (held_quantity * held_cost + quantity * cost_per_share) / pooled_quantity]
        else:
            lots.append([quantity, cost_per_share])

    def sell(
        self,
        ticker: str,
        quantity: float,
        net_proceeds_usd: float,
        settle_fx: float,
        *,
        trade_session: date,
        settle_session: date,
    ) -> RealizedDisposal:
        """Dispose of ``quantity`` shares; proceeds are net of sell commission.

        Raises:
            TaxLotError: On a short sale (quantity above holdings beyond 1e-9), non-finite
                inputs, or an unknown ticker.
        """
        _require_positive(quantity, "quantity")
        if not _is_number(net_proceeds_usd) or net_proceeds_usd < 0:
            raise TaxLotError(f"net_proceeds_usd must be finite and nonnegative, got {net_proceeds_usd!r}")
        _require_positive(settle_fx, "settle_fx")
        lots = self._lots.get(ticker)
        if not lots:
            raise TaxLotError(f"no holdings for unknown ticker {ticker!r}")
        held = sum(entry[0] for entry in lots)
        if quantity > held + _SHORT_TOLERANCE:
            raise TaxLotError(f"short sale of {ticker!r}: asked {quantity!r} above holdings {held!r}")
        consume = min(quantity, held)
        basis_krw = 0.0
        remaining = consume
        for entry in lots:
            if remaining <= 0:
                break
            take = min(entry[0], remaining)
            basis_krw += take * entry[1]
            entry[0] -= take
            remaining -= take
        kept = [entry for entry in lots if entry[0] > _SHORT_TOLERANCE]
        if kept:
            self._lots[ticker] = kept
        else:
            del self._lots[ticker]
        proceeds_krw = net_proceeds_usd * settle_fx
        return RealizedDisposal(
            ticker=ticker,
            trade_session=trade_session,
            settle_session=settle_session,
            quantity=consume,
            proceeds_krw=proceeds_krw,
            basis_krw=basis_krw,
            gain_krw=proceeds_krw - basis_krw,
        )

    def apply_split(self, ticker: str, ratio: Fraction) -> float:
        """Scale every lot's quantity by ``ratio`` and its per-share cost by ``1/ratio``.

        Total KRW basis is invariant. For reverse splits producing a fractional
        remainder in an integer-lot book, returns the fractional share quantity that
        the caller must dispose of as cash-in-lieu; forward splits return 0.0.
        """
        if ratio <= 0:
            raise TaxLotError(f"split ratio must be positive, got {ratio!r}")
        lots = self._lots.get(ticker, [])
        scale = float(ratio)
        remainder = 0.0
        if scale >= 1.0:
            for entry in lots:
                entry[0] *= scale
                entry[1] /= scale
            return 0.0
        for entry in lots:
            grown = entry[0] * scale
            kept_quantity = math.floor(grown)
            remainder += grown - kept_quantity
            entry[0] = kept_quantity
            entry[1] /= scale
        if ticker in self._lots:
            kept = [entry for entry in lots if entry[0] > _SHORT_TOLERANCE]
            if kept:
                self._lots[ticker] = kept
            else:
                del self._lots[ticker]
        return remainder

    def quantity(self, ticker: str) -> float:
        """Total held shares of ``ticker``; zero when the ticker is unknown."""
        return sum(entry[0] for entry in self._lots.get(ticker, []))

    def basis_krw(self, ticker: str) -> float:
        """Total KRW cost basis of ``ticker``; zero when the ticker is unknown."""
        return sum(entry[0] * entry[1] for entry in self._lots.get(ticker, []))

    def tickers(self) -> tuple[str, ...]:
        """Tickers with a nonzero holding, in sorted order."""
        return tuple(sorted(self._lots))

    def unrealized_gain_krw(self, ticker: str, net_price_usd: float, fx: float) -> float:
        """Gain if every share of ``ticker`` were sold at ``net_price_usd`` (after sell commission) and ``fx``."""
        _require_nonnegative_price(net_price_usd, fx)
        return self.quantity(ticker) * net_price_usd * fx - self.basis_krw(ticker)

    def harvestable_quantity(
        self, ticker: str, net_price_usd: float, fx: float, gain_budget_krw: float
    ) -> float:
        """Largest quantity whose disposal realizes a gain no greater than ``gain_budget_krw``.

        FIFO walks lots oldest-first and stops at the first lot with non-positive gain
        (a forced FIFO disposal cannot skip it); MOVING_AVERAGE uses the pooled cost.
        Returns 0.0 when the budget is non-positive or no gain exists.
        """
        _require_nonnegative_price(net_price_usd, fx)
        if not _is_number(gain_budget_krw):
            raise TaxLotError(f"gain_budget_krw must be finite, got {gain_budget_krw!r}")
        if gain_budget_krw <= 0:
            return 0.0
        net_krw = net_price_usd * fx
        if self._method is BasisMethod.MOVING_AVERAGE:
            lots = self._lots.get(ticker, [])
            if not lots:
                return 0.0
            per_share_gain = net_krw - lots[0][1]
            if per_share_gain <= 0:
                return 0.0
            return min(lots[0][0], gain_budget_krw / per_share_gain)
        harvested = 0.0
        realized = 0.0
        for entry in self._lots.get(ticker, []):
            per_share_gain = net_krw - entry[1]
            if per_share_gain <= 0:
                break
            take = min(entry[0], (gain_budget_krw - realized) / per_share_gain)
            harvested += take
            realized += take * per_share_gain
            if take < entry[0] or realized >= gain_budget_krw - _GAIN_BUDGET_TOLERANCE:
                break
        return harvested


def _is_number(value: object) -> bool:
    """Finite-number check that rejects booleans alongside non-numeric types."""
    return not isinstance(value, bool) and isinstance(value, int | float) and math.isfinite(value)


def _require_positive(value: object, name: str) -> None:
    """Raise TaxLotError unless ``value`` is finite and strictly positive."""
    if not _is_number(value) or value <= 0:  # type: ignore[operator]
        raise TaxLotError(f"{name} must be finite and positive, got {value!r}")


def _require_nonnegative_price(net_price_usd: object, fx: object) -> None:
    """Raise TaxLotError unless price and FX are finite with a positive FX."""
    if not _is_number(net_price_usd) or net_price_usd < 0:  # type: ignore[operator]
        raise TaxLotError(f"net_price_usd must be finite and nonnegative, got {net_price_usd!r}")
    if not _is_number(fx) or fx <= 0:  # type: ignore[operator]
        raise TaxLotError(f"fx must be finite and positive, got {fx!r}")
