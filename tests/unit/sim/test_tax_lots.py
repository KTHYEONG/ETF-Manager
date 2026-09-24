"""Unit tests for the lot-level KRW cost-basis book."""

from __future__ import annotations

import math
from datetime import date
from fractions import Fraction

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.sim.tax import BasisMethod
from src.sim.tax_lots import RealizedDisposal, TaxLotBook, TaxLotError

_TRADE = date(2025, 6, 10)
_SETTLE = date(2025, 6, 11)


def _buy(book: TaxLotBook, quantity: float, cost_per_share_krw: float, fx: float = 1000.0) -> None:
    book.buy(
        "T",
        quantity,
        quantity * cost_per_share_krw / fx,
        fx,
        trade_session=_TRADE,
        settle_session=_SETTLE,
    )


def _sell(book: TaxLotBook, quantity: float, proceeds_krw: float, fx: float = 1000.0) -> RealizedDisposal:
    return book.sell(
        "T",
        quantity,
        proceeds_krw / fx,
        fx,
        trade_session=_TRADE,
        settle_session=_SETTLE,
    )


def test_fifo_consumes_oldest_lot() -> None:
    """Five shares out of the cheap oldest lot carry 5M basis on 15M proceeds."""
    book = TaxLotBook(BasisMethod.FIFO)
    _buy(book, 10.0, 1_000_000.0)
    _buy(book, 10.0, 2_000_000.0)
    disposal = _sell(book, 5.0, 15_000_000.0)
    assert disposal.basis_krw == pytest.approx(5_000_000.0)
    assert disposal.gain_krw == pytest.approx(10_000_000.0)
    assert disposal.proceeds_krw == pytest.approx(15_000_000.0)
    assert book.quantity("T") == pytest.approx(15.0)


def test_moving_average_pools_cost() -> None:
    """The same buys under averaging carry the pooled 1.5M per-share cost."""
    book = TaxLotBook(BasisMethod.MOVING_AVERAGE)
    _buy(book, 10.0, 1_000_000.0)
    _buy(book, 10.0, 2_000_000.0)
    disposal = _sell(book, 5.0, 15_000_000.0)
    assert disposal.basis_krw == pytest.approx(7_500_000.0)
    assert disposal.gain_krw == pytest.approx(7_500_000.0)


def test_methods_agree_without_disposals() -> None:
    """Before any sale both methods report identical basis and unrealized gain."""
    fifo = TaxLotBook(BasisMethod.FIFO)
    average = TaxLotBook(BasisMethod.MOVING_AVERAGE)
    for book in (fifo, average):
        _buy(book, 10.0, 1_000_000.0)
        _buy(book, 10.0, 2_000_000.0)
    assert fifo.basis_krw("T") == pytest.approx(average.basis_krw("T"))
    assert fifo.unrealized_gain_krw("T", 1800.0, 1000.0) == pytest.approx(
        average.unrealized_gain_krw("T", 1800.0, 1000.0)
    )


def test_short_sale_fails_closed() -> None:
    """Selling above holdings raises and leaves the book untouched."""
    book = TaxLotBook(BasisMethod.FIFO)
    _buy(book, 3.0, 1_000_000.0)
    with pytest.raises(TaxLotError, match="short sale"):
        _sell(book, 4.0, 4_000_000.0)
    assert book.quantity("T") == pytest.approx(3.0)
    assert book.basis_krw("T") == pytest.approx(3_000_000.0)


def test_split_preserves_basis() -> None:
    """A 3:1 split triples quantity with basis invariant to 1e-12."""
    book = TaxLotBook(BasisMethod.FIFO)
    _buy(book, 10.0, 1_000_000.0)
    _buy(book, 5.0, 2_000_000.0)
    before = book.basis_krw("T")
    remainder = book.apply_split("T", Fraction(3))
    assert remainder == 0.0
    assert book.quantity("T") == pytest.approx(45.0)
    assert book.basis_krw("T") == pytest.approx(before, rel=1e-12)


def test_reverse_split_cash_in_lieu_remainder() -> None:
    """Seven shares through a 1:2 split keep 3 and return 0.5 for cash-in-lieu."""
    book = TaxLotBook(BasisMethod.FIFO)
    _buy(book, 7.0, 1_000_000.0)
    remainder = book.apply_split("T", Fraction(1, 2))
    assert remainder == pytest.approx(0.5)
    assert book.quantity("T") == pytest.approx(3.0)


def test_reverse_split_to_zero_clears_ticker() -> None:
    """One share through a 1:2 split leaves nothing and drops the ticker."""
    book = TaxLotBook(BasisMethod.FIFO)
    _buy(book, 1.0, 1_000_000.0)
    assert book.apply_split("T", Fraction(1, 2)) == pytest.approx(0.5)
    assert book.quantity("T") == 0.0
    assert book.tickers() == ()


def test_harvest_respects_budget() -> None:
    """Harvesting stops at 15 shares: 2.5M realized, the next share would exceed."""
    book = TaxLotBook(BasisMethod.FIFO)
    _buy(book, 5.0, 700_000.0, fx=1.0)
    _buy(book, 20.0, 900_000.0, fx=1.0)
    harvest = book.harvestable_quantity("T", 1_000_000.0, 1.0, 2_500_000.0)
    assert harvest == pytest.approx(15.0)
    disposal = _sell(book, harvest, harvest * 1_000_000.0, fx=1.0)
    assert disposal.gain_krw <= 2_500_000.0 + 1e-6
    assert disposal.gain_krw + 100_000.0 > 2_500_000.0


def test_fifo_harvest_stops_at_loss_lot() -> None:
    """An oldest lot under water blocks harvesting despite a profitable newer lot."""
    book = TaxLotBook(BasisMethod.FIFO)
    _buy(book, 10.0, 2_000_000.0, fx=1.0)
    _buy(book, 10.0, 500_000.0, fx=1.0)
    assert book.harvestable_quantity("T", 1_000_000.0, 1.0, 2_500_000.0) == 0.0


def test_tax_year_follows_settlement() -> None:
    """A trade in 2025 settling in 2026 is taxed in 2026."""
    book = TaxLotBook(BasisMethod.FIFO)
    book.buy("T", 10.0, 10_000.0, 1000.0, trade_session=date(2025, 12, 31), settle_session=date(2025, 12, 31))
    disposal = book.sell(
        "T",
        10.0,
        12_000.0,
        1000.0,
        trade_session=date(2025, 12, 31),
        settle_session=date(2026, 1, 2),
    )
    assert disposal.tax_year == 2026


def test_moving_average_harvest_paths() -> None:
    """Pooled harvest takes the budget-limited fraction and zero on pooled losses."""
    book = TaxLotBook(BasisMethod.MOVING_AVERAGE)
    _buy(book, 10.0, 1_000_000.0, fx=1.0)
    assert book.harvestable_quantity("T", 1_500_000.0, 1.0, 1_000_000.0) == pytest.approx(2.0)
    assert book.harvestable_quantity("T", 900_000.0, 1.0, 1_000_000.0) == 0.0
    assert book.harvestable_quantity("X", 1_500_000.0, 1.0, 1_000_000.0) == 0.0
    assert book.harvestable_quantity("T", 1_500_000.0, 1.0, 0.0) == 0.0


def test_lot_operation_boundaries() -> None:
    """Invalid quantities, costs, FX, proceeds, ratios, and tickers fail closed."""
    book = TaxLotBook(BasisMethod.FIFO)
    with pytest.raises(TaxLotError):
        book.buy("T", 0.0, 1000.0, 1000.0, trade_session=_TRADE, settle_session=_SETTLE)
    with pytest.raises(TaxLotError):
        book.buy("T", 1.0, -1000.0, 1000.0, trade_session=_TRADE, settle_session=_SETTLE)
    with pytest.raises(TaxLotError):
        book.buy("T", 1.0, 1000.0, 0.0, trade_session=_TRADE, settle_session=_SETTLE)
    with pytest.raises(TaxLotError):
        book.buy("T", float("nan"), 1000.0, 1000.0, trade_session=_TRADE, settle_session=_SETTLE)
    with pytest.raises(TaxLotError, match="unknown ticker"):
        _sell(book, 1.0, 1000.0)
    _buy(book, 2.0, 1_000_000.0)
    with pytest.raises(TaxLotError):
        book.sell("T", 1.0, float("inf"), 1000.0, trade_session=_TRADE, settle_session=_SETTLE)
    with pytest.raises(TaxLotError):
        book.sell("T", 1.0, -1000.0, 1000.0, trade_session=_TRADE, settle_session=_SETTLE)
    with pytest.raises(TaxLotError):
        book.apply_split("T", Fraction(0))
    with pytest.raises(TaxLotError):
        book.harvestable_quantity("T", 1500.0, 1000.0, float("nan"))
    with pytest.raises(TaxLotError):
        book.harvestable_quantity("T", -1500.0, 1000.0, 1000.0)
    with pytest.raises(TaxLotError):
        book.unrealized_gain_krw("T", 1500.0, 0.0)
    assert book.quantity("T") == pytest.approx(2.0)


def test_split_unknown_ticker_returns_zero() -> None:
    """Splits on an empty book are no-ops without creating phantom tickers."""
    book = TaxLotBook(BasisMethod.FIFO)
    assert book.apply_split("X", Fraction(1, 2)) == 0.0
    assert book.apply_split("X", Fraction(2)) == 0.0
    assert book.tickers() == ()
    assert book.quantity("X") == 0.0
    assert book.basis_krw("X") == 0.0
    assert book.unrealized_gain_krw("X", 100.0, 1000.0) == 0.0


def test_tickers_sorted_and_full_sale_clears() -> None:
    """Tickers list in sorted order; a full sale removes the ticker."""
    book = TaxLotBook(BasisMethod.FIFO)
    _buy(book, 2.0, 1_000_000.0)
    book.buy("A", 1.0, 1000.0, 1000.0, trade_session=_TRADE, settle_session=_SETTLE)
    assert book.tickers() == ("A", "T")
    _sell(book, 2.0, 2_000_000.0)
    assert book.tickers() == ("A",)
    assert book.quantity("T") == 0.0


@settings(max_examples=40)
@given(st.lists(st.sampled_from(["buy", "sell", "split2", "split3", "rsplit"]), min_size=1, max_size=30))
def test_lot_conservation_property(ops: list[str]) -> None:
    """Random buy/sell/split sequences never short the book or impair basis."""
    for method in (BasisMethod.FIFO, BasisMethod.MOVING_AVERAGE):
        book = TaxLotBook(method)
        for op in ops:
            day = date(2020, 1, 2)
            if op == "buy":
                book.buy("T", 4.0, 4000.0, 1000.0, trade_session=day, settle_session=day)
            elif op == "sell":
                held = book.quantity("T")
                if held > 0:
                    book.sell("T", held / 2.0, held / 2.0 * 500.0, 1000.0, trade_session=day, settle_session=day)
            elif op == "split2":
                book.apply_split("T", Fraction(2))
            elif op == "split3":
                book.apply_split("T", Fraction(3))
            else:
                book.apply_split("T", Fraction(1, 2))
            assert book.quantity("T") >= 0.0
            assert book.basis_krw("T") >= 0.0
            assert math.isfinite(book.quantity("T"))
            assert math.isfinite(book.basis_krw("T"))
        assert book.quantity("T") >= 0.0
