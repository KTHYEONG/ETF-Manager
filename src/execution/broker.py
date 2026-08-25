"""In-memory paper broker and position reconciliation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, Protocol

from src.execution.orders import ExecutionError, orders_from_snapshots

if TYPE_CHECKING:
    from src.execution.orders import BuyOrder
    from src.sim.allocation import AllocationResult, AllocationSnapshot

__all__ = ["Broker", "PaperBroker", "reconcile", "replay_paper"]

_INT_LOT_TOLERANCE: Final[float] = 1e-9


class Broker(Protocol):
    """Integer-lot book used by paper (and later live) execution."""

    def position(self, ticker: str) -> int:
        """Held shares of ``ticker``; unknown names are zero."""

    def submit_buy(self, order: BuyOrder) -> None:
        """Increase the lot for ``order.ticker`` by ``order.quantity``."""

    def tickers(self) -> frozenset[str]:
        """Tickers currently held with a nonzero lot."""


class PaperBroker:
    """In-memory buy-only book; never decreases a position."""

    def __init__(self) -> None:
        self._lots: dict[str, int] = {}

    def position(self, ticker: str) -> int:
        return self._lots.get(ticker, 0)

    def submit_buy(self, order: BuyOrder) -> None:
        self._lots[order.ticker] = self._lots.get(order.ticker, 0) + order.quantity

    def tickers(self) -> frozenset[str]:
        return frozenset(self._lots)


def reconcile(broker: Broker, snapshot: AllocationSnapshot) -> None:
    """Fail closed when integer lots disagree with ``snapshot.shares``.

    Compares the union of ledger share keys and held tickers so extras with
    nonzero quantity also fail.

    Raises:
        ExecutionError: On any ticker quantity mismatch.
    """
    mismatches: list[str] = []
    for ticker in sorted(set(snapshot.shares) | broker.tickers()):
        expected = _ledger_lot(ticker, snapshot.shares.get(ticker, 0.0))
        held = broker.position(ticker)
        if held != expected:
            mismatches.append(f"{ticker}: book={held} ledger={expected}")
    if mismatches:
        raise ExecutionError("; ".join(mismatches))


def replay_paper(result: AllocationResult) -> PaperBroker:
    """Submit every derived buy onto a fresh paper book and reconcile the last snapshot.

    Raises:
        ValueError: When the path has no snapshots.
        ExecutionError: When orders or reconcile fail closed.
    """
    book = PaperBroker()
    for order in orders_from_snapshots(result.snapshots):
        book.submit_buy(order)
    reconcile(book, result.snapshots[-1])
    return book


def _ledger_lot(ticker: str, value: float) -> int:
    nearest = round(value)
    if abs(value - nearest) > _INT_LOT_TOLERANCE:
        raise ExecutionError(f"non-integer lot for {ticker!r}: {value!r}")
    return int(nearest)
