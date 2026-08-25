"""Buy-only orders from consecutive allocation snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import date

    from src.sim.allocation import AllocationSnapshot

__all__ = ["BuyOrder", "ExecutionError", "orders_from_snapshots"]

_INT_LOT_TOLERANCE: Final[float] = 1e-9


class ExecutionError(RuntimeError):
    """Paper execution failed closed (sell detected or non-integer lots)."""


@dataclass(frozen=True, slots=True)
class BuyOrder:
    """One integer buy fill at an execution session."""

    session: date
    ticker: str
    quantity: int

    def __post_init__(self) -> None:
        if self.quantity < 1:
            raise ValueError(f"quantity must be >= 1, got {self.quantity!r}")


def _integer_lot(ticker: str, value: float) -> int:
    """Whole-lot view of a share value; anything beyond 1e-9 from an int fails closed."""
    nearest = round(value)
    if abs(value - nearest) > _INT_LOT_TOLERANCE:
        raise ExecutionError(f"non-integer lot for {ticker!r}: {value!r}")
    return int(nearest)


def orders_from_snapshots(snapshots: Sequence[AllocationSnapshot]) -> tuple[BuyOrder, ...]:
    """Emit buy orders from consecutive snapshot share lots; never sells.

    Raises:
        ValueError: When ``snapshots`` is empty.
        ExecutionError: When a lot is non-integer or a share count falls.
    """
    if not snapshots:
        raise ValueError("orders require at least one snapshot")
    prior_lots: dict[str, int] = {}
    orders: list[BuyOrder] = []
    for snapshot in snapshots:
        for ticker in sorted(snapshot.shares):
            lot = _integer_lot(ticker, snapshot.shares[ticker])
            delta = lot - prior_lots.get(ticker, 0)
            prior_lots[ticker] = lot
            if delta < 0:
                raise ExecutionError(f"sell detected for {ticker!r} on {snapshot.session.isoformat()}")
            if delta >= 1:
                orders.append(BuyOrder(session=snapshot.session, ticker=ticker, quantity=delta))
    return tuple(orders)
