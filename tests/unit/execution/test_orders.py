"""Unit tests for buy-only order generation."""

from __future__ import annotations

from datetime import date

import pytest

from src.execution.orders import BuyOrder, ExecutionError, orders_from_snapshots
from src.sim.allocation import AllocationSnapshot


def _snapshot(session: date, shares: dict[str, float]) -> AllocationSnapshot:
    return AllocationSnapshot(
        session=session,
        cash_krw=0.0,
        cash_usd=0.0,
        shares=shares,
        mark_krw=0.0,
        contribution_krw=1_000_000.0,
        fees_krw=0.0,
    )


@pytest.mark.parametrize("scenario_id", ["EXE-X01-buy-only-deltas"])
def test_exe_x01_buy_only_deltas(scenario_id: str) -> None:
    """EXE-X01-buy-only-deltas"""
    d1 = date(2024, 1, 2)
    d2 = date(2024, 2, 1)
    d3 = date(2024, 3, 1)

    orders = orders_from_snapshots((_snapshot(d1, {}), _snapshot(d2, {"VTI": 10})))
    assert orders == (BuyOrder(session=d2, ticker="VTI", quantity=10),)

    falling = (_snapshot(d1, {}), _snapshot(d2, {"VTI": 10}), _snapshot(d3, {"VTI": 8}))
    with pytest.raises(ExecutionError):
        orders_from_snapshots(falling)

    with pytest.raises(ValueError, match="quantity"):
        BuyOrder(session=d2, ticker="VTI", quantity=0)

    with pytest.raises(ValueError, match="snapshot"):
        orders_from_snapshots(())
