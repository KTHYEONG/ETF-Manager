"""Unit tests for PaperBroker and reconcile."""

from __future__ import annotations

from datetime import date

import pytest

from src.etf_manager.execution.broker import PaperBroker, reconcile
from src.etf_manager.execution.orders import BuyOrder, ExecutionError
from src.etf_manager.sim.allocation import AllocationSnapshot


def _snapshot(shares: dict[str, float], session: date) -> AllocationSnapshot:
    return AllocationSnapshot(
        session=session,
        cash_krw=0.0,
        cash_usd=0.0,
        shares=shares,
        mark_krw=0.0,
        contribution_krw=1_000_000.0,
        fees_krw=0.0,
    )


@pytest.mark.parametrize("scenario_id", ["EXE-X02-paper-reconcile"])
def test_exe_x02_paper_reconcile(scenario_id: str) -> None:
    """EXE-X02-paper-reconcile"""
    session = date(2024, 1, 2)
    book = PaperBroker()
    book.submit_buy(BuyOrder(session=session, ticker="VT", quantity=3))
    assert book.position("VT") == 3
    assert book.position("VTI") == 0

    reconcile(book, _snapshot({"VT": 3}, session))

    with pytest.raises(ExecutionError):
        reconcile(book, _snapshot({"VT": 2}, session))
