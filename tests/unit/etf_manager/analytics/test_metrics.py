"""Unit tests for accumulation performance metrics."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from src.etf_manager.analytics.metrics import XirrError, max_drawdown, xirr


def test_met_d06_xirr_and_mdd() -> None:
    """MET-D06-xirr-and-mdd"""
    t0 = datetime(2024, 1, 31, 21, 0, tzinfo=UTC)
    one_year = t0 + timedelta(days=365.25)

    assert xirr([(t0, -100.0), (one_year, 110.0)]) == pytest.approx(0.10, abs=1e-6)
    assert max_drawdown([100.0, 120.0, 90.0, 90.0]) == pytest.approx(-0.25, abs=1e-12)
    assert max_drawdown([7.0]) == 0.0

    with pytest.raises(ValueError, match="non-empty"):
        max_drawdown([])
    with pytest.raises(XirrError):
        xirr([(t0, 100.0), (one_year, 150.0)])
