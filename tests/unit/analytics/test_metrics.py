"""Unit tests for accumulation performance metrics."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from src.analytics.metrics import XirrError, max_drawdown, real_krw, xirr


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


def test_met_f01_real_krw() -> None:
    """MET-F01-real-krw"""
    assert real_krw(1300.0, cpi_index=130.0, cpi_base=100.0) == pytest.approx(1000.0)
    assert real_krw(50.0, cpi_index=100.0, cpi_base=100.0) == 50.0

    with pytest.raises(ValueError, match="positive"):
        real_krw(50.0, cpi_index=0.0, cpi_base=100.0)
    with pytest.raises(ValueError, match="positive"):
        real_krw(50.0, cpi_index=130.0, cpi_base=-1.0)
