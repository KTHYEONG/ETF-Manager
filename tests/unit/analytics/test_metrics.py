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


def test_xirr_converges_on_multi_billion_krw_paths() -> None:
    """Float resolution at 1e9-scale cashflows exceeds an absolute 1e-6 bound; tolerance must scale."""
    from datetime import UTC, datetime, timedelta

    start = datetime(2006, 10, 2, tzinfo=UTC)
    monthly = 20_000_000.0
    flows = [(start + timedelta(days=30 * month), -monthly) for month in range(240)]
    terminal = monthly * 240 * 9.4
    flows.append((flows[-1][0], terminal))

    rate = xirr(flows)

    assert 0.05 < rate < 0.60
    npv = sum(amount * (1.0 + rate) ** (-((when - start).total_seconds() / (365.25 * 86400.0))) for when, amount in flows)
    assert abs(npv) < 1e-9 * monthly * 240 * 20


def test_xirr_reports_non_convergence_for_unidentifiable_rate() -> None:
    """A payoff so small that the implied rate is essentially -100% must fail closed, not return garbage."""
    from datetime import UTC, datetime, timedelta

    start = datetime(2020, 1, 1, tzinfo=UTC)

    with pytest.raises(XirrError, match="did not converge"):
        xirr([(start, -100.0), (start + timedelta(days=365), 1e-9)])
