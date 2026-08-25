"""Unit tests for ex-post factor attribution."""

from __future__ import annotations

import calendar as _calendar
from datetime import date

import polars as pl
import pytest

from src.etf_manager.analytics.attribution import attribute_factor_returns
from src.etf_manager.features.factors import FACTOR_COLUMNS

_ROWS = 36


def _month_ends(count: int) -> list[date]:
    ends: list[date] = []
    year, month = 2020, 1
    for _ in range(count):
        ends.append(date(year, month, _calendar.monthrange(year, month)[1]))
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return ends


def _factor_panel(rows: int, mkt_rf: list[float]) -> pl.DataFrame:
    data: dict[str, list[object]] = {"period_end": _month_ends(rows)}
    for name in FACTOR_COLUMNS:
        data[name] = mkt_rf if name == "mkt_rf" else [0.0] * rows
    return pl.DataFrame(data)


@pytest.mark.parametrize("scenario_id", ["MET-H05-attribution-r2"])
def test_met_h05_attribution_r2(scenario_id: str) -> None:
    """MET-H05-attribution-r2"""
    mkt_rf = [0.01 if index % 2 == 0 else -0.005 for index in range(_ROWS)]
    excess = pl.Series("excess", [0.5 * value for value in mkt_rf])

    result = attribute_factor_returns(excess, _factor_panel(_ROWS, mkt_rf))

    assert result.betas["mkt_rf"] == pytest.approx(0.5, abs=1e-6)
    assert result.r_squared == pytest.approx(1.0, abs=1e-6)
    assert result.alpha == pytest.approx(0.0, abs=1e-9)
    for name in ("smb", "hml", "rmw", "cma", "mom"):
        assert result.betas[name] == 0.0

    with pytest.raises(ValueError, match="36"):
        attribute_factor_returns(excess.slice(0, _ROWS - 1), _factor_panel(_ROWS - 1, mkt_rf[:-1]))
