"""Unit tests for integer-lot fills that recycle prior USD dust."""

from __future__ import annotations

import pytest

from src.etf_manager.sim.lots import fill_integer_buys

_FX_GROSS = 1300.0


@pytest.mark.parametrize("scenario_id", ["LOT-R-identity-and-bounds"])
def test_lot_r_identity_and_bounds(scenario_id: str) -> None:
    """LOT-R-identity-and-bounds"""
    weights = {"VTI": 1.0}

    bought, residual_usd, fees_krw = fill_integer_buys(
        cash_usd=0.0,
        sleeve_budget_krw=_FX_GROSS * 700.0,
        fx_gross=_FX_GROSS,
        weights=weights,
        prices={"VTI": 250.0},
        commission_bps=0.0,
    )
    assert bought == {"VTI": 2}
    assert residual_usd == pytest.approx(200.0)
    assert fees_krw == pytest.approx(0.0)
    assert bought["VTI"] * 250.0 + residual_usd == pytest.approx(700.0)

    bought, residual_usd, _ = fill_integer_buys(
        cash_usd=200.0,
        sleeve_budget_krw=_FX_GROSS * 700.0,
        fx_gross=_FX_GROSS,
        weights=weights,
        prices={"VTI": 250.0},
        commission_bps=0.0,
    )
    assert bought == {"VTI": 3}
    assert residual_usd == pytest.approx(150.0)

    # Commission bills the whole ticket: recycled dust plus the fresh conversion.
    bought, residual_usd, fees_krw = fill_integer_buys(
        cash_usd=100.0,
        sleeve_budget_krw=_FX_GROSS * 700.0,
        fx_gross=_FX_GROSS,
        weights=weights,
        prices={"VTI": 250.0},
        commission_bps=100.0,
    )
    assert bought == {"VTI": 3}
    assert residual_usd == pytest.approx(42.0)
    assert fees_krw == pytest.approx(8.0 * _FX_GROSS)

    with pytest.raises(ValueError, match="cash_usd"):
        fill_integer_buys(
            cash_usd=-1.0,
            sleeve_budget_krw=_FX_GROSS * 700.0,
            fx_gross=_FX_GROSS,
            weights=weights,
            prices={"VTI": 250.0},
            commission_bps=0.0,
        )
    with pytest.raises(ValueError, match="price"):
        fill_integer_buys(
            cash_usd=0.0,
            sleeve_budget_krw=_FX_GROSS * 700.0,
            fx_gross=_FX_GROSS,
            weights=weights,
            prices={"VTI": 0.0},
            commission_bps=0.0,
        )
