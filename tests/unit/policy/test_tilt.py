"""Unit tests for fixed long-only factor tilt."""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl
import pytest

from src.etf_manager.data.pit import AVAILABLE_AT
from src.etf_manager.policy.targets import PolicyError, PolicyId, resolve_targets
from src.etf_manager.policy.tilt import FactorTilt, apply_fixed_tilt, resolve_tilted_targets

_SIGNAL_AT = datetime(2024, 6, 3, 21, tzinfo=UTC)
_BASE = {"VTI": 0.5, "VEA": 0.3, "VWO": 0.2}


def _loadings(hml_by_sleeve: dict[str, float]) -> dict[str, dict[str, float]]:
    return {
        sleeve: {"mkt_rf": 1.0, "smb": 0.0, "hml": hml, "rmw": 0.0, "cma": 0.0, "mom": 0.0}
        for sleeve, hml in hml_by_sleeve.items()
    }


def _prices_frame() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "ticker": pl.String,
            "date": pl.Date,
            "adjusted_close": pl.Float64,
            AVAILABLE_AT: pl.Datetime("us", "UTC"),
        }
    )


@pytest.mark.parametrize("scenario_id", ["POL-H03-tilt-simplex"])
def test_pol_h03_tilt_simplex(scenario_id: str) -> None:
    """POL-H03-tilt-simplex"""
    loadings = _loadings({"VTI": 1.0, "VEA": 0.0, "VWO": -1.0})
    tilted = apply_fixed_tilt(_BASE, loadings, FactorTilt(factor="hml", intensity=0.1))

    assert tilted["VTI"] > 0.5
    assert tilted["VWO"] < 0.2
    assert min(tilted.values()) >= 0.0
    assert abs(sum(tilted.values()) - 1.0) <= 1e-6
    assert tilted["VTI"] == pytest.approx(0.55)
    assert tilted["VWO"] == pytest.approx(0.15)


@pytest.mark.parametrize("scenario_id", ["POL-H03-tilt-simplex"])
def test_pol_h03_none_tilt_is_identity(scenario_id: str) -> None:
    """POL-H03-tilt-simplex"""
    prices = _prices_frame()
    assert resolve_tilted_targets(
        PolicyId.S2_REGIONAL, prices, prices, _SIGNAL_AT, None
    ) == resolve_targets(PolicyId.S2_REGIONAL, prices, _SIGNAL_AT)


@pytest.mark.parametrize("scenario_id", ["POL-H03-tilt-simplex"])
def test_pol_h03_degenerate_dispersion_fails_closed(scenario_id: str) -> None:
    """POL-H03-tilt-simplex"""
    loadings = _loadings({"VTI": 0.5, "VEA": 0.5, "VWO": 0.5})
    with pytest.raises(PolicyError):
        apply_fixed_tilt(_BASE, loadings, FactorTilt(factor="hml", intensity=0.1))


@pytest.mark.parametrize("scenario_id", ["POL-H03-tilt-simplex"])
def test_pol_h03_tilt_validation(scenario_id: str) -> None:
    """POL-H03-tilt-simplex"""
    with pytest.raises(ValueError, match="tilt factor"):
        FactorTilt(factor="mkt_rf", intensity=0.1)
    with pytest.raises(ValueError, match="intensity"):
        FactorTilt(factor="hml", intensity=0.0)
    with pytest.raises(ValueError, match="intensity"):
        FactorTilt(factor="hml", intensity=0.26)
