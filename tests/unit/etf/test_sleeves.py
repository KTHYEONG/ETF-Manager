"""Unit tests for economic sleeve to vehicle resolution."""

from __future__ import annotations

import pytest

from src.etf.sleeves import RESEARCH_SATELLITE_VEHICLES, SleeveId, VehicleId, VehicleRole, resolve_vehicle


@pytest.mark.parametrize("scenario_id", ["SLEEVE-01-nasdaq-execution-qqq"])
def test_sleeve_01_nasdaq_execution_qqq(scenario_id: str) -> None:
    """SLEEVE-01-nasdaq-execution-qqq"""
    assert resolve_vehicle(SleeveId.NASDAQ_100, VehicleRole.EXECUTION) is VehicleId.QQQ
    assert resolve_vehicle(SleeveId.NASDAQ_100, VehicleRole.HISTORICAL) is VehicleId.QQQ
    assert VehicleId.QQQM is VehicleId.QQQM
    assert resolve_vehicle(SleeveId.NASDAQ_100, VehicleRole.EXECUTION) is not VehicleId.QQQM
    assert resolve_vehicle(SleeveId.NASDAQ_100, VehicleRole.HISTORICAL) is not VehicleId.QQQM


@pytest.mark.parametrize("scenario_id", ["SLEEVE-02-us-buckets-and-satellites"])
def test_sleeve_02_us_buckets_and_satellites(scenario_id: str) -> None:
    """SLEEVE-02-us-buckets-and-satellites"""
    cases = (
        (SleeveId.US_TOTAL_MARKET, VehicleId.VTI),
        (SleeveId.US_LARGE_CAP, VehicleId.IVV),
        (SleeveId.AI_SEMICONDUCTOR, VehicleId.SOXX),
        (SleeveId.AI_POWER_EQUIPMENT, VehicleId.PAVE),
        (SleeveId.PHYSICAL_AUTOMATION, VehicleId.BOTZ),
    )
    for sleeve, expected in cases:
        assert resolve_vehicle(sleeve, VehicleRole.EXECUTION) is expected
        assert resolve_vehicle(sleeve, VehicleRole.HISTORICAL) is expected
    assert SleeveId("us_nasdaq_100") is SleeveId.NASDAQ_100


@pytest.mark.parametrize("scenario_id", ["SLEEVE-03-unknown-sleeve-fails"])
def test_sleeve_03_unknown_sleeve_fails(scenario_id: str) -> None:
    """SLEEVE-03-unknown-sleeve-fails"""
    with pytest.raises(ValueError, match="not_a_sleeve"):
        SleeveId("not_a_sleeve")
    with pytest.raises(ValueError, match="unknown sleeve"):
        resolve_vehicle("QQQ")  # type: ignore[arg-type]


@pytest.mark.parametrize("scenario_id", ["test_pave_vehicle_id_and_sleeve_resolve"])
def test_pave_vehicle_id_and_sleeve_resolve(scenario_id: str) -> None:
    """test_pave_vehicle_id_and_sleeve_resolve"""
    assert VehicleId.PAVE.value == "PAVE"
    assert resolve_vehicle(SleeveId.AI_POWER_EQUIPMENT, VehicleRole.EXECUTION) is VehicleId.PAVE
    assert resolve_vehicle(SleeveId.AI_POWER_EQUIPMENT, VehicleRole.HISTORICAL) is VehicleId.PAVE
    assert VehicleId.PAVE in RESEARCH_SATELLITE_VEHICLES
    assert VehicleId.GRID in RESEARCH_SATELLITE_VEHICLES
