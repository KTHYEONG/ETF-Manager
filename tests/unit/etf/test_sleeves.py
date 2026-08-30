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
        (SleeveId.PHYSICAL_AUTOMATION, VehicleId.ROBO),
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


@pytest.mark.parametrize("scenario_id", ["test_physical_automation_vehicle_split_not_reopened"])
def test_physical_automation_vehicle_split_not_reopened(scenario_id: str) -> None:
    """test_physical_automation_vehicle_split_not_reopened"""
    import json
    from pathlib import Path

    from src.data.panel_freshness import THESIS_PANEL_TICKERS

    assert resolve_vehicle(SleeveId.PHYSICAL_AUTOMATION, VehicleRole.EXECUTION) is VehicleId.ROBO
    assert resolve_vehicle(SleeveId.PHYSICAL_AUTOMATION, VehicleRole.HISTORICAL) is VehicleId.ROBO
    sleeve_members = set(SleeveId.__members__)
    assert "INDUSTRIAL_AUTOMATION" not in sleeve_members
    assert "HUMANOID" not in sleeve_members
    assert "HUMANOID_OPTIONALITY" not in sleeve_members
    vehicle_members = set(VehicleId.__members__.values())
    assert VehicleId.BOTZ.value == "BOTZ"
    assert "HUMANOID" not in VehicleId.__members__
    assert THESIS_PANEL_TICKERS == ("BOTZ", "GRID", "PAVE", "QQQ", "ROBO", "SOXX")
    mapping = json.loads(Path("configs/etf_metadata/nport_series_map.json").read_text(encoding="utf-8"))
    assert mapping["S000054693"] == "BOTZ"
    assert mapping["S000042659"] == "ROBO"
    assert VehicleId.BOTZ in vehicle_members
    assert VehicleId.ROBO in vehicle_members


@pytest.mark.parametrize("scenario_id", ["test_robo_vehicle_id_and_sleeve_resolve"])
def test_robo_vehicle_id_and_sleeve_resolve(scenario_id: str) -> None:
    """test_robo_vehicle_id_and_sleeve_resolve"""
    assert VehicleId.ROBO.value == "ROBO"
    assert resolve_vehicle(SleeveId.PHYSICAL_AUTOMATION, VehicleRole.EXECUTION) is VehicleId.ROBO
    assert resolve_vehicle(SleeveId.PHYSICAL_AUTOMATION, VehicleRole.HISTORICAL) is VehicleId.ROBO
    assert VehicleId.ROBO in RESEARCH_SATELLITE_VEHICLES
    assert VehicleId.BOTZ in VehicleId.__members__.values()


@pytest.mark.parametrize("scenario_id", ["test_physical_automation_vehicle_reopened_to_robo"])
def test_physical_automation_vehicle_reopened_to_robo(scenario_id: str) -> None:
    """test_physical_automation_vehicle_reopened_to_robo"""
    import json
    from pathlib import Path

    from src.data.panel_freshness import THESIS_PANEL_TICKERS

    assert resolve_vehicle(SleeveId.PHYSICAL_AUTOMATION, VehicleRole.EXECUTION) is VehicleId.ROBO
    assert resolve_vehicle(SleeveId.PHYSICAL_AUTOMATION, VehicleRole.HISTORICAL) is VehicleId.ROBO
    assert "HUMANOID" not in SleeveId.__members__
    assert "INDUSTRIAL_AUTOMATION" not in SleeveId.__members__
    assert "HUMANOID" not in VehicleId.__members__
    assert THESIS_PANEL_TICKERS == ("BOTZ", "GRID", "PAVE", "QQQ", "ROBO", "SOXX")
    mapping = json.loads(Path("configs/etf_metadata/nport_series_map.json").read_text(encoding="utf-8"))
    assert mapping["S000042659"] == "ROBO"
    assert mapping["S000054693"] == "BOTZ"


@pytest.mark.parametrize("scenario_id", ["test_nport_series_map_includes_robo"])
def test_nport_series_map_includes_robo(scenario_id: str) -> None:
    """test_nport_series_map_includes_robo"""
    import json
    from pathlib import Path

    mapping = json.loads(Path("configs/etf_metadata/nport_series_map.json").read_text(encoding="utf-8"))
    assert mapping["S000042659"] == "ROBO"
    assert mapping["S000054693"] == "BOTZ"
    assert mapping["S000056509"] == "PAVE"
