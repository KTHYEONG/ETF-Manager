"""Economic sleeve to listed vehicle resolution (research identity)."""

from __future__ import annotations

from enum import StrEnum
from typing import Final

__all__ = [
    "RESEARCH_SATELLITE_VEHICLES",
    "SleeveId",
    "VehicleId",
    "VehicleRole",
    "resolve_vehicle",
]


class SleeveId(StrEnum):
    """Economic exposure sleeves; three US buckets mirror UsEquityUniverse values."""

    US_TOTAL_MARKET = "us_total_market"
    US_LARGE_CAP = "us_large_cap"
    NASDAQ_100 = "us_nasdaq_100"
    AI_SEMICONDUCTOR = "ai_semiconductor"
    AI_POWER_EQUIPMENT = "ai_power_equipment"
    PHYSICAL_AUTOMATION = "physical_automation"


class VehicleId(StrEnum):
    """Listed implementation vehicles; includes QQQM as VehicleId only."""

    QQQ = "QQQ"
    QQQM = "QQQM"
    VTI = "VTI"
    IVV = "IVV"
    SOXX = "SOXX"
    GRID = "GRID"
    BOTZ = "BOTZ"
    IBB = "IBB"
    ITA = "ITA"
    IWF = "IWF"
    XLI = "XLI"
    PAVE = "PAVE"


class VehicleRole(StrEnum):
    """Vehicle use; Wave 0 maps both roles to the same ticker."""

    EXECUTION = "execution"
    HISTORICAL = "historical"


_SLEEVE_VEHICLE: Final[dict[SleeveId, VehicleId]] = {
    SleeveId.NASDAQ_100: VehicleId.QQQ,
    SleeveId.US_TOTAL_MARKET: VehicleId.VTI,
    SleeveId.US_LARGE_CAP: VehicleId.IVV,
    SleeveId.AI_SEMICONDUCTOR: VehicleId.SOXX,
    SleeveId.AI_POWER_EQUIPMENT: VehicleId.PAVE,
    SleeveId.PHYSICAL_AUTOMATION: VehicleId.BOTZ,
}


def resolve_vehicle(sleeve: SleeveId, role: VehicleRole = VehicleRole.EXECUTION) -> VehicleId:  # noqa: ARG001
    """Resolve an economic sleeve to its listed vehicle for the given role.

    Wave 0 maps both EXECUTION and HISTORICAL to the same vehicle; QQQM is
    never returned by this resolver.
    """
    try:
        return _SLEEVE_VEHICLE[sleeve]
    except KeyError as exc:
        raise ValueError(f"unknown sleeve {sleeve!r}") from exc


RESEARCH_SATELLITE_VEHICLES: Final[tuple[VehicleId, ...]] = tuple(
    sorted(
        (
            VehicleId.BOTZ,
            VehicleId.GRID,
            VehicleId.PAVE,
            VehicleId.IBB,
            VehicleId.ITA,
            VehicleId.IWF,
            VehicleId.SOXX,
            VehicleId.XLI,
        ),
        key=lambda v: v.value,
    )
)
