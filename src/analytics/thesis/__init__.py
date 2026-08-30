"""Analytics thesis package."""

from __future__ import annotations

from typing import Final

THESIS_MODULES: Final[tuple[str, ...]] = (
    "evidence",
    "meaning",
    "decision",
    "report",
    "wave",
    "structural",
    "valuation",
    "crowding",
    "purity",
    "wave_d_exit",
    "incremental",
)

__all__ = ["THESIS_MODULES"]
