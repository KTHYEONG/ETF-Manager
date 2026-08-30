"""CLI command modules package."""

from __future__ import annotations

from typing import Final

CLI_COMMAND_MODULES: Final[tuple[str, ...]] = (
    "parser",
    "resolvers",
    "ingest",
    "thesis",
    "diagnose",
    "sim_run",
    "campaign",
)
