"""Ensure co-modification wiring for cli."""
from __future__ import annotations

import pytest

@pytest.mark.parametrize("scenario_id", ["test_cli_import"])
def test_cli_import(scenario_id: str) -> None:
    import src.cli as mod
    assert hasattr(mod, "main")
