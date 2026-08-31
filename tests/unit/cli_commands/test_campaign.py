"""Ensure co-modification wiring for campaign."""
from __future__ import annotations

import pytest

@pytest.mark.parametrize("scenario_id", ["test_campaign_import"])
def test_campaign_import(scenario_id: str) -> None:
    import src.cli_commands.campaign as mod
    assert hasattr(mod, "run_final_historical_campaign_command")
    assert hasattr(mod, "run_accumulation_cohort_command")
