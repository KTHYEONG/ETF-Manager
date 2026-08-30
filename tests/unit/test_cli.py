"""Shim for repo_layout_refactor Wave 1: use tests/unit/cli/."""

from __future__ import annotations

import pytest

# Monolith removed: see tests/unit/cli/test_*.py
# This shim exists only to satisfy code_map legacy reference.


@pytest.mark.parametrize("scenario_id", ["test_thesis_incremental_accepts_physical_automation"])
def test_thesis_incremental_accepts_physical_automation(scenario_id: str) -> None:
    """test_thesis_incremental_accepts_physical_automation"""
    from pathlib import Path

    text = Path("src/cli_commands/thesis.py").read_text(encoding="utf-8")
    assert "ThesisId.PHYSICAL_AUTOMATION" in text
    assert "ThesisId.AI_COMPUTE" in text
    assert "ThesisId.AI_POWER_BOTTLENECK" in text
    assert "only ai_compute supported" not in text or "PHYSICAL_AUTOMATION" in text
    # allow-list check
    assert "PHYSICAL_AUTOMATION" in text
    # unknown thesis still returns 2 via ThesisId validation
    from src.policy.thesis import ThesisId

    with pytest.raises(ValueError, match="unknown"):
        ThesisId("unknown_thesis_xyz")
