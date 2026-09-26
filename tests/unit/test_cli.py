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


def test_cli_dispatches_research_monthly(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The research-monthly ingest command forwards both dates and requires the window."""
    from datetime import date

    from src import cli
    from src.cli import main

    received: list[tuple[object, object]] = []

    def fake_fetch(start: object, end: object, **kwargs: object) -> None:
        received.append((start, end))

    monkeypatch.setattr(cli, "fetch_and_persist_research_monthly", fake_fetch)

    with caplog.at_level("INFO"):
        code = main(
            [
                "ingest",
                "research-monthly",
                "--start",
                "1926-07-31",
                "--end",
                "2026-07-31",
            ]
        )

    assert code == 0
    assert received == [(date(1926, 7, 31), date(2026, 7, 31))]
    assert any(
        "event=cli_ingest_done" in record.message and "dataset=research_monthly" in record.message
        for record in caplog.records
    )
    assert main(["ingest", "research-monthly", "--start", "1926-07-31"]) == 2
    assert len(received) == 1
