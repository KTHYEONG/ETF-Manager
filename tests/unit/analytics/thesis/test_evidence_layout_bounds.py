"""Layout bounds for thesis evidence split."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.parametrize("scenario_id", ["test_thesis_evidence_split_under_550_lines"])
def test_thesis_evidence_split_under_550_lines(scenario_id: str) -> None:
    base = Path("tests/unit/analytics/thesis")
    candidates = sorted(base.glob("test_evidence_*.py"))
    if len(candidates) < 3:
        flat = sorted(Path("tests/unit/analytics").glob("test_thesis_evidence_*.py"))
        candidates = [p for p in flat if p.name != "test_thesis_evidence.py"]
    assert len(candidates) >= 3, f"expected >=3 evidence test modules, got {candidates}"
    for p in candidates:
        lines = p.read_text(encoding="utf-8").splitlines()
        assert len(lines) <= 550, f"{p} has {len(lines)} > 550"
    legacy = Path("tests/unit/analytics/test_thesis_evidence.py")
    if legacy.exists():
        lines = legacy.read_text(encoding="utf-8").splitlines()
        assert len(lines) <= 80, f"legacy monolith {legacy} has {len(lines)} > 80"
