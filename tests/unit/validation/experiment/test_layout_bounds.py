"""Layout bounds for validation experiment split."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.parametrize("scenario_id", ["test_experiment_split_files_under_550_lines"])
def test_experiment_split_files_under_550_lines(scenario_id: str) -> None:
    base = Path("tests/unit/validation/experiment")
    files = sorted(base.glob("test_*.py"))
    assert len(files) >= 4, f"expected at least 4 test files, got {len(files)}: {files}"
    for p in files:
        lines = p.read_text(encoding="utf-8").splitlines()
        assert len(lines) <= 550, f"{p} has {len(lines)} > 550"
    legacy = Path("tests/unit/validation/test_experiment.py")
    if legacy.exists():
        lines = legacy.read_text(encoding="utf-8").splitlines()
        assert len(lines) <= 80, f"legacy monolith {legacy} has {len(lines)} > 80"
