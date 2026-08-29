"""Result path helpers."""

from __future__ import annotations

from pathlib import Path

from src.data.paths import audits_dir, experiments_dir, thesis_reports_dir
from src.data.settings import DataSettings


def test_result_paths_resolve_under_data_results(tmp_path: Path) -> None:
    settings = DataSettings(data_root=tmp_path / "data")
    root = settings.resolved_data_root()

    assert experiments_dir(settings) == root / "results" / "experiments"
    assert audits_dir(settings) == root / "results" / "audits"
    assert thesis_reports_dir(settings) == root / "results" / "thesis"

    assert str(experiments_dir(settings)).endswith("data/results/experiments")
    assert str(audits_dir(settings)).endswith("data/results/audits")
    assert str(thesis_reports_dir(settings)).endswith("data/results/thesis")
