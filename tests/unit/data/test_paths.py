"""Result path helpers."""

from __future__ import annotations

from pathlib import Path

from src.data.paths import legacy_results_dir, results_root
from src.data.settings import DataSettings


def test_results_root_under_data_root(tmp_path: Path) -> None:
    settings = DataSettings(data_root=tmp_path / "data")
    assert results_root(settings) == tmp_path / "data" / "results"
    assert legacy_results_dir(settings) == tmp_path / "data" / "results" / "_legacy"
    assert not (tmp_path / "data" / "results").exists()
    assert not (tmp_path / "data" / "results" / "_legacy").exists()
