"""Result path helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.data.paths import (
    RepositoryPaths,
    legacy_results_dir,
    resolve_repository_paths,
    results_root,
)
from src.data.settings import DataSettings
from src.validation.experiment import load_experiment_config, resolve_experiment_config_path


def test_results_root_under_data_root(tmp_path: Path) -> None:
    settings = DataSettings(data_root=tmp_path / "data")
    assert results_root(settings) == tmp_path / "data" / "results"
    assert legacy_results_dir(settings) == tmp_path / "data" / "results" / "_legacy"
    assert not (tmp_path / "data" / "results").exists()
    assert not (tmp_path / "data" / "results" / "_legacy").exists()


def test_resolve_repository_paths_working_directory_independent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same repository settings resolve identical canonical targets from any process directory."""
    repo_root = Path(__file__).resolve().parents[3]
    settings = DataSettings(data_root=tmp_path / "data")
    monkeypatch.chdir(repo_root)
    from_repo = resolve_repository_paths(settings)
    assert isinstance(from_repo, RepositoryPaths)
    assert from_repo.root == repo_root
    assert from_repo.experiments == repo_root / "experiments"
    assert from_repo.prospective_records == repo_root / "records" / "prospective"
    monkeypatch.chdir(tmp_path)
    from_tmp = resolve_repository_paths(settings)
    assert from_tmp == from_repo


def test_resolve_repository_paths_results_separated_from_inputs(tmp_path: Path) -> None:
    """Generated results can never land under versioned inputs or frozen records."""
    settings = DataSettings(data_root=tmp_path / "data")
    paths = resolve_repository_paths(settings)
    assert paths.results == paths.data / "results"
    assert paths.results != paths.experiments
    assert paths.results != paths.prospective_records
    assert paths.results.relative_to(paths.data)
    with pytest.raises(ValueError, match="experiments"):
        resolve_repository_paths(DataSettings(data_root="experiments"))
    with pytest.raises(ValueError, match="records/prospective"):
        resolve_repository_paths(DataSettings(data_root="records/prospective"))


def test_load_experiment_config_legacy_alias_stable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Historical configs/experiments references resolve to versioned definitions without a copy."""
    settings = DataSettings()
    legacy_ref = "configs/experiments/acc_qqq_baseline_120m.json"
    current_ref = "experiments/acc_qqq_baseline_120m.json"
    assert not Path(legacy_ref).exists()
    assert resolve_experiment_config_path(legacy_ref, settings=settings) == resolve_experiment_config_path(
        current_ref, settings=settings
    )
    assert load_experiment_config(legacy_ref, settings=settings) == load_experiment_config(
        current_ref, settings=settings
    )
    assert not Path(legacy_ref).exists()
    monkeypatch.chdir(tmp_path)
    assert resolve_experiment_config_path(legacy_ref, settings=settings) == resolve_experiment_config_path(
        current_ref, settings=settings
    )
    assert not (tmp_path / legacy_ref).exists()
