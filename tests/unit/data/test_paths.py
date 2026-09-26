"""Result path helpers."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from src.data.paths import (
    RepositoryPaths,
    legacy_results_dir,
    resolve_repo_path,
    resolve_repository_paths,
    results_root,
)
from src.data.settings import DataSettings
from src.validation.experiment import load_experiment_config, resolve_experiment_config_path


def test_results_root_under_data_root(tmp_path: Path) -> None:
    settings = DataSettings(data_root=tmp_path / "data")
    assert results_root(settings) == tmp_path / "data" / "runs"
    assert legacy_results_dir(settings) == tmp_path / "data" / "runs" / "_legacy"
    assert not (tmp_path / "data" / "runs").exists()
    assert not (tmp_path / "data" / "runs" / "_legacy").exists()


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
    assert from_repo.experiments == repo_root / "configs" / "research"
    assert from_repo.decisions == repo_root / "configs" / "decision"
    assert from_repo.results == tmp_path / "data" / "runs"
    assert from_repo.frozen == tmp_path / "data" / "frozen"
    assert from_repo.research == tmp_path / "data" / "research"
    monkeypatch.chdir(tmp_path)
    from_tmp = resolve_repository_paths(settings)
    assert from_tmp == from_repo


def test_resolve_repository_paths_results_separated_from_inputs(tmp_path: Path) -> None:
    """Generated results can never land under versioned inputs or frozen records."""
    settings = DataSettings(data_root=tmp_path / "data")
    paths = resolve_repository_paths(settings)
    assert paths.results == paths.data / "runs"
    assert paths.frozen == paths.data / "frozen"
    assert paths.research == paths.data / "research"
    assert paths.results != paths.experiments
    assert paths.results != paths.decisions
    assert paths.results.relative_to(paths.data)
    with pytest.raises(ValueError, match="research"):
        resolve_repository_paths(DataSettings(data_root="configs/research"))
    with pytest.raises(ValueError, match="decision"):
        resolve_repository_paths(DataSettings(data_root="configs/decision"))


def test_resolve_repository_paths_custom_data_root() -> None:
    """Every generated root sits under a custom data root."""
    paths = resolve_repository_paths(DataSettings(data_root="data_alt"))
    assert paths.results == paths.data / "runs"
    assert paths.frozen == paths.data / "frozen"
    assert paths.research == paths.data / "research"
    assert paths.data.name == "data_alt"


def test_load_experiment_config_legacy_alias_stable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Historical configs/experiments references resolve to versioned definitions without a copy."""
    settings = DataSettings()
    legacy_ref = "configs/experiments/m_thesis_ai_compute_soxx.json"
    current_ref = "experiments/m_thesis_ai_compute_soxx.json"
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


def _write(tmp_path: Path, relative: str, content: str = "{}") -> Path:
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_resolve_repo_path_existing_file_wins_without_alias_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A file present at the cited path is returned as-is and logs no alias event."""
    monkeypatch.chdir(tmp_path)
    _write(tmp_path, "experiments/kept.json")
    _write(tmp_path, "configs/research/kept.json")
    with caplog.at_level(logging.INFO, logger="src.data.paths"):
        resolved = resolve_repo_path("experiments/kept.json")
    assert resolved == (tmp_path / "experiments" / "kept.json").resolve()
    assert "repo_path_alias" not in caplog.text


def test_resolve_repo_path_root_experiments_alias(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Frozen experiments/ citations map to the versioned research definitions."""
    monkeypatch.chdir(tmp_path)
    expected = _write(tmp_path, "configs/research/wf_vti_qqq.json")
    assert resolve_repo_path("experiments/wf_vti_qqq.json") == expected.resolve()


def test_resolve_repo_path_legacy_configs_alias(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Historical configs/experiments/ citations map to the same research definitions."""
    monkeypatch.chdir(tmp_path)
    expected = _write(tmp_path, "configs/research/wf_vti_qqq.json")
    assert resolve_repo_path("configs/experiments/wf_vti_qqq.json") == expected.resolve()


def test_resolve_repo_path_archived_config_alias(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Superseded experiment citations fall through to the research archive."""
    monkeypatch.chdir(tmp_path)
    expected = _write(tmp_path, "configs/research/archive/m_qqq_grid.json")
    assert resolve_repo_path("experiments/m_qqq_grid.json") == expected.resolve()
    assert resolve_repo_path("configs/experiments/m_qqq_grid.json") == expected.resolve()


@pytest.mark.parametrize(
    ("cited", "target"),
    [
        ("experiments/pension_decision_v3.json", "configs/decision/pension.json"),
        ("experiments/final_historical_campaign_v1.json", "configs/decision/general.json"),
        ("experiments/isa_household_v1.json", "configs/decision/isa.json"),
        ("configs/experiments/pension_decision_v3.json", "configs/decision/pension.json"),
    ],
)
def test_resolve_repo_path_decision_rename_alias(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cited: str, target: str
) -> None:
    """Frozen decision citations map to their renamed decision specs."""
    monkeypatch.chdir(tmp_path)
    expected = _write(tmp_path, target)
    assert resolve_repo_path(cited) == expected.resolve()


@pytest.mark.parametrize(
    ("cited", "target"),
    [
        ("records/pension_decisions/record.json", "data/frozen/pension/record.json"),
        ("records/isa_household/record.json", "data/frozen/isa/record.json"),
        ("records/prospective/record.json", "data/frozen/prospective/record.json"),
    ],
)
def test_resolve_repo_path_records_alias(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cited: str, target: str
) -> None:
    """Frozen records/ citations map into the data-root-relative frozen records."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ETF_MANAGER_DATA_ROOT", raising=False)
    expected = _write(tmp_path, target)
    assert resolve_repo_path(cited) == expected.resolve()


def test_resolve_repo_path_unresolvable_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A citation with no existing candidate fails closed."""
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError):
        resolve_repo_path("experiments/does_not_exist_zz.json")


def test_resolve_repo_path_directory_is_not_a_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Directories never resolve, neither as-given nor as alias targets."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "somedir").mkdir()
    with pytest.raises(FileNotFoundError):
        resolve_repo_path("somedir")
    (tmp_path / "configs" / "research" / "adir.json").mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        resolve_repo_path("experiments/adir.json")


def test_resolve_repository_paths_versioned_inside_generated_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A repository root nested under the data root fails closed naming the directory."""
    import src.data.paths as paths_mod

    nested = tmp_path / "data" / "runs" / "nested"
    nested.mkdir(parents=True)
    monkeypatch.setattr(paths_mod, "_repository_root", lambda: nested)
    with pytest.raises(ValueError, match="experiments"):
        resolve_repository_paths(DataSettings(data_root=tmp_path / "data"))


def test_resolve_repo_path_alias_anchored_at_repository_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Alias candidates are also tried anchored at the repository root."""
    import src.data.paths as paths_mod

    fake_root = tmp_path / "fake_root"
    expected = _write(fake_root, "configs/research/anchored.json")
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    monkeypatch.setattr(paths_mod, "_repository_root", lambda: fake_root)
    assert resolve_repo_path("experiments/anchored.json") == expected.resolve()
