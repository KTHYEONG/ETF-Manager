"""Unit tests for the repository-local data-root settings boundary."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from src.data.settings import DataSettings
from src.validation.experiment import load_experiment_config


@pytest.mark.parametrize("scenario_id", ["ST-B06-settings-boundary"])
def test_settings_boundary(scenario_id: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """ST-B06-settings-boundary"""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ETF_MANAGER_DATA_ROOT", raising=False)

    default_settings = DataSettings()
    default_root = default_settings.resolved_data_root()
    assert default_root == tmp_path / "data"

    monkeypatch.setenv("ETF_MANAGER_DATA_ROOT", "inside")
    inside_root = DataSettings().resolved_data_root()
    assert inside_root == tmp_path / "inside"

    sibling = tmp_path.parent / "sibling-outside"
    with pytest.raises(ValueError, match="outside cwd"):
        DataSettings(data_root=sibling).resolved_data_root()

    with pytest.raises(ValueError, match="outside cwd"):
        DataSettings(data_root="../outside").resolved_data_root()

    assert default_root.exists() is False
    assert inside_root.exists() is False


def test_data_settings_invalid_root_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Out-of-repository data roots fail closed while naming the unsafe root."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ETF_MANAGER_DATA_ROOT", raising=False)
    unsafe = tmp_path.parent / "sibling-outside-settings-probe"
    with pytest.raises(ValueError, match=re.escape("sibling-outside-settings-probe")):
        DataSettings(data_root=unsafe).resolved_data_root()
    with pytest.raises(ValueError, match=re.escape("../outside")):
        DataSettings(data_root="../outside").resolved_data_root()


def test_data_settings_frozen_campaign_value_wins() -> None:
    """Explicit frozen campaign values override operational defaults; report identity is stable."""
    spec_from_default_root = load_experiment_config(
        "experiments/acc_qqq_baseline_120m.json", settings=DataSettings(data_root="data")
    )
    spec_from_alt_root = load_experiment_config(
        "experiments/acc_qqq_baseline_120m.json", settings=DataSettings(data_root="data_alt")
    )
    assert spec_from_default_root == spec_from_alt_root
    assert spec_from_default_root.contribution_krw == spec_from_alt_root.contribution_krw
