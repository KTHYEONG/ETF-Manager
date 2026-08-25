"""Unit tests for the repository-local data-root settings boundary."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.data.settings import DataSettings


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
