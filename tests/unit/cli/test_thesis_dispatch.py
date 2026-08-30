"""Thesis dispatch tests."""

from __future__ import annotations

import pytest

from src import cli
from src.cli import main


@pytest.mark.parametrize("scenario_id", ["CLI-THESIS-01-dispatch-no-gate"])
def test_cli_thesis_01_dispatch_no_gate(scenario_id: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """test_cli_thesis_01_dispatch_no_gate"""
    captured: dict[str, object] = {}

    def fake_thesis(**kwargs: object) -> int:
        captured.update(kwargs)
        return 0

    def forbidden_adoption(*args: object, **kwargs: object) -> bool:
        raise AssertionError("thesis inspect must never call adoption_passes")

    monkeypatch.setattr(cli, "run_thesis_command", fake_thesis)
    monkeypatch.setattr(cli, "adoption_passes", forbidden_adoption, raising=False)
    # also patch thesis module for direct call
    import src.cli_commands.thesis as thesis_mod

    monkeypatch.setattr(thesis_mod, "adoption_passes", forbidden_adoption, raising=False)

    assert main(["run", "thesis"]) == 0
    assert captured["config_dir"] == "configs/theses"
    assert captured["thesis_id"] is None


@pytest.mark.parametrize("scenario_id", ["CLI-THESIS-WAVE-dispatch"])
def test_cli_thesis_wave_dispatch(scenario_id: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """test_cli_thesis_wave_dispatch"""
    captured: dict[str, object] = {}

    def fake_wave(**kwargs: object) -> int:
        captured.update(kwargs)
        return 0

    def forbidden(*args: object, **kwargs: object):
        raise AssertionError("must never call adoption_passes")

    monkeypatch.setattr(cli, "run_thesis_wave_command", fake_wave)
    monkeypatch.setattr(cli, "run_thesis_wave", fake_wave, raising=False)
    monkeypatch.setattr(cli, "adoption_passes", forbidden, raising=False)
    import src.cli_commands.thesis as thesis_mod

    monkeypatch.setattr(thesis_mod, "run_thesis_wave", fake_wave, raising=False)

    assert main(["run", "thesis-wave"]) == 0
    assert main(["run", "thesis-wave", "--as-of", "2025-04-30T00:00:00+00:00"]) == 0
