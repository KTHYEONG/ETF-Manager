"""CLI contract tests for standalone pension ETF selection."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import src.cli_commands.campaign as campaign_mod
from src import cli
from src.cli import main
from src.cli_commands.parser import _build_parser
from src.data.settings import DataSettings
from src.data.storage import UntrustedDatasetError
from src.sim.pension_engine import PensionDataError
from src.validation import pension_selection as pension_selection_module

_REPO = Path(__file__).resolve().parents[3]
_CONFIG_PATH = _REPO / "experiments" / "pension_selection_v1.json"
_GIT_COMMIT = "0" * 40


def _fake_manifest(_settings: DataSettings, dataset: object) -> SimpleNamespace:
    if dataset in {
        pension_selection_module.Dataset.FX,
        pension_selection_module.Dataset.CPI,
    }:
        raise UntrustedDatasetError("optional fixture is absent")
    return SimpleNamespace(manifest=SimpleNamespace(normalized_sha256=f"hash-{dataset}"))


def _stub_report() -> SimpleNamespace:
    return SimpleNamespace(
        name="pension_selection_v1",
        status="SELECTED",
        selected_arm_id="sp500_100",
        verdicts=(SimpleNamespace(arm_id="sp500_100"),),
    )


def _stub_writer(
    tmp_path: Path,
    captured: list[tuple[str, dict[str, str]]],
) -> object:
    def write(
        _report: object,
        _settings: DataSettings,
        *,
        experiment_id: str,
        provenance: dict[str, str],
    ) -> Path:
        captured.append((experiment_id, provenance))
        path = tmp_path / f"selection_{experiment_id}.json"
        path.write_text(json.dumps({"status": "SELECTED", "provenance": provenance}), encoding="utf-8")
        return path

    return write


def test_pension_selection_command_writes_report_and_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The command returns zero and persists deterministic config and source provenance."""
    captured: list[tuple[str, dict[str, str]]] = []
    monkeypatch.setattr(campaign_mod, "_resolve_git_commit", lambda: _GIT_COMMIT)
    monkeypatch.setattr(campaign_mod, "latest_artifact", _fake_manifest)
    monkeypatch.setattr(pension_selection_module, "run_pension_selection", lambda *_args, **_kwargs: _stub_report())
    monkeypatch.setattr(
        pension_selection_module,
        "write_pension_selection_report",
        _stub_writer(tmp_path, captured),
    )
    with caplog.at_level("INFO"):
        code = campaign_mod.run_pension_selection_command(
            config_path=str(_CONFIG_PATH),
            settings=DataSettings(data_root=str(tmp_path / "data")),
            seed=17,
        )
    assert code == 0
    assert len(captured) == 1
    experiment_id, provenance = captured[0]
    assert len(experiment_id) == 16
    assert len(provenance["config_sha256"]) == 64
    assert len(provenance["tax_regime_sha256"]) == 64
    assert len(provenance["etf_identity_sha256"]) == 64
    assert len(provenance["campaign_config_sha256"].split(",")) == 2
    assert provenance["git_commit"] == _GIT_COMMIT
    assert provenance["seed"] == "17"
    assert (tmp_path / f"selection_{experiment_id}.json").is_file()
    assert "pension_selection_cli_done" in caplog.text


def test_pension_selection_command_fails_closed_on_data_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Pension data errors return one and expose the typed CLI failure event."""
    monkeypatch.setattr(campaign_mod, "_resolve_git_commit", lambda: _GIT_COMMIT)
    monkeypatch.setattr(campaign_mod, "latest_artifact", _fake_manifest)

    def fail(*_args: object, **_kwargs: object) -> object:
        raise PensionDataError("missing certified panel")

    monkeypatch.setattr(pension_selection_module, "run_pension_selection", fail)
    with caplog.at_level("ERROR"):
        code = campaign_mod.run_pension_selection_command(
            config_path=str(_CONFIG_PATH),
            settings=DataSettings(data_root=str(tmp_path / "data")),
            seed=17,
        )
    assert code == 1
    assert "pension_selection_cli_failed" in caplog.text
    assert "PensionDataError" in caplog.text


def test_selection_digest_changes_with_historical_campaign_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A historical config edit changes the experiment id without changing selection bytes."""
    document = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
    campaign = tmp_path / "campaign.json"
    campaign.write_text('{"name":"first"}', encoding="utf-8")
    document["historical"]["campaign_config_paths"] = [str(campaign)]  # type: ignore[index]
    document["etf_identity_path"] = str(_REPO / "configs" / "data" / "pension_etfs_2026.json")
    document["tax_regime_path"] = str(_REPO / "configs" / "tax" / "kr_pension_2026.json")
    config = tmp_path / "selection.json"
    config.write_text(json.dumps(document), encoding="utf-8")
    captured: list[tuple[str, dict[str, str]]] = []
    monkeypatch.setattr(campaign_mod, "_resolve_git_commit", lambda: _GIT_COMMIT)
    monkeypatch.setattr(campaign_mod, "latest_artifact", _fake_manifest)
    monkeypatch.setattr(pension_selection_module, "run_pension_selection", lambda *_args, **_kwargs: _stub_report())
    monkeypatch.setattr(
        pension_selection_module,
        "write_pension_selection_report",
        _stub_writer(tmp_path, captured),
    )
    settings = DataSettings(data_root=str(tmp_path / "data"))
    assert campaign_mod.run_pension_selection_command(config_path=str(config), settings=settings, seed=17) == 0
    campaign.write_text('{"name":"second"}', encoding="utf-8")
    assert campaign_mod.run_pension_selection_command(config_path=str(config), settings=settings, seed=17) == 0
    assert captured[0][0] != captured[1][0]
    assert captured[0][1]["config_sha256"] == captured[1][1]["config_sha256"]
    assert captured[0][1]["campaign_config_sha256"] != captured[1][1]["campaign_config_sha256"]


def test_parser_exposes_pension_selection_target() -> None:
    """The facade parser exposes config and required deterministic seed inputs."""
    args = _build_parser().parse_args(
        ["run", "pension-selection", "--config", "selection.json", "--seed", "1"]
    )
    assert args.target == "pension-selection"
    assert args.config == "selection.json"
    assert args.seed == 1


def test_pension_selection_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    """The facade dispatches once with config and seed; seed remains mandatory."""
    captured: dict[str, object] = {}

    def fake_command(*, config_path: str, settings: object, seed: int) -> int:
        captured["config_path"] = config_path
        captured["settings"] = settings
        captured["seed"] = seed
        return 0

    monkeypatch.setattr(cli, "run_pension_selection_command", fake_command)
    assert main(["run", "pension-selection", "--config", "c.json", "--seed", "7"]) == 0
    assert captured["config_path"] == "c.json"
    assert captured["seed"] == 7
    assert isinstance(captured["settings"], DataSettings)
    assert main(["run", "pension-selection", "--config", "c.json"]) == 2
