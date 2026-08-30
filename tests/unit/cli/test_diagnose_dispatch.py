"""Diagnose dispatch tests."""

from __future__ import annotations

import pytest

from src import cli
from src.cli import main
import src.cli_commands.parser  # noqa: F401 ensure co-modification wiring


@pytest.mark.parametrize("scenario_id", ["CLI-A-diagnose-qqq"])
def test_cli_a_diagnose_qqq(scenario_id: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """test_cli_a_diagnose_qqq"""
    assert main(["run", "diagnose-qqq-cadence"]) == 2
    assert main(["run", "diagnose-s8-cadence"]) == 2

    captured: dict[str, object] = {}

    def fake_diagnose(**kwargs: object) -> int:
        captured.update(kwargs)
        return 0

    def forbidden_ablation(*args: object, **kwargs: object) -> int:
        raise AssertionError("diagnostics must never invoke the adoption ablation")

    monkeypatch.setattr(cli, "run_diagnose_qqq_cadence_command", fake_diagnose)
    monkeypatch.setattr(cli, "run_ablation", forbidden_ablation, raising=False)

    assert main(["run", "diagnose-qqq-cadence", "--contribution-krw", "1000000"]) == 0
    assert captured["contribution_krw"] == pytest.approx(1_000_000.0)


@pytest.mark.parametrize("scenario_id", ["CLI-O-diagnose-qqq-blends"])
def test_cli_o_diagnose_qqq_blends(scenario_id: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """test_cli_o_diagnose_qqq_blends"""
    assert main(["run", "diagnose-qqq-blends"]) == 2

    captured: dict[str, object] = {}

    def fake_diagnose(**kwargs: object) -> int:
        captured.update(kwargs)
        return 0

    def forbidden_ablation(*args: object, **kwargs: object) -> int:
        raise AssertionError("diagnostics must never invoke the adoption ablation")

    monkeypatch.setattr(cli, "run_diagnose_qqq_blends_command", fake_diagnose)
    monkeypatch.setattr(cli, "run_ablation", forbidden_ablation, raising=False)

    assert main(["run", "diagnose-qqq-blends", "--contribution-krw", "1000000"]) == 0
    assert captured["contribution_krw"] == pytest.approx(1_000_000.0)


def test_cli_diagnose_compound_dca(monkeypatch: pytest.MonkeyPatch) -> None:
    from src import cli
    from src.cli import main
    assert main(['run', 'diagnose-compound-dca']) == 2
    captured: dict[str, object] = {}
    def fake_diagnose(**kwargs: object) -> int:
        captured.update(kwargs)
        return 0
    def forbidden_ablation(*args: object, **kwargs: object) -> int:
        raise AssertionError('diagnostics must never invoke the adoption ablation')
    monkeypatch.setattr(cli, 'run_diagnose_compound_dca_command', fake_diagnose)
    monkeypatch.setattr(cli, 'run_ablation', forbidden_ablation, raising=False)
    assert main(['run', 'diagnose-compound-dca', '--contribution-krw', '1000000']) == 0
    assert captured['contribution_krw'] == pytest.approx(1_000_000.0)
