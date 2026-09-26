"""Command tests for maintain results actions."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from src.cli import main
from src.cli_commands.results import run_results_command
from src.data.result_store import ResultKind, write_result
from src.data.settings import DataSettings


def _hermetic(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ETF_MANAGER_DATA_ROOT", raising=False)


def _write(exp: str, kind: ResultKind, run: str, day: int):  # type: ignore[no-untyped-def]
    return write_result(
        DataSettings(data_root="data"),
        experiment=exp,
        kind=kind,
        run_id=run,
        payload={"run": run},
        written_at=datetime(2026, 1, day, tzinfo=UTC),
    )


def test_cli_list_prints_ledger_rows(tmp_path: Path, monkeypatch, capsys) -> None:  # type: ignore[no-untyped-def]
    """CLI list prints ledger rows."""
    _hermetic(tmp_path, monkeypatch)
    _write("wf_qqq_adaptive_v5", ResultKind.WALK_FORWARD, "abc123", 1)

    assert main(["maintain", "results", "list"]) == 0
    out = capsys.readouterr().out
    assert "wf_qqq_adaptive_v5" in out
    assert "abc123" in out


def test_cli_prune_defaults_to_dry_run(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """CLI prune defaults to dry run."""
    _hermetic(tmp_path, monkeypatch)
    for day in (1, 2, 3, 4):
        _write("exp", ResultKind.WALK_FORWARD, f"r{day}", day)

    assert main(["maintain", "results", "prune", "--keep", "1"]) == 0
    assert len(list((tmp_path / "data" / "runs" / "exp").glob("*.json"))) == 4


def test_cli_promote_success(tmp_path: Path, monkeypatch, capsys) -> None:  # type: ignore[no-untyped-def]
    """CLI promote copies the artifact and prints paths."""
    _hermetic(tmp_path, monkeypatch)
    _write("exp", ResultKind.WALK_FORWARD, "a", 1)

    assert main(["maintain", "results", "promote", "--experiment", "exp", "--kind", "walk_forward", "--run-id", "a"]) == 0
    out = capsys.readouterr().out
    assert "data/research/exp/walk_forward_a.json" in out
    assert (tmp_path / "data" / "research" / "exp" / "walk_forward_a.json").is_file()


def test_cli_promote_divergence_returns_failure(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """CLI promote divergence returns failure."""
    _hermetic(tmp_path, monkeypatch)
    _write("exp", ResultKind.WALK_FORWARD, "a", 1)
    dest = tmp_path / "data" / "research" / "exp" / "walk_forward_a.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({"other": True}), encoding="utf-8")

    assert main(["maintain", "results", "promote", "--experiment", "exp", "--kind", "walk_forward", "--run-id", "a"]) == 1
    assert json.loads(dest.read_text(encoding="utf-8")) == {"other": True}


def test_cli_migrate_apply(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """CLI migrate apply."""
    _hermetic(tmp_path, monkeypatch)
    flat = tmp_path / "data" / "runs" / "experiments"
    flat.mkdir(parents=True)
    (flat / "wf_qqq_adaptive_v5_893bec.json").write_text(json.dumps({"name": "wf_qqq_adaptive_v5"}), encoding="utf-8")

    assert main(["maintain", "results", "migrate", "--apply"]) == 0
    assert (tmp_path / "data" / "runs" / "wf_qqq_adaptive_v5" / "legacy_893bec.json").is_file()


def test_run_results_command_rejects_unknown_action(tmp_path: Path) -> None:
    """Unknown results action returns failure."""
    settings = DataSettings(data_root=tmp_path / "data")
    assert run_results_command(argparse.Namespace(results_action="bogus"), settings) == 1
