"""Ensure co-modification wiring for cli."""
from __future__ import annotations

import argparse
from datetime import date

import pytest

from src.cli import main
from src.data.storage import UntrustedDatasetError


@pytest.mark.parametrize("scenario_id", ["test_cli_import"])
def test_cli_import(scenario_id: str) -> None:
    import src.cli as mod
    assert hasattr(mod, "main")


def test_main_dispatches_ingest_history_to_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each registered ingest target reaches exactly its real handler."""
    import src.cli as cli_mod

    seen: dict[str, object] = {}

    def fake_run_ingest_history(**kwargs: object) -> int:
        seen.update(kwargs)
        return 0

    monkeypatch.setattr(cli_mod, "run_ingest_history", fake_run_ingest_history)
    monkeypatch.setattr(cli_mod, "load_provider_secrets", lambda: object())

    assert main(["ingest", "history", "--start", "2020-01-01", "--end", "2020-12-31"]) == 0
    assert seen["start"] == date(2020, 1, 1)
    assert seen["end"] == date(2020, 12, 31)
    assert seen["fx_provider"] == "fred"


def test_main_dispatches_run_and_maintain_handlers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run and maintain targets dispatch to their real command functions."""
    import src.cli as cli_mod
    import src.cli_commands.maintenance as maintenance_mod

    baseline_seen: dict[str, object] = {}
    maintain_seen: dict[str, object] = {}

    def fake_baseline(**kwargs: object) -> int:
        baseline_seen.update(kwargs)
        return 0

    def fake_maintain(args: argparse.Namespace, settings: object) -> int:
        maintain_seen["target"] = getattr(args, "target", None)
        return 0

    monkeypatch.setattr(cli_mod, "run_baseline_command", fake_baseline)
    monkeypatch.setattr(maintenance_mod, "run_maintain_data_command", fake_maintain)

    assert (
        main(
            [
                "run",
                "baseline",
                "--id",
                "dca_global",
                "--ticker",
                "QQQ",
                "--start",
                "2020-01-01",
                "--end",
                "2020-12-31",
                "--contribution-krw",
                "1000000",
            ]
        )
        == 0
    )
    assert baseline_seen["baseline_id"] == "dca_global"
    assert baseline_seen["ticker"] == "QQQ"

    assert main(["maintain", "data"]) == 0
    assert maintain_seen["target"] == "data"


def test_main_trust_error_returns_one_with_actionable_diagnostic(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Untrusted Silver during ingest exits one with a repair diagnostic and no publication."""
    import src.cli as cli_mod
    import src.cli_commands.ingest as ingest_mod

    calls = {"count": 0}

    def raising_preflight(settings: object, datasets: object) -> None:
        raise UntrustedDatasetError("ingest preflight refuses fx: latest silver is damaged; repair before ingesting")

    def counting_fetch(*args: object, **kwargs: object) -> object:
        calls["count"] += 1
        raise AssertionError("vendor fetch must not run after failed preflight")

    monkeypatch.setattr(ingest_mod, "_preflight_relevant_silver", raising_preflight)
    monkeypatch.setattr(ingest_mod, "fetch_and_persist_fx", counting_fetch)
    monkeypatch.setattr(ingest_mod, "fetch_and_persist_prices", counting_fetch)
    monkeypatch.setattr(cli_mod, "load_provider_secrets", lambda: object())

    with caplog.at_level("ERROR"):
        code = main(["ingest", "history", "--start", "2020-01-01", "--end", "2020-12-31"])

    assert code == 1
    assert calls["count"] == 0
    assert any(
        "[DATA]" in record.message
        and "cli_ingest_failed" in record.message
        and "maintain data" in record.message
        for record in caplog.records
    )


def test_legacy_command_syntax_preserved() -> None:
    """Existing documented command lines keep target, options, and exit codes."""
    from src.cli_commands.parser import _build_parser

    args = _build_parser().parse_args(
        ["ingest", "prices", "--tickers", "QQQ", "--start", "2020-01-01", "--end", "2020-12-31"]
    )
    assert args.dataset == "prices"
    assert args.tickers == ["QQQ"]

    maintain_args = _build_parser().parse_args(["maintain", "data"])
    assert maintain_args.target == "data"
    assert maintain_args.apply is False

    assert main(["ingest", "history"]) == 2
    assert main(["ingest", "prices", "--start", "2020-01-01", "--end", "2020-12-31"]) == 2
    assert main(["ingest", "fx", "--start", "2020-01-01", "--end", "2020-12-31"]) == 2


def test_thesis_panel_nport_partial_keeps_success(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A failed optional N-PORT leg keeps thesis-panel success with an actionable warning."""
    import src.cli as cli_mod
    import src.data.fetch as fetch_mod
    from src.data.providers.base import ProviderError

    monkeypatch.setattr(
        fetch_mod,
        "fetch_and_persist_static_dca_datasets",
        lambda **kwargs: {"prices": 8, "fx": 8, "cpi": 8},
    )
    monkeypatch.setattr(
        "src.data.panel_freshness.iter_nport_quarters_for_panel", lambda *a, **k: ["2019q4"]
    )

    def failing_nport(**kwargs: object) -> object:
        raise ProviderError("nport rejected")

    monkeypatch.setattr(cli_mod, "fetch_and_persist_nport_quarters", failing_nport)
    monkeypatch.setattr(cli_mod, "load_provider_secrets", lambda: object())

    with caplog.at_level("WARNING"):
        code = main(["ingest", "thesis-panel"])

    assert code == 0
    assert any(
        "thesis_panel_nport_partial" in record.message
        and "dataset=nport" in record.message
        and "maintain data" in record.message
        for record in caplog.records
    )
