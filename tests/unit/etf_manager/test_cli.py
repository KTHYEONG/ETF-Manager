"""Unit tests for the ingest CLI dispatch."""

from __future__ import annotations

from datetime import date

import pytest

from src.etf_manager import cli
from src.etf_manager.cli import main


@pytest.fixture(autouse=True)
def _provider_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolve secrets from the process env so no sops or network call happens."""
    for name in ("TIINGO_API", "FRED_API", "ECOS_API"):
        monkeypatch.setenv(name, f"unit-test-{name}")


def test_cli_d07_ingest_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI-D07-ingest-dispatch"""
    captured: dict[str, object] = {}

    def fake_prices(tickers: tuple[str, ...], start: date, end: date, **kwargs: object) -> None:
        captured["tickers"] = tickers
        captured["start"] = start
        captured["end"] = end

    monkeypatch.setattr(cli, "fetch_and_persist_prices", fake_prices)

    exit_code = main(["ingest", "prices", "--tickers", "VT", "--start", "2024-01-01", "--end", "2024-02-01"])
    assert exit_code == 0
    assert captured["tickers"] == ("VT",)
    assert captured["start"] == date(2024, 1, 1)
    assert captured["end"] == date(2024, 2, 1)

    fx_calls: list[int] = []

    def fake_fx(**kwargs: object) -> None:
        fx_calls.append(1)

    monkeypatch.setattr(cli, "fetch_and_persist_fx", fake_fx)

    invalid_exit = main(["ingest", "fx", "--start", "2024-01-01", "--end", "2024-02-01"])
    assert invalid_exit == 2
    assert fx_calls == []
