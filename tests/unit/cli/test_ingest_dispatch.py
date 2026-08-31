"""Ingest dispatch tests."""

from __future__ import annotations

import pytest

from src import cli
from src.cli import main
from src.data.providers.base import ProviderError
from src.data.schema import Dataset


class _FakeManifest:
    def __init__(self, row_count: int) -> None:
        self.row_count = row_count
        self.normalized_sha256 = "f" * 64


class _FakeArtifact:
    def __init__(self, row_count: int) -> None:
        self.manifest = _FakeManifest(row_count)


@pytest.mark.parametrize("scenario_id", ["CLI-E02-ingest-smoke-required-ok"])
def test_cli_e02_ingest_smoke_required_ok(scenario_id: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """test_cli_e02_ingest_smoke_required_ok"""
    calls = {"fx": 0, "prices": 0, "cpi": 0}
    seen_datasets: list[Dataset] = []

    def fake_fx(**kwargs: object) -> _FakeArtifact:
        calls["fx"] += 1
        return _FakeArtifact(4)

    def fake_prices(tickers: tuple[str, ...], start, end, **kwargs: object) -> _FakeArtifact:
        calls["prices"] += 1
        return _FakeArtifact(4)

    def fake_cpi(start, end, **kwargs: object) -> _FakeArtifact:
        calls["cpi"] += 1
        raise ProviderError("ecos cpi rejected")

    def fake_latest(settings: object, dataset: Dataset) -> _FakeArtifact:
        seen_datasets.append(dataset)
        return _FakeArtifact(4)

    import src.cli_commands.ingest as ingest_mod

    monkeypatch.setattr(ingest_mod, "fetch_and_persist_fx", fake_fx)
    monkeypatch.setattr(ingest_mod, "fetch_and_persist_prices", fake_prices)
    monkeypatch.setattr(ingest_mod, "fetch_and_persist_cpi", fake_cpi)
    monkeypatch.setattr(ingest_mod, "latest_artifact", fake_latest)
    # also patch cli for compatibility if facade re-exports
    monkeypatch.setattr(cli, "fetch_and_persist_fx", fake_fx, raising=False)
    monkeypatch.setattr(cli, "fetch_and_persist_prices", fake_prices, raising=False)
    monkeypatch.setattr(cli, "fetch_and_persist_cpi", fake_cpi, raising=False)

    exit_code = main(["ingest", "smoke"])

    assert exit_code == 0
    assert calls == {"fx": 1, "prices": 1, "cpi": 1}
    assert set(seen_datasets) == {Dataset.PRICES, Dataset.FX}


@pytest.mark.parametrize("scenario_id", ["CLI-F03-ingest-history"])
def test_cli_f03_ingest_history(scenario_id: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """test_cli_f03_ingest_history"""
    from src.analytics.us_vehicles import history_price_tickers
    from src.policy.targets import all_policy_tickers

    calls = {"fx": 0, "prices": 0, "cpi": 0, "factors": 0, "macro": 0, "research": 0}
    seen_tickers: tuple[str, ...] = ()
    seen_series_ids: list[str] = []
    seen_datasets: list[Dataset] = []

    def fake_fx(**kwargs: object) -> _FakeArtifact:
        calls["fx"] += 1
        return _FakeArtifact(8)

    def fake_prices(tickers: tuple[str, ...], start, end, **kwargs: object) -> _FakeArtifact:
        nonlocal seen_tickers
        calls["prices"] += 1
        seen_tickers = tickers
        return _FakeArtifact(8)

    def fake_cpi(start, end, **kwargs: object) -> _FakeArtifact:
        calls["cpi"] += 1
        return _FakeArtifact(8)

    def fake_factors(start, end, **kwargs: object) -> _FakeArtifact:
        calls["factors"] += 1
        return _FakeArtifact(8)

    def fake_macro(series_id: object, start, end, **kwargs: object) -> _FakeArtifact:
        calls["macro"] += 1
        seen_series_ids.append(series_id)
        return _FakeArtifact(8)

    def fake_research(start, end, **kwargs: object) -> _FakeArtifact:
        calls["research"] += 1
        return _FakeArtifact(8)

    def fake_metadata(settings: object, **kwargs: object) -> _FakeArtifact:
        return _FakeArtifact(8)

    def fake_latest(settings: object, dataset: Dataset) -> _FakeArtifact:
        seen_datasets.append(dataset)
        return _FakeArtifact(8)

    import src.cli_commands.ingest as ingest_mod

    monkeypatch.setattr(ingest_mod, "fetch_and_persist_fx", fake_fx)
    monkeypatch.setattr(ingest_mod, "fetch_and_persist_prices", fake_prices)
    monkeypatch.setattr(ingest_mod, "fetch_and_persist_cpi", fake_cpi)
    monkeypatch.setattr(ingest_mod, "fetch_and_persist_factors", fake_factors)
    monkeypatch.setattr(ingest_mod, "fetch_and_persist_macro", fake_macro)
    monkeypatch.setattr(ingest_mod, "fetch_and_persist_research_returns", fake_research)
    monkeypatch.setattr(ingest_mod, "persist_bootstrap_etf_metadata", fake_metadata)
    monkeypatch.setattr(ingest_mod, "latest_artifact", fake_latest)

    exit_code = main(["ingest", "history", "--start", "2020-01-01", "--end", "2020-12-31"])

    assert exit_code == 0
    assert calls == {"fx": 1, "prices": 1, "cpi": 1, "factors": 1, "macro": 1, "research": 1}
    assert seen_tickers == history_price_tickers()
    assert "QQQ" in seen_tickers
    assert set(seen_tickers) - set(all_policy_tickers()) == {
        "BOTZ",
        "GRID",
        "IBB",
        "IEMG",
        "ITA",
        "ITOT",
        "IWF",
        "PAVE",
        "ROBO",
        "SCHF",
        "SOXX",
        "XLI",
    }
    assert seen_series_ids == [("VIXCLS", "BAA10Y")]
    expected_datasets = {
        Dataset.PRICES,
        Dataset.FX,
        Dataset.CPI,
        Dataset.FACTORS,
        Dataset.MACRO,
        Dataset.RESEARCH_RETURNS,
        Dataset.ETF_METADATA,
    }
    assert set(seen_datasets) == expected_datasets

    assert main(["ingest", "history"]) == 2
