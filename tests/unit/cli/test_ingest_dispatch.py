"""Ingest dispatch tests."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl
import pytest

from src import cli
from src.cli import main
from src.data.providers.base import ProviderError
from src.data.schema import Dataset
from src.data.settings import DataSettings


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
    monkeypatch.setattr(ingest_mod, "fetch_and_persist_fx_krw_base", lambda *a, **k: _FakeArtifact(8))
    monkeypatch.setattr(ingest_mod, "fetch_and_persist_rates", lambda *a, **k: _FakeArtifact(8))
    monkeypatch.setattr(ingest_mod, "latest_artifact", fake_latest)

    exit_code = main(["ingest", "history", "--start", "2020-01-01", "--end", "2020-12-31"])

    assert exit_code == 0
    assert calls == {"fx": 1, "prices": 1, "cpi": 1, "factors": 1, "macro": 1, "research": 1}
    assert seen_tickers == history_price_tickers()
    assert "QQQ" in seen_tickers
    assert set(seen_tickers) - set(all_policy_tickers()) == {
        "BOTZ",
        "EFA",
        "EWJ",
        "GLD",
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
        "SPY",
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
        Dataset.FX_KRW_BASE,
        Dataset.RATES,
    }
    assert set(seen_datasets) == expected_datasets

    assert main(["ingest", "history"]) == 2


def test_history_ingest_empty_base_rate_fails_with_reason(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """An empty FX_KRW_BASE partition fails history ingest with an empty-catalog reason."""
    import src.cli_commands.ingest as ingest_mod

    monkeypatch.setattr(ingest_mod, "fetch_and_persist_fx", lambda **k: _FakeArtifact(8))
    monkeypatch.setattr(ingest_mod, "fetch_and_persist_prices", lambda *a, **k: _FakeArtifact(8))
    monkeypatch.setattr(ingest_mod, "fetch_and_persist_cpi", lambda *a, **k: _FakeArtifact(8))
    monkeypatch.setattr(ingest_mod, "fetch_and_persist_factors", lambda *a, **k: _FakeArtifact(8))
    monkeypatch.setattr(ingest_mod, "fetch_and_persist_macro", lambda *a, **k: _FakeArtifact(8))
    monkeypatch.setattr(ingest_mod, "fetch_and_persist_research_returns", lambda *a, **k: _FakeArtifact(8))
    monkeypatch.setattr(ingest_mod, "persist_bootstrap_etf_metadata", lambda *a, **k: _FakeArtifact(8))
    monkeypatch.setattr(ingest_mod, "fetch_and_persist_fx_krw_base", lambda *a, **k: _FakeArtifact(0))
    monkeypatch.setattr(ingest_mod, "fetch_and_persist_rates", lambda *a, **k: _FakeArtifact(8))

    def fake_latest(settings: object, dataset: Dataset) -> _FakeArtifact:
        return _FakeArtifact(0 if dataset is Dataset.FX_KRW_BASE else 8)

    monkeypatch.setattr(ingest_mod, "latest_artifact", fake_latest)

    with caplog.at_level("ERROR"):
        code = ingest_mod.run_ingest_history(
            start=date(2020, 1, 1),
            end=date(2020, 12, 31),
            fx_provider="fred",
            settings=DataSettings(),
            secrets=None,  # type: ignore[arg-type]
        )

    assert code == 1
    assert any("reason=empty_catalog" in record.message and "fx_krw_base" in record.message for record in caplog.records)


def test_macro_cli_retains_other_series(monkeypatch: pytest.MonkeyPatch) -> None:
    """A single-series macro CLI refresh never drops other series rows."""
    import src.cli as cli_mod

    seen: dict[str, object] = {}

    def fake_macro(series_id: object, start: object, end: object, **kwargs: object) -> _FakeArtifact:
        seen.update(kwargs)
        return _FakeArtifact(8)

    monkeypatch.setattr(cli_mod, "fetch_and_persist_macro", fake_macro)
    monkeypatch.setattr(cli_mod, "load_provider_secrets", lambda: object())

    code = main(["ingest", "macro", "--series-id", "DTB3", "--start", "2020-01-01", "--end", "2020-12-31"])

    assert code == 0
    assert seen.get("retain_other_series") is True


def _seed_truncated_rates(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> DataSettings:
    """Persist a 10-row partition plus a newer truncated 3-row latest for recover tests."""
    from src.data.pipeline import persist_ingest
    from src.data.schema import spec_for
    from src.data.storage import RawPayload

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ETF_MANAGER_DATA_ROOT", raising=False)
    settings = DataSettings(data_root="data")
    spec = spec_for(Dataset.RATES)
    retrieved = datetime(2024, 2, 1, 5, 0, tzinfo=UTC)

    def frame(days: range, base: float) -> pl.DataFrame:
        rows = list(days)
        return pl.DataFrame(
            {
                "series_id": ["X"] * len(rows),
                "observation_date": [date(2024, 1, day) for day in rows],
                "value": [base + day for day in rows],
                "source": ["fred"] * len(rows),
                "retrieved_at": [retrieved] * len(rows),
            },
            schema=dict(spec.columns),
        )

    def payload(content: bytes, at: datetime) -> RawPayload:
        return RawPayload(
            provider="synthetic",
            endpoint="recover/test",
            request_params={"format": "json"},
            retrieved_at=at,
            extension="json",
            content=content,
        )

    persist_ingest(frame(range(1, 11), 1.0), Dataset.RATES, payload(b"full", datetime(2024, 3, 1, tzinfo=UTC)), settings)
    persist_ingest(
        frame(range(6, 9), 100.0), Dataset.RATES, payload(b"truncated", datetime(2024, 4, 1, tzinfo=UTC)), settings
    )
    return settings


def _data_files(tmp_path: Path) -> set[str]:
    root = tmp_path / "data"
    return {path.relative_to(tmp_path).as_posix() for path in root.rglob("*")} if root.is_dir() else set()


def test_recover_dry_run_writes_nothing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Dry-run recover exits 0 without touching the data root."""
    _seed_truncated_rates(monkeypatch, tmp_path)
    before = _data_files(tmp_path)
    assert main(["maintain", "recover", "rates"]) == 0
    assert _data_files(tmp_path) == before


def test_recover_apply_publishes_recovered_partition(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Apply publishes a new manifest and the catalog serves the recovered rows."""
    from src.data.catalog import latest_artifact

    settings = _seed_truncated_rates(monkeypatch, tmp_path)
    assert main(["maintain", "recover", "rates", "--apply"]) == 0
    selected = latest_artifact(settings, Dataset.RATES)
    assert selected.manifest.row_count == 10
    assert selected.manifest.prior_manifest_sha256 is not None


def test_recover_unknown_dataset_rejected(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """An unknown dataset name is a usage error with no writes."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ETF_MANAGER_DATA_ROOT", raising=False)
    assert main(["maintain", "recover", "bogus"]) == 2
    assert _data_files(tmp_path) == set()
