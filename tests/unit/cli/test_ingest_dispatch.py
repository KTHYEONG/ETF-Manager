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


def test_recover_healthy_latest_reports_none_without_writes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A latest partition covering the union reports none and writes nothing."""
    from datetime import datetime

    from src.data.pipeline import persist_ingest
    from src.data.schema import spec_for
    from src.data.storage import RawPayload

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ETF_MANAGER_DATA_ROOT", raising=False)
    settings = DataSettings(data_root="data")
    spec = spec_for(Dataset.RATES)
    retrieved = datetime(2024, 2, 1, 5, 0, tzinfo=UTC)
    frame = pl.DataFrame(
        {
            "series_id": ["X"] * 3,
            "observation_date": [date(2024, 1, day) for day in range(1, 4)],
            "value": [1.0, 2.0, 3.0],
            "source": ["fred"] * 3,
            "retrieved_at": [retrieved] * 3,
        },
        schema=dict(spec.columns),
    )
    persist_ingest(
        frame,
        Dataset.RATES,
        RawPayload(
            provider="synthetic",
            endpoint="recover/test",
            request_params={"format": "json"},
            retrieved_at=retrieved,
            extension="json",
            content=b"full",
        ),
        settings,
    )
    before = _data_files(tmp_path)
    with caplog.at_level("INFO"):
        assert main(["maintain", "recover", "rates"]) == 0
    assert _data_files(tmp_path) == before
    assert any("event=recover_none" in record.message for record in caplog.records)


def _seed_fx_pair(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[DataSettings, bytes]:
    """Persist one FX partition and return its settings plus original raw bytes."""
    from src.data.pipeline import persist_ingest
    from src.data.schema import spec_for
    from src.data.storage import RawPayload

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ETF_MANAGER_DATA_ROOT", raising=False)
    settings = DataSettings(data_root="data")
    spec = spec_for(Dataset.FX)
    retrieved = datetime(2024, 2, 1, 5, 0, tzinfo=UTC)
    frame = pl.DataFrame(
        {
            "date": [date(2024, 1, 30), date(2024, 1, 31)],
            "usdkrw": [1300.0, 1301.0],
            "source": ["synthetic", "synthetic"],
            "retrieved_at": [retrieved, retrieved],
        },
        schema=dict(spec.columns),
    )
    content = b'{"v": 1}'
    artifact = persist_ingest(
        frame,
        Dataset.FX,
        RawPayload(
            provider="synthetic",
            endpoint="usdkrw/daily",
            request_params={"interval": "daily"},
            retrieved_at=retrieved,
            extension="json",
            content=content,
        ),
        settings,
    )
    raw_path = tmp_path / "data" / Path(*artifact.manifest.raw_artifact.relative_path.parts)
    raw_path.unlink()
    return settings, content


def _seed_fx_stale_pair(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[DataSettings, bytes]:
    """Persist old plus latest FX partitions with the latest Bronze missing."""
    from src.data.pipeline import persist_ingest
    from src.data.schema import spec_for
    from src.data.storage import RawPayload

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ETF_MANAGER_DATA_ROOT", raising=False)
    settings = DataSettings(data_root="data")
    spec = spec_for(Dataset.FX)

    def frame(dates: list[date], rates: list[float], retrieved: datetime) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "date": dates,
                "usdkrw": rates,
                "source": ["synthetic"] * len(dates),
                "retrieved_at": [retrieved] * len(dates),
            },
            schema=dict(spec.columns),
        )

    def payload(content: bytes, retrieved: datetime) -> RawPayload:
        return RawPayload(
            provider="synthetic",
            endpoint="usdkrw/daily",
            request_params={"interval": "daily"},
            retrieved_at=retrieved,
            extension="json",
            content=content,
        )

    early = datetime(2024, 2, 1, 5, 0, tzinfo=UTC)
    late = datetime(2024, 2, 2, 5, 0, tzinfo=UTC)
    persist_ingest(
        frame([date(2024, 1, 30), date(2024, 1, 31)], [1300.0, 1301.0], early),
        Dataset.FX,
        payload(b'{"v": 1}', early),
        settings,
    )
    latest = persist_ingest(
        frame([date(2024, 1, 30), date(2024, 1, 31)], [1300.5, 1302.0], late),
        Dataset.FX,
        payload(b'{"v": 2}', late),
        settings,
    )
    raw_path = tmp_path / "data" / Path(*latest.manifest.raw_artifact.relative_path.parts)
    raw_path.unlink()
    return settings, b'{"v": 2}'


def test_maintain_data_dry_run_reports_without_fetch_or_deletion(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Dry `maintain data` emits the health summary while leaving every file untouched."""
    _seed_fx_pair(monkeypatch, tmp_path)
    before = _data_files(tmp_path)

    with caplog.at_level("INFO"):
        assert main(["maintain", "data"]) == 0

    assert _data_files(tmp_path) == before
    assert any(
        "event=maintain_data" in record.message and "missing_bronze=1" in record.message
        for record in caplog.records
    )


def test_maintain_data_dry_run_reports_degraded_summary_with_damaged_silver(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Dry `maintain data` still reports health when retention cannot be planned safely."""
    _seed_fx_stale_pair(monkeypatch, tmp_path)
    from src.data.catalog import latest_artifact
    from src.data.settings import DataSettings as _Settings

    latest = latest_artifact(_Settings(data_root="data"), Dataset.FX)
    tampered = pl.read_parquet(latest.normalized_path).with_columns(pl.col("usdkrw") + 1.0)
    tampered.write_parquet(latest.normalized_path)
    before = _data_files(tmp_path)

    with caplog.at_level("INFO"):
        assert main(["maintain", "data"]) == 0

    assert _data_files(tmp_path) == before
    assert any(
        "event=maintain_data" in record.message
        and "damaged_silver=1" in record.message
        and "retention=unknown" in record.message
        for record in caplog.records
    )


def test_maintain_data_apply_repairs_before_prune(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Apply restores Bronze first, then prunes the stale unpinned partition."""
    import src.cli_commands.maintenance as maintenance_mod

    settings, content = _seed_fx_stale_pair(monkeypatch, tmp_path)
    manifests_dir = tmp_path / "data" / "manifests" / "fx"
    stale_manifests = sorted(manifests_dir.glob("*.json"))
    assert len(stale_manifests) == 2
    monkeypatch.setattr(maintenance_mod, "BRONZE_FETCHER", lambda dataset, manifest: content)

    assert main(["maintain", "data", "--apply"]) == 0

    from src.data.catalog import latest_artifact

    latest = latest_artifact(settings, Dataset.FX)
    raw_path = tmp_path / "data" / Path(*latest.manifest.raw_artifact.relative_path.parts)
    assert raw_path.read_bytes() == content
    assert sorted(manifests_dir.glob("*.json")) == [latest.manifest_path]


def test_maintain_data_apply_mismatch_keeps_stale_partitions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A failed repair raises before any prune deletion occurs."""
    import src.cli_commands.maintenance as maintenance_mod

    from src.data.doctor import SourceMismatchError

    _seed_fx_stale_pair(monkeypatch, tmp_path)
    manifests_dir = tmp_path / "data" / "manifests" / "fx"
    before = sorted(manifests_dir.glob("*.json"))
    monkeypatch.setattr(maintenance_mod, "BRONZE_FETCHER", lambda dataset, manifest: b'{"v": "wrong"}')

    with pytest.raises(SourceMismatchError):
        main(["maintain", "data", "--apply"])

    assert sorted(manifests_dir.glob("*.json")) == before


def test_ingest_history_preflight_fails_closed_on_damaged_silver(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A damaged latest Silver blocks vendor fetch and publication entirely."""
    import src.cli_commands.ingest as ingest_mod

    settings, content = _seed_fx_pair(monkeypatch, tmp_path)
    from src.data.catalog import latest_artifact

    latest = latest_artifact(settings, Dataset.FX)
    raw_path = tmp_path / "data" / Path(*latest.manifest.raw_artifact.relative_path.parts)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_bytes(content)
    tampered = pl.read_parquet(latest.normalized_path).with_columns(pl.col("usdkrw") + 1.0)
    tampered.write_parquet(latest.normalized_path)
    before = _data_files(tmp_path)

    calls = {"count": 0}

    def counting_fetch(*args: object, **kwargs: object) -> object:
        calls["count"] += 1
        raise AssertionError("vendor fetch must not run after failed preflight")

    monkeypatch.setattr(ingest_mod, "fetch_and_persist_fx", counting_fetch)
    monkeypatch.setattr(ingest_mod, "fetch_and_persist_prices", counting_fetch)
    monkeypatch.setattr(ingest_mod, "fetch_and_persist_cpi", counting_fetch)
    monkeypatch.setattr(ingest_mod, "fetch_and_persist_factors", counting_fetch)
    monkeypatch.setattr(ingest_mod, "fetch_and_persist_macro", counting_fetch)
    monkeypatch.setattr(ingest_mod, "fetch_and_persist_research_returns", counting_fetch)
    monkeypatch.setattr(ingest_mod, "persist_bootstrap_etf_metadata", counting_fetch)
    monkeypatch.setattr(ingest_mod, "fetch_and_persist_fx_krw_base", counting_fetch)
    monkeypatch.setattr(ingest_mod, "fetch_and_persist_rates", counting_fetch)

    assert main(["ingest", "history", "--start", "2020-01-01", "--end", "2020-12-31"]) == 1
    assert calls["count"] == 0
    assert _data_files(tmp_path) == before


def test_static_dca_preflight_fails_closed_on_damaged_silver(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Static-DCA ingest refuses to fetch once its FX prior cannot be verified."""
    import src.cli_commands.ingest as ingest_mod

    settings, content = _seed_fx_pair(monkeypatch, tmp_path)
    from src.data.catalog import latest_artifact

    latest = latest_artifact(settings, Dataset.FX)
    raw_path = tmp_path / "data" / Path(*latest.manifest.raw_artifact.relative_path.parts)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_bytes(content)
    tampered = pl.read_parquet(latest.normalized_path).with_columns(pl.col("usdkrw") + 1.0)
    tampered.write_parquet(latest.normalized_path)

    calls = {"count": 0}

    def counting_fetch(*args: object, **kwargs: object) -> object:
        calls["count"] += 1
        raise AssertionError("vendor fetch must not run after failed preflight")

    monkeypatch.setattr(ingest_mod, "fetch_and_persist_static_dca_datasets", counting_fetch)

    assert (
        ingest_mod.run_ingest_static_dca(
            start=date(2020, 1, 1),
            end=date(2020, 12, 31),
            tickers=("AAA",),
            fx_provider="fred",
            settings=settings,
            secrets=None,  # type: ignore[arg-type]
        )
        == 1
    )
    assert calls["count"] == 0


def test_maintain_prune_dry_run_leaves_tree_untouched(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Dry `maintain prune` exits 0 without touching the data root."""
    _seed_fx_pair(monkeypatch, tmp_path)
    before = _data_files(tmp_path)
    assert main(["maintain", "prune"]) == 0
    assert _data_files(tmp_path) == before


def test_run_ingest_history_untrusted_preflight_raises_without_publication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed prior-partition verification raises without any fresh-slice publication."""
    import src.cli_commands.ingest as ingest_mod
    from src.data.storage import UntrustedDatasetError

    def raising_preflight(settings: object, datasets: object) -> None:
        raise UntrustedDatasetError("ingest preflight refuses fx: latest silver is damaged")

    calls = {"count": 0}

    def counting_fetch(*args: object, **kwargs: object) -> object:
        calls["count"] += 1
        raise AssertionError("vendor fetch must not run after failed preflight")

    monkeypatch.setattr(ingest_mod, "_preflight_relevant_silver", raising_preflight)
    monkeypatch.setattr(ingest_mod, "fetch_and_persist_fx", counting_fetch)
    monkeypatch.setattr(ingest_mod, "fetch_and_persist_prices", counting_fetch)
    monkeypatch.setattr(ingest_mod, "fetch_and_persist_cpi", counting_fetch)
    monkeypatch.setattr(ingest_mod, "fetch_and_persist_factors", counting_fetch)
    monkeypatch.setattr(ingest_mod, "fetch_and_persist_macro", counting_fetch)
    monkeypatch.setattr(ingest_mod, "fetch_and_persist_research_returns", counting_fetch)
    monkeypatch.setattr(ingest_mod, "persist_bootstrap_etf_metadata", counting_fetch)
    monkeypatch.setattr(ingest_mod, "fetch_and_persist_fx_krw_base", counting_fetch)
    monkeypatch.setattr(ingest_mod, "fetch_and_persist_rates", counting_fetch)

    with pytest.raises(UntrustedDatasetError):
        ingest_mod.run_ingest_history(
            start=date(2020, 1, 1),
            end=date(2020, 12, 31),
            fx_provider="fred",
            settings=DataSettings(),
            secrets=None,  # type: ignore[arg-type]
        )
    assert calls["count"] == 0


def test_run_ingest_history_provider_failure_returns_one_with_repair_action(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A vendor failure surfaces exit one with dataset, command, and repair action."""
    import src.cli_commands.ingest as ingest_mod

    monkeypatch.setattr(ingest_mod, "_preflight_relevant_silver", lambda *a, **k: None)
    monkeypatch.setattr(
        ingest_mod, "fetch_and_persist_fx", lambda **k: (_ for _ in ()).throw(ProviderError("fx down"))
    )

    with caplog.at_level("ERROR"):
        code = ingest_mod.run_ingest_history(
            start=date(2020, 1, 1),
            end=date(2020, 12, 31),
            fx_provider="fred",
            settings=DataSettings(),
            secrets=None,  # type: ignore[arg-type]
        )

    assert code == 1
    assert any(
        "history_failed" in record.message
        and "dataset=history" in record.message
        and "maintain data" in record.message
        for record in caplog.records
    )


def test_run_ingest_smoke_required_failure_reports_repair_action(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A required smoke fetch failure exits one with dataset, command, and repair action."""
    import src.cli_commands.ingest as ingest_mod

    monkeypatch.setattr(ingest_mod, "_preflight_relevant_silver", lambda *a, **k: None)
    monkeypatch.setattr(
        ingest_mod, "fetch_and_persist_fx", lambda **k: (_ for _ in ()).throw(ProviderError("fx down"))
    )

    with caplog.at_level("ERROR"):
        code = ingest_mod.run_ingest_smoke(
            start=date(2020, 1, 1),
            end=date(2020, 1, 5),
            ticker="VT",
            fx_provider="fred",
            settings=DataSettings(),
            secrets=None,  # type: ignore[arg-type]
        )

    assert code == 1
    assert any(
        "smoke_required_failed" in record.message
        and "dataset=smoke" in record.message
        and "maintain data" in record.message
        for record in caplog.records
    )
