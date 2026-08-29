"""Integration tests wiring provider clients into persist_ingest."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

import httpx
import pytest

from src.data.fetch import (
    fetch_and_persist_cpi,
    fetch_and_persist_fx,
    fetch_and_persist_macro,
    fetch_and_persist_prices,
    fetch_and_persist_static_dca_datasets,
    resolve_price_http_window,
)
from src.data.pit import AVAILABLE_AT
from src.data.schema import Dataset, MissingPolicy, spec_for
from src.data.secrets import ProviderSecrets
from src.data.settings import DataSettings
from src.data.storage import DataStore

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "providers"
_SECRETS = ProviderSecrets(tiingo_api="wire-tiingo-token", fred_api="wire-fred-key", ecos_api="wire-ecos-key")


def _client_serving(body: bytes) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body)

    return httpx.Client(transport=httpx.MockTransport(handler))


def _fresh_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> DataSettings:
    monkeypatch.chdir(tmp_path)
    return DataSettings(data_root="data")


def _vintage_body(obs_day: str, value: str, realtime_end: str) -> bytes:
    observations = [
        {
            "date": obs_day,
            "value": value,
            "realtime_start": realtime_end,
            "realtime_end": realtime_end,
        }
    ]
    return json.dumps({"observations": observations}).encode("utf-8")


@pytest.mark.parametrize("scenario_id", ["FT-C07-persist-prices-redaction"])
def test_prices_persist_redacts_token(
    scenario_id: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """FT-C07-persist-prices-redaction"""
    settings = _fresh_settings(monkeypatch, tmp_path)
    body = (FIXTURES / "tiingo_spy_one_bar.json").read_bytes()

    with _client_serving(body) as http:
        artifact = fetch_and_persist_prices(
            ("SPY",), date(2024, 1, 30), date(2024, 1, 31),
            secrets=_SECRETS, settings=settings, client=http,
        )

    assert artifact.normalized_path.is_file()
    assert artifact.manifest_path.is_file()
    assert not {"token", "api_key", "Authorization"} & set(artifact.manifest.request_params)
    assert all(finding.severity.value != "error" for finding in artifact.manifest.quality_findings)
    raw_files = list((tmp_path / "data" / "raw").rglob("payload.*"))
    assert len(raw_files) == 1
    assert raw_files[0].read_bytes() == body
    manifest_document = artifact.manifest_path.read_text(encoding="utf-8")
    assert "wire-tiingo-token" not in manifest_document


@pytest.mark.parametrize("scenario_id", ["FT-C08-fx-explicit-gap-persists"])
def test_fx_gap_session_still_persists(
    scenario_id: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """FT-C08-fx-explicit-gap-persists"""
    assert spec_for(Dataset.FX).missing_policy is MissingPolicy.EXPLICIT_GAP
    settings = _fresh_settings(monkeypatch, tmp_path)
    # Fixture covers Tue 2024-01-16 and Thu 2024-01-18; Wed 2024-01-17 is an XNYS session.
    body = (FIXTURES / "fred_dexkous_gap.json").read_bytes()

    with _client_serving(body) as http:
        artifact = fetch_and_persist_fx(
            provider="fred", start=date(2024, 1, 15), end=date(2024, 1, 19),
            secrets=_SECRETS, settings=settings, client=http,
        )
        with pytest.raises(ValueError, match="unknown fx provider"):
            fetch_and_persist_fx(
                provider="unknown", start=date(2024, 1, 15), end=date(2024, 1, 19),
                secrets=_SECRETS, settings=settings, client=http,
            )

    stored = DataStore(settings).read_normalized(artifact, spec_for(Dataset.FX))
    assert stored.height == 2
    assert stored.get_column("usdkrw").null_count() == 1


def test_macro_single_vintage_persists(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """FT-SUP01-macro-vintage-persist-seam"""
    settings = _fresh_settings(monkeypatch, tmp_path)
    body = _vintage_body("2024-01-16", "14.10", "2024-01-17")

    with _client_serving(body) as http:
        artifact = fetch_and_persist_macro(
            "VIXCLS", date(2024, 1, 15), date(2024, 1, 19),
            secrets=_SECRETS, settings=settings, client=http,
        )

    stored = DataStore(settings).read_normalized(artifact, spec_for(Dataset.MACRO))
    assert stored.get_column("series_id").to_list() == ["VIXCLS"]
    assert stored.get_column(AVAILABLE_AT).to_list()[0] == datetime(2024, 1, 17, tzinfo=UTC)


def test_cpi_monthly_persists_with_fixed_lag(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """FT-SUP02-cpi-fixed-lag-persist-seam"""
    settings = _fresh_settings(monkeypatch, tmp_path)
    body = (FIXTURES / "ecos_cpi_monthly.json").read_bytes()

    with _client_serving(body) as http:
        artifact = fetch_and_persist_cpi(
            date(2023, 12, 1), date(2023, 12, 31),
            secrets=_SECRETS, settings=settings, client=http,
        )

    stored = DataStore(settings).read_normalized(artifact, spec_for(Dataset.CPI))
    assert stored.get_column("period_end").to_list() == [date(2023, 12, 31)]
    assert stored.get_column(AVAILABLE_AT).to_list()[0] == datetime(2024, 2, 14, tzinfo=UTC)


@pytest.mark.parametrize("scenario_id", ["WAV2-ING-static-dca"])
def test_WAV2_ING_static_dca(scenario_id: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:  # noqa: N802
    """WAV2-ING-static-dca"""
    settings = _fresh_settings(monkeypatch, tmp_path)
    tiingo_body = (FIXTURES / "tiingo_spy_one_bar.json").read_bytes()
    fred_body = (FIXTURES / "fred_dexkous_gap.json").read_bytes()
    cpi_body = (FIXTURES / "ecos_cpi_monthly.json").read_bytes()

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "tiingo" in url:
            return httpx.Response(200, content=tiingo_body)
        if "fred" in url:
            return httpx.Response(200, content=fred_body)
        if "ecos" in url:
            return httpx.Response(200, content=cpi_body)
        return httpx.Response(404, content=b"{}")

    # Ensure macro/factors/research are not called
    import src.data.fetch as fetch_module

    orig_macro = fetch_module.fetch_and_persist_macro
    orig_factors = fetch_module.fetch_and_persist_factors
    orig_research = fetch_module.fetch_and_persist_research_returns

    def fail_macro(*args, **kwargs):
        raise AssertionError("fetch_and_persist_macro must not be called")

    def fail_factors(*args, **kwargs):
        raise AssertionError("fetch_and_persist_factors must not be called")

    def fail_research(*args, **kwargs):
        raise AssertionError("fetch_and_persist_research_returns must not be called")

    monkeypatch.setattr(fetch_module, "fetch_and_persist_macro", fail_macro)
    monkeypatch.setattr(fetch_module, "fetch_and_persist_factors", fail_factors)
    monkeypatch.setattr(fetch_module, "fetch_and_persist_research_returns", fail_research)

    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        counts = fetch_and_persist_static_dca_datasets(
            start=date(2024, 1, 15),
            end=date(2024, 1, 19),
            tickers=("SPY",),
            fx_provider="fred",
            secrets=_SECRETS,
            settings=settings,
            client=http,
        )
    assert counts["prices"] >= 1
    assert counts["fx"] >= 1
    assert counts["cpi"] >= 1
    # No new MACRO partition
    macro_manifests = list((tmp_path / "data" / "manifests" / "macro").glob("*.json")) if (tmp_path / "data" / "manifests" / "macro").exists() else []
    assert len(macro_manifests) == 0


def test_resolve_price_http_window_skips_when_catalog_covers_end() -> None:
    """test_resolve_price_http_window_skips_when_catalog_covers_end"""
    import polars as pl

    from src.data.schema import TS_DTYPE

    start = date(2024, 1, 15)
    end = date(2024, 1, 31)
    retrieved = datetime(2024, 2, 1, tzinfo=UTC)
    covered = pl.DataFrame(
        {
            "ticker": ["SPY"],
            "date": [end],
            "open": [1.0],
            "high": [1.0],
            "low": [1.0],
            "close": [1.0],
            "volume": [1],
            "adjusted_close": [1.0],
            "dividend": [0.0],
            "split_factor": [1.0],
            "source": ["tiingo"],
            "retrieved_at": [retrieved],
        }
    ).cast({"retrieved_at": TS_DTYPE})
    assert resolve_price_http_window("SPY", covered, start, end) is None

    partial = covered.with_columns(pl.lit(date(2024, 1, 20)).alias("date"))
    window = resolve_price_http_window("SPY", partial, start, end)
    assert window is not None
    effective_start, window_end = window
    assert effective_start > start
    assert effective_start <= end
    assert window_end == end


def test_incremental_prices_skips_http_for_covered_ticker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """test_incremental_prices_skips_http_for_covered_ticker"""
    settings = _fresh_settings(monkeypatch, tmp_path)
    body = (FIXTURES / "tiingo_spy_one_bar.json").read_bytes()
    start = date(2024, 1, 30)
    end = date(2024, 1, 31)
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, content=body)

    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        first = fetch_and_persist_prices(
            ("SPY",), start, end, secrets=_SECRETS, settings=settings, client=http
        )
        prior_rows = first.manifest.row_count
        requests.clear()
        second = fetch_and_persist_prices(
            ("SPY",), start, end, secrets=_SECRETS, settings=settings, client=http, incremental=True
        )

    assert len(requests) == 0
    assert second.manifest.row_count == prior_rows


def test_static_dca_prices_path_uses_incremental(monkeypatch: pytest.MonkeyPatch) -> None:
    """test_static_dca_prices_path_uses_incremental"""
    import src.data.fetch as fetch_module

    prices_kwargs: dict[str, object] = {}

    def prices_spy(*args: object, **kwargs: object) -> object:
        prices_kwargs.update(kwargs)

        class _Art:
            manifest = type("M", (), {"row_count": 1})()

        return _Art()

    monkeypatch.setattr(fetch_module, "fetch_and_persist_prices", prices_spy)
    monkeypatch.setattr(
        fetch_module,
        "fetch_and_persist_fx",
        lambda *args, **kwargs: type("A", (), {"manifest": type("M", (), {"row_count": 1})()})(),
    )
    monkeypatch.setattr(
        fetch_module,
        "fetch_and_persist_cpi",
        lambda *args, **kwargs: type("A", (), {"manifest": type("M", (), {"row_count": 1})()})(),
    )

    fetch_and_persist_static_dca_datasets(
        start=date(2024, 1, 15),
        end=date(2024, 1, 19),
        tickers=("SPY",),
        fx_provider="fred",
        secrets=_SECRETS,
        settings=DataSettings(data_root=Path("data")),
    )
    assert prices_kwargs.get("incremental") is True
