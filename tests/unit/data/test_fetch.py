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
