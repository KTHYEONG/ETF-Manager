"""Unit tests for the FRED/ALFRED observation and vintage client."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

import httpx
import polars as pl
import pytest

from src.etf_manager.data.providers.base import ProviderError
from src.etf_manager.data.providers.fred import FredClient

FIXTURES = Path(__file__).resolve().parents[3] / "fixtures" / "providers"
_TOKEN = "test-fred-token"
_WINDOW: tuple[date, date] = (date(2024, 1, 15), date(2024, 1, 19))


def _client_serving(
    body: bytes,
) -> tuple[httpx.Client, list[httpx.Request]]:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, content=body)

    return httpx.Client(transport=httpx.MockTransport(handler)), captured


def test_fx_dot_value_maps_to_null_gap() -> None:
    """FR-C04-fred-fx-null-dot"""
    body = (FIXTURES / "fred_dexkous_gap.json").read_bytes()
    http, captured = _client_serving(body)
    with http:
        payload, frame = FredClient(_TOKEN, http).fetch_fx(*_WINDOW)

    assert frame.height == 2
    assert frame.get_column("usdkrw").to_list() == [1300.5, None]
    assert frame.get_column("date").to_list() == [date(2024, 1, 16), date(2024, 1, 18)]
    assert frame.get_column("source").unique().to_list() == ["fred"]
    assert payload.request_params["series_id"] == "DEXKOUS"
    lineage_text = json.dumps(payload.request_params)
    assert "api_key" not in payload.request_params
    assert _TOKEN not in lineage_text
    wire = captured[0]
    assert wire.url.params["api_key"] == _TOKEN
    assert wire.url.params["series_id"] == "DEXKOUS"
    assert wire.url.params["file_type"] == "json"


@pytest.mark.parametrize("scenario_id", ["FR-C05-fred-vintages"])
def test_vintage_rows_carry_distinct_release_dates(scenario_id: str) -> None:
    """FR-C05-fred-vintages"""
    body = (FIXTURES / "alfred_vixcls_vintages.json").read_bytes()
    http, captured = _client_serving(body)
    with http:
        payload, frame = FredClient(_TOKEN, http).fetch_macro_vintages("VIXCLS", *_WINDOW)

    assert frame.height == 2
    assert frame.get_column("series_id").to_list() == ["VIXCLS", "VIXCLS"]
    assert frame.get_column("observation_date").to_list() == [date(2024, 1, 30), date(2024, 1, 30)]
    assert frame.get_column("value").to_list() == [13.82, 13.9]
    releases = frame.get_column("release_date").to_list()
    assert len(set(releases)) == 2
    assert releases == [
        datetime(2024, 1, 31, tzinfo=UTC),
        datetime(2024, 2, 1, tzinfo=UTC),
    ]
    assert frame.schema["release_date"] == pl.Datetime("us", "UTC")
    params = payload.request_params
    assert {"realtime_start", "realtime_end"} <= set(params)
    assert "api_key" not in params
    wire = captured[0]
    assert wire.url.params["realtime_start"] == "2024-01-15"
    assert wire.url.params["api_key"] == _TOKEN


def test_missing_realtime_end_and_empty_observations_fail_closed() -> None:
    """FR-SUP01-vintage-and-empty-payload-boundary"""
    no_realtime = b'{"observations": [{"date": "2024-01-16", "value": "1300.5"}]}'
    http, _ = _client_serving(no_realtime)
    with http, pytest.raises(ProviderError, match="realtime_end"):
        FredClient(_TOKEN, http).fetch_macro_vintages("VIXCLS", *_WINDOW)

    empty = b'{"observations": []}'
    http2, _ = _client_serving(empty)
    with http2, pytest.raises(ProviderError, match="no observations"):
        FredClient(_TOKEN, http2).fetch_fx(*_WINDOW)
