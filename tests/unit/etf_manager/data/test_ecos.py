"""Unit tests for the Bank of Korea ECOS client."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import httpx
import polars as pl
import pytest

from src.etf_manager.data.providers.base import ProviderError
from src.etf_manager.data.providers.ecos import EcosClient

FIXTURES = Path(__file__).resolve().parents[3] / "fixtures" / "providers"
_TOKEN = "test-ecos-key"
_WINDOW: tuple[date, date] = (date(2024, 1, 1), date(2024, 1, 31))


def _client_serving(body: bytes) -> tuple[httpx.Client, list[httpx.Request]]:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, content=body)

    return httpx.Client(transport=httpx.MockTransport(handler)), captured


@pytest.mark.parametrize("scenario_id", ["EC-C06-ecos-fx-and-cpi"])
def test_fx_row_maps_to_usdkrw(scenario_id: str) -> None:
    """EC-C06-ecos-fx-and-cpi"""
    body = (FIXTURES / "ecos_fx_usdkrw.json").read_bytes()
    http, captured = _client_serving(body)
    with http:
        payload, frame = EcosClient(_TOKEN, http).fetch_fx(*_WINDOW)

    assert frame.height == 1
    assert frame.get_column("date").to_list() == [date(2024, 1, 16)]
    assert frame.get_column("usdkrw").to_list() == [1350.1]
    assert frame.get_column("source").unique().to_list() == ["ecos"]
    lineage_text = json.dumps(payload.request_params)
    assert _TOKEN not in lineage_text
    assert _TOKEN not in payload.endpoint
    wire_path = captured[0].url.path
    assert "731Y001" in wire_path
    assert "0000001" in wire_path
    assert _TOKEN in wire_path


def test_cpi_monthly_row_maps_to_period_end() -> None:
    """EC-C06-ecos-fx-and-cpi"""
    body = (FIXTURES / "ecos_cpi_monthly.json").read_bytes()
    http, captured = _client_serving(body)
    with http:
        payload, frame = EcosClient(_TOKEN, http).fetch_cpi(*_WINDOW)

    spec_frame_columns = {"period_end", "value", "source", "retrieved_at"}
    assert set(frame.columns) == spec_frame_columns
    assert frame.schema["period_end"] == pl.Date()
    assert frame.get_column("period_end").to_list() == [date(2023, 12, 31)]
    assert frame.get_column("value").to_list() == [112.49]
    assert frame.get_column("source").unique().to_list() == ["ecos"]
    assert captured[0].url.path.count("901Y009") == 1
    assert _TOKEN not in json.dumps(payload.request_params)


def test_error_result_code_raises_provider_error() -> None:
    """EC-C06-ecos-fx-and-cpi"""
    body = (FIXTURES / "ecos_error_result.json").read_bytes()
    http, _captured = _client_serving(body)
    with http, pytest.raises(ProviderError, match="ERROR-600"):
        EcosClient(_TOKEN, http).fetch_fx(*_WINDOW)


def test_monthly_time_and_bad_value_boundaries(tmp_path: Path) -> None:
    """EC-SUP01-period-parsing-boundary"""
    single_row_dict = (
        b'{"StatisticSearch": {"row": '
        b'{"STAT_CODE": "901Y009", "ITEM_CODE1": "0", "TIME": "202402", "DATA_VALUE": "113.1"}}}'
    )
    http, _ = _client_serving(single_row_dict)
    with http:
        _, frame = EcosClient(_TOKEN, http).fetch_cpi(*_WINDOW)
    assert frame.get_column("period_end").to_list() == [date(2024, 2, 29)]

    bad_value = (
        b'{"StatisticSearch": {"row": '
        b'[{"STAT_CODE": "731Y001", "ITEM_CODE1": "0000001", "TIME": "2024-01-xx", "DATA_VALUE": "1350"}]}}'
    )
    http2, _ = _client_serving(bad_value)
    with http2, pytest.raises(ProviderError, match="TIME"):
        EcosClient(_TOKEN, http2).fetch_fx(*_WINDOW)
