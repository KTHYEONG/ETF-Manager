"""Unit tests for the Kenneth French client."""

from __future__ import annotations

import io
import json
import zipfile
from datetime import date
from pathlib import Path

import httpx
import polars as pl
import pytest

from src.data.providers.base import ProviderError
from src.data.providers.french import FrenchClient

_WINDOW: tuple[date, date] = (date(2010, 1, 1), date(2010, 12, 31))

_FIVE_TEXT = """\
,F-F_Research_Data_5_Factors_2x3

  ,Mkt-RF,SMB,HML,RMW,CMA,RF
  200912,   9.90,  -1.00,   0.50,   0.30,  -0.20,   0.10
  201001,   1.00,   0.20,  -0.40,   0.10,   0.05,   0.02
  201002,  -0.50,  -0.10,   0.30,  -0.05,  -0.02,   0.02

Annual Factors: January-December

    2010,  10.00,  -2.00,   1.00,   2.00,  -1.00,   0.20
"""

_MOM_TEXT = """\
,F-F_Momentum_Factor

  ,Mom
  201001,   2.00
  201002,  -1.00

Momentum: January-December

    2010,   5.00
"""


def _zip_bytes(member: str, text: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(member, text)
    return buffer.getvalue()


def _client_serving(five: bytes, mom: bytes) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        body = mom if "Momentum" in str(request.url) else five
        return httpx.Response(200, content=body)

    return httpx.Client(transport=httpx.MockTransport(handler))


def _fetch(five_text: str, mom_text: str) -> tuple[object, pl.DataFrame]:
    http = _client_serving(_zip_bytes("F-F_Research_Data_5_Factors_2x3.CSV", five_text),
                           _zip_bytes("F-F_Momentum_Factor.CSV", mom_text))
    with http:
        return FrenchClient(http).fetch_factors(*_WINDOW)


@pytest.mark.parametrize("scenario_id", ["FR-H01-french-parse-percent"])
def test_fr_h01_french_parse_percent(scenario_id: str) -> None:
    """FR-H01-french-parse-percent"""
    payload, frame = _fetch(_FIVE_TEXT, _MOM_TEXT)

    assert frame.height == 2
    assert frame.schema["mkt_rf"] == pl.Float64()
    january = frame.filter(pl.col("period_end") == date(2010, 1, 31)).row(0, named=True)
    assert january["period_end"] == date(2010, 1, 31)
    assert january["mkt_rf"] == pytest.approx(0.01)
    assert january["mom"] == pytest.approx(0.02)
    assert frame.get_column("source").unique().to_list() == ["ken_french"]
    assert payload.provider == "ken_french"  # type: ignore[attr-defined]
    assert "api_key" not in json.dumps(payload.request_params)  # type: ignore[attr-defined]


def test_fr_h01_missing_mom_stays_null_and_window_filters() -> None:
    """FR-H01-french-parse-percent"""
    mom_only_january = _MOM_TEXT.replace("  201002,  -1.00\n", "")
    payload, frame = _fetch(_FIVE_TEXT, mom_only_january)

    february = frame.filter(pl.col("period_end") == date(2010, 2, 28)).row(0, named=True)
    assert february["mom"] is None
    assert february["rf"] == pytest.approx(0.0002)
    # 200912 parses but falls outside the requested window.
    assert frame.get_column("period_end").min() == date(2010, 1, 31)
    assert "api_key" not in json.dumps(payload.request_params)  # type: ignore[attr-defined]


def test_fr_h01_empty_parse_raises_provider_error(tmp_path: Path) -> None:
    """FR-H01-french-parse-percent"""
    headers_only = ",F-F_Momentum_Factor\n\n  ,Mom\n\nMomentum: January-December\n\n    2010,   5.00\n"
    http = _client_serving(_zip_bytes("five.CSV", _FIVE_TEXT), _zip_bytes("mom.CSV", headers_only))
    with http, pytest.raises(ProviderError, match="no monthly rows"):
        FrenchClient(http).fetch_factors(*_WINDOW)


_DAILY_WINDOW: tuple[date, date] = (date(2010, 1, 1), date(2010, 12, 31))

_DAILY_TEXT = """\
,F-F_Research_Data_5_Factors_2x3_daily_CSV

  ,Mkt-RF,SMB,HML,RMW,CMA,RF
  20100104,   1.00,   0.20,  -0.40,   0.10,   0.05,   0.20
  20100105,  -0.50,  -0.10,   0.30,  -0.05,  -0.02,   0.20

Missing Data are coded as -99.99 or -999

Copyright 2019 Kenneth R. French
"""


def _daily_client_serving(daily: bytes) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=daily)

    return httpx.Client(transport=httpx.MockTransport(handler))


@pytest.mark.parametrize("scenario_id", ["FR-C-daily-market-parse"])
def test_fr_c_daily_market_parse(scenario_id: str) -> None:
    """FR-C-daily-market-parse"""
    http = _daily_client_serving(_zip_bytes("F-F_Research_Data_5_Factors_2x3_daily.CSV", _DAILY_TEXT))
    with http:
        payload, frame = FrenchClient(http).fetch_daily_market_returns(*_DAILY_WINDOW)

    assert frame.height == 2
    assert "ticker" not in frame.columns
    january = frame.filter(pl.col("date") == date(2010, 1, 4)).row(0, named=True)
    # Percent row Mkt-RF=1.00 RF=0.20 becomes decimal mkt_rf + rf = 0.01 + 0.002.
    assert january["simple_return"] == pytest.approx(0.012)
    assert january["series_id"] == "us_mkt_ff_daily"
    assert january["label"] == "research_proxy"
    assert frame.get_column("source").unique().to_list() == ["ken_french"]
    assert payload.provider == "ken_french"  # type: ignore[attr-defined]
    assert "api_key" not in json.dumps(payload.request_params)  # type: ignore[attr-defined]


@pytest.mark.parametrize("scenario_id", ["FR-C-daily-market-parse"])
def test_fr_c_daily_empty_parse_raises_provider_error(scenario_id: str) -> None:
    """FR-C-daily-market-parse"""
    headers_only = ",F-F_Research_Data_5_Factors_2x3_daily_CSV\n\n  ,Mkt-RF,SMB,HML,RMW,CMA,RF\n"
    http = _daily_client_serving(_zip_bytes("daily.CSV", headers_only))
    with http, pytest.raises(ProviderError, match="no daily rows"):
        FrenchClient(http).fetch_daily_market_returns(*_DAILY_WINDOW)
