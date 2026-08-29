"""Unit tests for the Tiingo end-of-day price client."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date
from pathlib import Path

import httpx
import polars as pl
import pytest

from src.data.providers.base import ProviderError
from src.data.providers.tiingo import TiingoClient
from src.data.schema import TS_DTYPE, Dataset, spec_for

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "providers"
_TOKEN = "test-tiingo-token"
_WINDOW: tuple[date, date] = (date(2024, 1, 30), date(2024, 1, 31))


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _spy_body() -> bytes:
    return (FIXTURES / "tiingo_spy_one_bar.json").read_bytes()


def test_tiingo_normalizes_one_bar() -> None:
    """TG-C02-tiingo-normalize"""
    body = _spy_body()
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, content=body)

    with _client(handler) as http:
        payload, frame = TiingoClient(_TOKEN, http).fetch_prices(("SPY",), *_WINDOW)

    spec = spec_for(Dataset.PRICES)
    assert frame.height == 1
    assert set(frame.columns) == set(spec.columns)
    assert frame.schema["volume"] == pl.Int64()
    assert frame.schema["retrieved_at"] == TS_DTYPE
    assert frame.get_column("ticker")[0] == "SPY"
    assert frame.get_column("adjusted_close")[0] == 476.53
    assert frame.get_column("dividend")[0] == 0.0
    assert frame.get_column("split_factor")[0] == 1.0
    assert frame.get_column("volume")[0] == 74_125_631
    assert frame.get_column("source")[0] == "tiingo"
    assert payload.content == body
    request_params_keys = set(payload.request_params)
    assert not request_params_keys & {"Authorization", "token", "api_key"}
    assert len(captured) == 1
    assert captured[0].headers["Authorization"] == f"Token {_TOKEN}"


@pytest.mark.parametrize("scenario_id", ["TG-C03-tiingo-fail-closed"])
def test_http_404_fails_closed_without_retry(scenario_id: str) -> None:
    """TG-C03-tiingo-fail-closed"""
    attempts: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        return httpx.Response(404, content=b'{"detail": "not found"}')

    with _client(handler) as http, pytest.raises(ProviderError, match="404"):
        TiingoClient(_TOKEN, http).fetch_prices(("SPY",), *_WINDOW)

    assert len(attempts) == 1


def test_empty_payload_and_empty_tickers_fail_closed() -> None:
    """TG-C03-tiingo-fail-closed"""
    def empty_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"[]")

    with _client(empty_handler) as http:
        with pytest.raises(ProviderError, match="no prices"):
            TiingoClient(_TOKEN, http).fetch_prices(("SPY",), *_WINDOW)
        with pytest.raises(ProviderError, match="ticker"):
            TiingoClient(_TOKEN, http).fetch_prices((), *_WINDOW)


def test_transient_429_retried_then_success() -> None:
    """TG-SUP01-retry-boundary-429-then-success"""
    body = _spy_body()
    attempts: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        if len(attempts) < 2:
            return httpx.Response(429, content=b"rate limited")
        return httpx.Response(200, content=body)

    with _client(handler) as http:
        _, frame = TiingoClient(_TOKEN, http).fetch_prices(("SPY",), *_WINDOW)

    assert frame.height == 1
    assert len(attempts) == 2


def test_persistent_500_exhausts_three_attempts() -> None:
    """TG-SUP01-retry-boundary-max-attempts"""
    attempts: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        return httpx.Response(500, content=b"boom")

    with _client(handler) as http, pytest.raises(ProviderError, match="500"):
        TiingoClient(_TOKEN, http).fetch_prices(("SPY",), *_WINDOW)

    assert len(attempts) == 5
