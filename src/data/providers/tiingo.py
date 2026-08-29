"""Tiingo end-of-day equity/ETF price client."""

from __future__ import annotations

import logging
import math
import time
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING, Final

import polars as pl

from src.data.providers.base import ProviderError, get_json
from src.data.schema import Dataset, spec_for
from src.data.storage import RawPayload

if TYPE_CHECKING:
    import httpx

    from src.data.providers.base import ProviderResponse
    from src.data.storage import JSONValue

logger = logging.getLogger(__name__)

_BASE_URL: Final[str] = "https://api.tiingo.com/tiingo/daily"
_INTER_TICKER_SLEEP_S: Final[float] = 2.0
_TICKER_429_RETRIES: Final[int] = 4
_REQUIRED_FIELDS: Final[frozenset[str]] = frozenset(
    {"date", "open", "high", "low", "close", "volume", "adjClose", "divCash", "splitFactor"}
)


class TiingoClient:
    """Maps Tiingo daily prices onto Dataset.PRICES without writing storage."""

    def __init__(self, token: str, client: httpx.Client) -> None:
        self._token = token
        self._client = client

    def fetch_prices(self, tickers: tuple[str, ...], start: date, end: date) -> tuple[RawPayload, pl.DataFrame]:
        """GET /tiingo/daily/{ticker}/prices for each ticker; concatenate normalized rows."""
        if not tickers:
            raise ProviderError("tiingo fetch requires at least one ticker")
        spec = spec_for(Dataset.PRICES)
        retrieved_at = datetime.now(UTC)
        bodies: list[bytes] = []
        records: list[dict[str, object]] = []
        for index, ticker in enumerate(tickers):
            if index > 0:
                time.sleep(_INTER_TICKER_SLEEP_S)
            response: ProviderResponse | None = None
            for burst in range(_TICKER_429_RETRIES):
                try:
                    response = get_json(
                        self._client,
                        f"{_BASE_URL}/{ticker}/prices",
                        params={"startDate": start.isoformat(), "endDate": end.isoformat(), "format": "json"},
                        headers={"Authorization": f"Token {self._token}"},
                    )
                    break
                except ProviderError as exc:
                    if "429" not in str(exc) or burst + 1 >= _TICKER_429_RETRIES:
                        raise
                    time.sleep(min(30.0, 2.0**burst))
            if response is None:
                raise ProviderError(f"tiingo rate limit persisted for ticker {ticker}")
            bodies.append(response.content)
            if not isinstance(response.body, list):
                raise ProviderError(f"tiingo payload for {ticker} is not a price array")
            if not response.body:
                raise ProviderError(f"tiingo returned no prices for ticker {ticker}")
            records.extend(_normalize_bar(bar, retrieved_at, ticker) for bar in response.body)
        frame = pl.DataFrame(records).select(*spec.columns).cast(pl.Schema(dict(spec.columns)))
        payload = RawPayload(
            provider="tiingo",
            endpoint=f"daily/{'+'.join(tickers)}/prices",
            request_params={
                "tickers": list(tickers),
                "startDate": start.isoformat(),
                "endDate": end.isoformat(),
                "format": "json",
            },
            retrieved_at=retrieved_at,
            extension="json",
            content=b"\n".join(bodies),
        )
        logger.info(
            "[DATA] event=fetch dataset=prices provider=tiingo tickers=%d rows=%d",
            len(tickers),
            frame.height,
        )
        return payload, frame


def _numeric(value: object, field: str, ticker: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        raise ProviderError(f"tiingo {ticker} field {field!r} is not numeric")
    try:
        return float(value)
    except ValueError as exc:
        raise ProviderError(f"tiingo {ticker} field {field!r} is not numeric") from exc


def _finite(value: object, field: str, ticker: str) -> float:
    number = _numeric(value, field, ticker)
    if not math.isfinite(number):
        raise ProviderError(f"tiingo {ticker} field {field!r} is non-finite")
    return number


def _normalize_bar(bar: JSONValue, retrieved_at: datetime, ticker: str) -> dict[str, object]:
    if not isinstance(bar, dict):
        raise ProviderError("tiingo price row is not a JSON object")
    missing = sorted(_REQUIRED_FIELDS.difference(bar))
    if missing:
        raise ProviderError(f"tiingo bar misses required fields {missing}")
    try:
        day = date.fromisoformat(str(bar["date"])[:10])
    except ValueError as exc:
        raise ProviderError(f"tiingo {ticker} date is malformed") from exc
    # int() truncates toward zero per the volume contract.
    return {
        "ticker": ticker,
        "date": day,
        "open": _finite(bar["open"], "open", ticker),
        "high": _finite(bar["high"], "high", ticker),
        "low": _finite(bar["low"], "low", ticker),
        "close": _finite(bar["close"], "close", ticker),
        "volume": int(_finite(bar["volume"], "volume", ticker)),
        "adjusted_close": _numeric(bar["adjClose"], "adjClose", ticker),
        "dividend": _numeric(bar["divCash"], "divCash", ticker),
        "split_factor": _numeric(bar["splitFactor"], "splitFactor", ticker),
        "source": "tiingo",
        "retrieved_at": retrieved_at,
    }
