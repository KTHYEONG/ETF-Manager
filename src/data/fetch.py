"""Orchestrate vendor fetch then persist_ingest."""

from __future__ import annotations

import contextlib
import json
import logging
from datetime import date
from typing import TYPE_CHECKING, Final

import httpx
import polars as pl

from src.data.pipeline import persist_ingest
from src.data.providers.base import DEFAULT_TIMEOUT_S, ProviderError
from src.data.providers.ecos import EcosClient
from src.data.providers.fred import FredClient
from src.data.providers.french import FrenchClient
from src.data.providers.tiingo import TiingoClient
from src.data.schema import Dataset
from src.data.secrets import ProviderSecrets
from src.data.settings import DataSettings
from src.data.storage import DatasetArtifact, RawPayload

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

logger = logging.getLogger(__name__)

_FX_PROVIDERS: Final[frozenset[str]] = frozenset({"fred", "ecos"})


def fetch_and_persist_prices(
    tickers: tuple[str, ...],
    start: date,
    end: date,
    *,
    secrets: ProviderSecrets,
    settings: DataSettings,
    client: httpx.Client | None = None,
) -> DatasetArtifact:
    """Fetch Tiingo EOD prices and persist through the ingest seam."""
    with _http(client) as session:
        payload, frame = TiingoClient(secrets.tiingo_api, session).fetch_prices(tickers, start, end)
        artifact = persist_ingest(frame, Dataset.PRICES, payload, settings)
    _log_done("prices", "tiingo", artifact.manifest.row_count)
    return artifact


def fetch_and_persist_fx(
    *,
    provider: str,
    start: date,
    end: date,
    secrets: ProviderSecrets,
    settings: DataSettings,
    client: httpx.Client | None = None,
) -> DatasetArtifact:
    """Fetch USD/KRW from fred or ecos and persist Dataset.FX."""
    if provider not in _FX_PROVIDERS:
        raise ValueError(f"unknown fx provider {provider!r}; expected one of {sorted(_FX_PROVIDERS)}")
    with _http(client) as session:
        if provider == "fred":
            payload, frame = FredClient(secrets.fred_api, session).fetch_fx(start, end)
        else:
            payload, frame = EcosClient(secrets.ecos_api, session).fetch_fx(start, end)
        artifact = persist_ingest(frame, Dataset.FX, payload, settings)
    _log_done("fx", provider, artifact.manifest.row_count)
    return artifact


def fetch_and_persist_macro(
    series_id: str | Sequence[str],
    start: date,
    end: date,
    *,
    secrets: ProviderSecrets,
    settings: DataSettings,
    client: httpx.Client | None = None,
) -> DatasetArtifact:
    """Fetch ALFRED vintage observations and persist Dataset.MACRO as one partition.

    A sequence of series ids fetches each vintage frame over one HTTP session and
    persists their vertical concat as a single MACRO partition; a bare string
    behaves exactly like the historical single-series path.
    """
    series_ids = (series_id,) if isinstance(series_id, str) else tuple(series_id)
    if not series_ids:
        raise ValueError("fetch_and_persist_macro requires at least one series id")
    if len(set(series_ids)) != len(series_ids):
        raise ValueError(f"fetch_and_persist_macro received duplicate series ids: {series_ids!r}")
    with _http(client) as session:
        fred = FredClient(secrets.fred_api, session)
        fetched = [fred.fetch_macro_vintages(sid, start, end) for sid in series_ids]
        if len(fetched) == 1:
            payload, frame = fetched[0]
        else:
            payload, frame = _merge_macro_payloads(series_ids, fetched)
        artifact = persist_ingest(frame, Dataset.MACRO, payload, settings)
    _log_done("macro", "fred", artifact.manifest.row_count)
    return artifact


def _merge_macro_payloads(
    series_ids: tuple[str, ...],
    fetched: list[tuple[RawPayload, pl.DataFrame]],
) -> tuple[RawPayload, pl.DataFrame]:
    """Concat vintage frames and fold raw observation documents into one payload."""
    observations: list[object] = []
    for payload, _frame in fetched:
        document = json.loads(payload.content.decode("utf-8"))
        rows = document.get("observations") if isinstance(document, dict) else None
        if not isinstance(rows, list):
            raise ProviderError(f"fred macro payload for {payload.endpoint!r} lacks an observations list")
        observations.extend(rows)
    merged_content = json.dumps({"observations": observations}).encode("utf-8")
    lineage = {
        "series_ids": ",".join(series_ids),
        "file_type": "json",
        "observation_start": fetched[0][0].request_params.get("observation_start", ""),
        "observation_end": fetched[0][0].request_params.get("observation_end", ""),
        "realtime_start": fetched[0][0].request_params.get("realtime_start", ""),
        "realtime_end": fetched[-1][0].request_params.get("realtime_end", ""),
        "vintage_chunk_days": fetched[0][0].request_params.get("vintage_chunk_days", ""),
    }
    retrieved_at = max(payload.retrieved_at for payload, _frame in fetched)
    payload = RawPayload(
        provider="fred",
        endpoint=f"series/observations/{'+'.join(series_ids)}",
        request_params=lineage,
        retrieved_at=retrieved_at,
        extension="json",
        content=merged_content,
    )
    frame = pl.concat([frame for _payload, frame in fetched], how="vertical")
    return payload, frame


def fetch_and_persist_cpi(
    start: date,
    end: date,
    *,
    secrets: ProviderSecrets,
    settings: DataSettings,
    client: httpx.Client | None = None,
) -> DatasetArtifact:
    """Fetch ECOS CPI and persist Dataset.CPI."""
    with _http(client) as session:
        payload, frame = EcosClient(secrets.ecos_api, session).fetch_cpi(start, end)
        artifact = persist_ingest(frame, Dataset.CPI, payload, settings)
    _log_done("cpi", "ecos", artifact.manifest.row_count)
    return artifact


def fetch_and_persist_factors(
    start: date,
    end: date,
    *,
    settings: DataSettings,
    client: httpx.Client | None = None,
) -> DatasetArtifact:
    """Fetch Ken French 5F+Mom and persist Dataset.FACTORS (no API secrets)."""
    with _http(client) as session:
        payload, frame = FrenchClient(session).fetch_factors(start, end)
        artifact = persist_ingest(frame, Dataset.FACTORS, payload, settings)
    _log_done("factors", "ken_french", artifact.manifest.row_count)
    return artifact


def fetch_and_persist_research_returns(
    start: date,
    end: date,
    *,
    settings: DataSettings,
    client: httpx.Client | None = None,
) -> DatasetArtifact:
    """Fetch Ken French daily market returns and persist Dataset.RESEARCH_RETURNS only."""
    with _http(client) as session:
        payload, frame = FrenchClient(session).fetch_daily_market_returns(start, end)
        artifact = persist_ingest(frame, Dataset.RESEARCH_RETURNS, payload, settings)
    _log_done("research_returns", "ken_french", artifact.manifest.row_count)
    return artifact


def fetch_and_persist_static_dca_datasets(
    *,
    start: date,
    end: date,
    tickers: Sequence[str],
    fx_provider: str,
    secrets: ProviderSecrets,
    settings: DataSettings,
    client: httpx.Client | None = None,
) -> dict[str, int]:
    """Fetch only PRICES, FX, CPI; never macro/factors/research."""
    if fx_provider not in _FX_PROVIDERS:
        raise ValueError(f"unknown fx provider {fx_provider!r}; expected one of {sorted(_FX_PROVIDERS)}")
    if not tickers:
        raise ValueError("tickers must be nonempty")
    prices_art = fetch_and_persist_prices(tuple(tickers), start, end, secrets=secrets, settings=settings, client=client)
    fx_art = fetch_and_persist_fx(provider=fx_provider, start=start, end=end, secrets=secrets, settings=settings, client=client)
    cpi_art = fetch_and_persist_cpi(start, end, secrets=secrets, settings=settings, client=client)
    return {
        "prices": int(prices_art.manifest.row_count),
        "fx": int(fx_art.manifest.row_count),
        "cpi": int(cpi_art.manifest.row_count),
    }


@contextlib.contextmanager
def _http(injected: httpx.Client | None) -> Iterator[httpx.Client]:
    """Pass an injected client through unchanged or open a short-lived default."""
    if injected is not None:
        yield injected
        return
    with httpx.Client(timeout=httpx.Timeout(DEFAULT_TIMEOUT_S)) as owned:
        yield owned


def _log_done(dataset: str, provider: str, row_count: int) -> None:
    logger.info("[DATA] event=fetch_persist dataset=%s provider=%s rows=%d", dataset, provider, row_count)
