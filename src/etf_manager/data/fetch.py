"""Orchestrate vendor fetch then persist_ingest."""

from __future__ import annotations

import contextlib
import logging
from datetime import date
from typing import TYPE_CHECKING, Final

import httpx

from src.etf_manager.data.pipeline import persist_ingest
from src.etf_manager.data.providers.base import DEFAULT_TIMEOUT_S
from src.etf_manager.data.providers.ecos import EcosClient
from src.etf_manager.data.providers.fred import FredClient
from src.etf_manager.data.providers.french import FrenchClient
from src.etf_manager.data.providers.tiingo import TiingoClient
from src.etf_manager.data.schema import Dataset
from src.etf_manager.data.secrets import ProviderSecrets
from src.etf_manager.data.settings import DataSettings
from src.etf_manager.data.storage import DatasetArtifact

if TYPE_CHECKING:
    from collections.abc import Iterator

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
    series_id: str,
    start: date,
    end: date,
    *,
    secrets: ProviderSecrets,
    settings: DataSettings,
    client: httpx.Client | None = None,
) -> DatasetArtifact:
    """Fetch ALFRED vintage observations and persist Dataset.MACRO."""
    with _http(client) as session:
        payload, frame = FredClient(secrets.fred_api, session).fetch_macro_vintages(series_id, start, end)
        artifact = persist_ingest(frame, Dataset.MACRO, payload, settings)
    _log_done("macro", "fred", artifact.manifest.row_count)
    return artifact


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
