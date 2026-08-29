"""Orchestrate vendor fetch then persist_ingest."""

from __future__ import annotations

import contextlib
import json
import logging
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING, Final

import httpx
import polars as pl

from src.data.catalog import latest_artifact
from src.data.nport_ingest import fetch_and_persist_nport_quarter
from src.data.pipeline import persist_ingest
from src.data.providers.base import DEFAULT_TIMEOUT_S, ProviderError
from src.data.providers.ecos import EcosClient
from src.data.providers.fred import FredClient
from src.data.providers.french import FrenchClient
from src.data.providers.quota import TIINGO_QUOTA, PacingGate
from src.data.providers.tiingo import TiingoClient
from src.data.schema import Dataset, spec_for
from src.data.secrets import ProviderSecrets
from src.data.settings import DataSettings
from src.data.storage import DatasetArtifact, DataStore, RawPayload, UntrustedDatasetError

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

logger = logging.getLogger(__name__)

_FX_PROVIDERS: Final[frozenset[str]] = frozenset({"fred", "ecos"})


def resolve_price_http_window(
    ticker: str,
    existing: pl.DataFrame | None,
    start: date,
    end: date,
    *,
    correction_sessions: int = 5,
) -> tuple[date, date] | None:
    """Determine per-ticker HTTP window with correction overlap.

    Returns None when the catalog already covers ``end`` for ``ticker``;
    otherwise returns ``(effective_start, end)`` where ``effective_start`` is
    ``max(start, d_max - L + 1)`` in XNYS sessions with ``L=correction_sessions``.
    Missing or untrusted catalog yields the full ``[start, end]`` window.
    """
    if existing is None or existing.is_empty():
        return (start, end)
    if "ticker" not in existing.columns or "date" not in existing.columns:
        return (start, end)
    filtered = existing.filter(pl.col("ticker") == ticker)
    if filtered.is_empty():
        return (start, end)
    try:
        d_max = filtered.get_column("date").max()
    except Exception:
        return (start, end)
    if d_max is None:
        return (start, end)
    if isinstance(d_max, str):
        try:
            d_max = date.fromisoformat(d_max[:10])
        except ValueError:
            return (start, end)
    # polars date may be python date already
    if not isinstance(d_max, date):
        return (start, end)
    if d_max >= end:
        return None
    if correction_sessions <= 1:
        effective_start = max(start, d_max)
        if effective_start > end:
            return None
        return (effective_start, end)
    # Compute overlap start: go back correction_sessions-1 sessions before d_max
    from src.data.calendar import load_calendar

    cal = load_calendar()
    cur = d_max
    steps = correction_sessions - 1
    for _ in range(steps):
        cur = cur - timedelta(days=1)
        while not cal.is_session(cur):
            cur -= timedelta(days=1)
    effective_start = cur if cur > start else start
    if effective_start > end:
        return None
    return (effective_start, end)


def fetch_and_persist_prices(
    tickers: tuple[str, ...],
    start: date,
    end: date,
    *,
    secrets: ProviderSecrets,
    settings: DataSettings,
    client: httpx.Client | None = None,
    incremental: bool = False,
) -> DatasetArtifact:
    """Fetch Tiingo EOD prices and persist through the ingest seam.

    When ``incremental`` is true each ticker is resolved via
    ``resolve_price_http_window``; tickers whose window is None contribute
    zero HTTP and keep prior catalog rows.
    """
    if not tickers:
        raise ValueError("fetch_and_persist_prices requires at least one ticker")
    with _http(client) as session:
        gate = PacingGate(TIINGO_QUOTA)
        tiingo = TiingoClient(secrets.tiingo_api, session)
        if incremental:
            spec = spec_for(Dataset.PRICES)
            store = DataStore(settings)
            try:
                existing = store.read_normalized(latest_artifact(settings, Dataset.PRICES), spec)
            except UntrustedDatasetError:
                existing = None
            windows: dict[str, tuple[date, date] | None] = {}
            for t in tickers:
                windows[t] = resolve_price_http_window(t, existing, start, end)
            # Collect fetches for windows that are not None
            bodies: list[bytes] = []
            frames: list[pl.DataFrame] = []
            fetched_tickers: list[str] = []
            for ticker in tickers:
                window = windows[ticker]
                if window is None:
                    continue
                w_start, w_end = window
                try:
                    payload, frame = tiingo.fetch_prices((ticker,), w_start, w_end, gate=gate)
                except ProviderError as exc:
                    if "429" not in str(exc):
                        raise
                    logger.warning("[DATA] event=prices_ticker_skipped ticker=%s reason=rate_limit", ticker)
                    continue
                bodies.append(payload.content)
                frames.append(frame)
                fetched_tickers.append(ticker)
            if not frames:
                if existing is not None and all(v is None for v in windows.values()):
                    retrieved_at = datetime.now(UTC)
                    merged = existing.select(*spec.columns).cast(pl.Schema(dict(spec.columns)))
                    payload = RawPayload(
                        provider="tiingo",
                        endpoint=f"daily/{'+'.join(tickers)}/prices",
                        request_params={
                            "tickers": list(tickers),
                            "startDate": start.isoformat(),
                            "endDate": end.isoformat(),
                            "format": "json",
                            "incremental": True,
                        },
                        retrieved_at=retrieved_at,
                        extension="json",
                        content=b"{}",
                    )
                    artifact = persist_ingest(merged, Dataset.PRICES, payload, settings)
                    _log_done("prices", "tiingo", artifact.manifest.row_count)
                    return artifact
                raise ProviderError("tiingo returned no prices for any requested ticker")
            merged = pl.concat(frames, how="vertical")
            if existing is not None:
                if fetched_tickers:
                    kept = existing.select(*spec.columns).filter(~pl.col("ticker").is_in(fetched_tickers))
                    if not kept.is_empty():
                        merged = pl.concat([kept, merged], how="vertical")
                else:
                    # No ticker fetched but we already handled all-None above
                    pass
            merged = merged.select(*spec.columns).cast(pl.Schema(dict(spec.columns)))
            # Use last frame retrieved_at as payload time
            try:
                _raw = frames[-1].get_column("retrieved_at").max()
                retrieved_at_val = _raw if isinstance(_raw, datetime) else datetime.now(UTC)
            except Exception:
                retrieved_at_val = datetime.now(UTC)
            payload = RawPayload(
                provider="tiingo",
                endpoint=f"daily/{'+'.join(tickers)}/prices",
                request_params={
                    "tickers": list(tickers),
                    "startDate": start.isoformat(),
                    "endDate": end.isoformat(),
                    "format": "json",
                    "incremental": True,
                },
                retrieved_at=retrieved_at_val,
                extension="json",
                content=b"\n".join(bodies),
            )
            artifact = persist_ingest(merged, Dataset.PRICES, payload, settings)
        else:
            payload, frame = tiingo.fetch_prices(tickers, start, end, gate=gate)
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
    prices_art = fetch_and_persist_prices(
        tuple(tickers), start, end, secrets=secrets, settings=settings, client=client, incremental=True
    )
    fx_art = fetch_and_persist_fx(
        provider=fx_provider, start=start, end=end, secrets=secrets, settings=settings, client=client
    )
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


# wiring for nport: fetch_and_persist_nport_quarter(
_ = fetch_and_persist_nport_quarter
