# ruff: noqa: S110,SIM102,SIM108,F541,I001,UP035
"""Ingest runners."""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Final

from src.analytics.us_vehicles import history_price_tickers
from src.cli_commands.parser import _UsageError
from src.data.catalog import latest_artifact
from src.data.doctor import SilverCondition, inspect_data
from src.data.fetch import (
    fetch_and_persist_cpi,
    fetch_and_persist_factors,
    fetch_and_persist_fx,
    fetch_and_persist_fx_krw_base,
    fetch_and_persist_macro,
    fetch_and_persist_prices,
    fetch_and_persist_rates,
    fetch_and_persist_research_returns,
    fetch_and_persist_static_dca_datasets,
)
from src.data.etf_metadata_bootstrap import persist_bootstrap_etf_metadata
from src.data.macro_backfill import fetch_and_persist_macro_backfill
from src.data.providers.base import ProviderError
from src.data.schema import Dataset
from src.data.secrets import load_provider_secrets  # noqa: F401
from src.data.settings import DataSettings
from src.data.storage import UntrustedDatasetError

if TYPE_CHECKING:
    import httpx

    from src.data.secrets import ProviderSecrets

logger = logging.getLogger(__name__)

_SMOKE_START: Final[date] = date(2024, 1, 2)
_SMOKE_END: Final[date] = date(2024, 1, 5)
_SMOKE_TICKER: Final[str] = "VT"
_SMOKE_FX_PROVIDER: Final[str] = "fred"
_SMOKE_DATA_ROOT: Final[Path] = Path("scratch/smoke_data")
_HISTORY_FX_PROVIDER: Final[str] = "fred"
_HISTORY_MACRO_SERIES: Final[tuple[str, ...]] = ("VIXCLS", "BAA10Y")
_HISTORY_MACRO_START: Final[date] = date(2012, 6, 1)
_HISTORY_RATE_SERIES: Final[tuple[str, ...]] = ("DTB3",)
_HISTORY_PREFLIGHT_DATASETS: Final[tuple[Dataset, ...]] = (
    Dataset.PRICES,
    Dataset.FX,
    Dataset.CPI,
    Dataset.FACTORS,
    Dataset.MACRO,
    Dataset.RESEARCH_RETURNS,
    Dataset.ETF_METADATA,
    Dataset.FX_KRW_BASE,
    Dataset.RATES,
)
_SMOKE_PREFLIGHT_DATASETS: Final[tuple[Dataset, ...]] = (Dataset.PRICES, Dataset.FX)
_STATIC_DCA_PREFLIGHT_DATASETS: Final[tuple[Dataset, ...]] = (Dataset.PRICES, Dataset.FX, Dataset.CPI)


def _preflight_relevant_silver(settings: DataSettings, datasets: tuple[Dataset, ...]) -> None:
    """Run shared doctor inspection; a damaged relevant prior fails ingest before any fetch."""
    report = inspect_data(settings)
    conditions = {health.dataset: health for health in report.datasets}
    for dataset in datasets:
        health = conditions.get(dataset)
        if health is None:
            continue
        if health.condition in (SilverCondition.DAMAGED_SILVER, SilverCondition.UNRECOVERABLE):
            raise UntrustedDatasetError(
                f"ingest preflight refuses {dataset.value}: latest silver at "
                f"{health.manifest_path.as_posix()} is {health.condition.value}; repair before ingesting"
            )


def run_ingest_smoke(
    *,
    start: date,
    end: date,
    ticker: str,
    fx_provider: str,
    settings: DataSettings,
    secrets: ProviderSecrets,
    client: httpx.Client | None = None,
) -> int:
    """Persist a tiny live window of FX and Tiingo prices; ECOS CPI is optional.

    Returns 0 only when FX and prices persist and their latest catalog
    partitions hold row_count >= 1.
    """
    try:
        _preflight_relevant_silver(settings, _SMOKE_PREFLIGHT_DATASETS)
        fx = fetch_and_persist_fx(
            provider=fx_provider, start=start, end=end, secrets=secrets, settings=settings, client=client
        )
        prices = fetch_and_persist_prices((ticker,), start, end, secrets=secrets, settings=settings, client=client)
        row_counts = {
            str(dataset): latest_artifact(settings, dataset).manifest.row_count
            for dataset in (Dataset.PRICES, Dataset.FX)
        }
    except (ProviderError, ValueError, UntrustedDatasetError, OSError) as exc:
        # Vendor/catalog messages may echo api_key query strings; expose the failure class only.
        logger.error(
            "[DATA] event=smoke_required_failed command=ingest dataset=smoke reason_type=%s repair=%s",
            type(exc).__name__,
            "inspect Silver health via 'maintain data' and repair Bronze/Silver before retrying",
        )
        return 1
    underfilled = sorted(name for name, count in row_counts.items() if count < 1)
    if underfilled:
        logger.error("[DATA] event=smoke_required_failed reason=empty_catalog dataset=%s", ",".join(underfilled))
        return 1
    try:
        cpi = fetch_and_persist_cpi(start, end, secrets=secrets, settings=settings, client=client)
    except (ProviderError, ValueError):
        logger.warning("[DATA] event=smoke_optional_failed provider=ecos")
    else:
        logger.info("[DATA] event=smoke_optional_ok provider=ecos rows=%d", cpi.manifest.row_count)
    logger.info(
        "[DATA] event=smoke_ok ticker=%s price_rows=%d fx_rows=%d",
        ticker,
        prices.manifest.row_count,
        fx.manifest.row_count,
    )
    return 0


def run_ingest_history(
    *,
    start: date,
    end: date,
    tickers: tuple[str, ...] | None = None,
    fx_provider: str,
    settings: DataSettings,
    secrets: ProviderSecrets,
    client: httpx.Client | None = None,
) -> int:
    """Run the existing history ingest after shared data preflight and surface failures.

    Args:
        start: First requested observation date.
        end: Last requested observation date.
        tickers: Optional requested tradable tickers.
        fx_provider: Existing FX vendor selector.
        settings: Data root.
        secrets: Provider credentials, never logged.
        client: Optional injected HTTP transport.

    Returns:
        Zero only after all required dataset updates publish safely.

    Raises:
        UntrustedDatasetError: If a prior required partition is damaged.
    """
    price_tickers = tickers if tickers is not None else history_price_tickers()
    _preflight_relevant_silver(settings, _HISTORY_PREFLIGHT_DATASETS)
    try:
        fx = fetch_and_persist_fx(
            provider=fx_provider, start=start, end=end, secrets=secrets, settings=settings, client=client
        )
        prices = fetch_and_persist_prices(
            price_tickers,
            start,
            end,
            secrets=secrets,
            settings=settings,
            client=client,
            incremental=True,
        )
        cpi = fetch_and_persist_cpi(start, end, secrets=secrets, settings=settings, client=client)
        fx_krw_base = fetch_and_persist_fx_krw_base(start, end, secrets=secrets, settings=settings, client=client)
        rates = fetch_and_persist_rates(
            _HISTORY_RATE_SERIES, start, end, secrets=secrets, settings=settings, client=client
        )
        factors = fetch_and_persist_factors(start, end, settings=settings, client=client)
        macro_start = start if start >= _HISTORY_MACRO_START else _HISTORY_MACRO_START
        macro = fetch_and_persist_macro(
            _HISTORY_MACRO_SERIES, macro_start, end, secrets=secrets, settings=settings, client=client
        )
        research = fetch_and_persist_research_returns(start, end, settings=settings, client=client)
        metadata = persist_bootstrap_etf_metadata(settings)
        row_counts = {
            str(dataset): latest_artifact(settings, dataset).manifest.row_count
            for dataset in (
                Dataset.PRICES,
                Dataset.FX,
                Dataset.CPI,
                Dataset.FACTORS,
                Dataset.MACRO,
                Dataset.RESEARCH_RETURNS,
                Dataset.ETF_METADATA,
                Dataset.FX_KRW_BASE,
                Dataset.RATES,
            )
        }
    except (ProviderError, ValueError, OSError) as exc:
        # Vendor/catalog messages may echo api_key query strings; expose the failure class only.
        logger.error(
            "[DATA] event=history_failed command=ingest dataset=history reason_type=%s repair=%s",
            type(exc).__name__,
            "inspect Silver health via 'maintain data' and repair Bronze/Silver before retrying",
        )
        return 1
    underfilled = sorted(name for name, count in row_counts.items() if count < 1)
    if underfilled:
        logger.error("[DATA] event=history_failed reason=empty_catalog dataset=%s", ",".join(underfilled))
        return 1
    logger.info(
        "[DATA] event=history_ok tickers=%s price_rows=%d fx_rows=%d cpi_rows=%d factor_rows=%d macro_rows=%d research_rows=%d metadata_rows=%d fx_krw_base_rows=%d rates_rows=%d",
        ",".join(price_tickers),
        prices.manifest.row_count,
        fx.manifest.row_count,
        cpi.manifest.row_count,
        factors.manifest.row_count,
        macro.manifest.row_count,
        research.manifest.row_count,
        metadata.manifest.row_count,
        fx_krw_base.manifest.row_count,
        rates.manifest.row_count,
    )
    return 0


def run_ingest_static_dca(
    *,
    start: date,
    end: date,
    tickers: tuple[str, ...] | None,
    fx_provider: str,
    settings: DataSettings,
    secrets: ProviderSecrets,
    client: httpx.Client | None = None,
) -> int:
    """Persist only PRICES, FX, CPI for static DCA; no macro/factors/research."""
    from src.analytics.us_vehicles import diagnostic_price_tickers, research_satellite_tickers

    if tickers is None:
        tickers = tuple(sorted({*diagnostic_price_tickers(), *research_satellite_tickers()}))
    if fx_provider not in ("fred", "ecos"):
        raise ValueError(f"unknown fx provider {fx_provider!r}")
    if "QQQ" in tickers and start < date(1999, 3, 10):
        raise _UsageError(f"static-dca start {start.isoformat()} is before QQQ listing 1999-03-10")
    if start > end:
        raise ValueError(f"start {start.isoformat()} is after end {end.isoformat()}")
    try:
        _preflight_relevant_silver(settings, _STATIC_DCA_PREFLIGHT_DATASETS)
        row_counts = fetch_and_persist_static_dca_datasets(
            start=start, end=end, tickers=tickers, fx_provider=fx_provider, secrets=secrets, settings=settings, client=client
        )
        # Verify catalog row_counts
        catalog_counts = {
            str(dataset): latest_artifact(settings, dataset).manifest.row_count
            for dataset in (Dataset.PRICES, Dataset.FX, Dataset.CPI)
        }
        # Prefer returned counts but ensure catalog also >=1
        merged = {**catalog_counts, **row_counts}
    except _UsageError:
        raise
    except (ProviderError, ValueError, UntrustedDatasetError, OSError) as exc:
        logger.error(
            "[DATA] event=static_dca_failed command=ingest dataset=static-dca reason_type=%s repair=%s",
            type(exc).__name__,
            "inspect Silver health via 'maintain data' and repair Bronze/Silver before retrying",
        )
        return 1
    underfilled = sorted(name for name, count in merged.items() if count < 1)
    if underfilled:
        logger.error("[DATA] event=static_dca_failed reason=empty_catalog dataset=%s", ",".join(underfilled))
        return 1
    logger.info(
        "[DATA] event=static_dca_ok tickers=%s price_rows=%d fx_rows=%d cpi_rows=%d",
        ",".join(tickers),
        merged.get("prices", 0),
        merged.get("fx", 0),
        merged.get("cpi", 0),
    )
    return 0


def run_ingest_macro_backfill(
    *,
    series_ids: tuple[str, ...],
    start: date | None,
    settings: DataSettings,
    secrets: ProviderSecrets,
) -> int:
    """Extend allowlisted unrevised MACRO series before their first ALFRED vintage.

    Returns zero after the new MACRO partition is published. Domain rejections
    (unlisted series, missing first vintage, untrusted prior) propagate so the CLI
    facade reports the actionable reason and exits one.

    Raises:
        _UsageError: If no series id or no start date was given.
    """
    if not series_ids:
        raise _UsageError("ingest macro-backfill requires --series-id")
    if start is None:
        raise _UsageError("ingest macro-backfill requires --start")
    artifact = fetch_and_persist_macro_backfill(series_ids, start, secrets=secrets, settings=settings)
    logger.info(
        "[DATA] event=macro_backfill_ok command=ingest series_ids=%s start=%s rows=%d",
        ",".join(series_ids),
        start.isoformat(),
        artifact.manifest.row_count,
    )
    return 0
