"""Korean pension ETF market ingest: identities and canonical KRX history import."""

from __future__ import annotations

import io
import json
import logging
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import polars as pl

from src.data.catalog import latest_artifact
from src.data.pipeline import persist_ingest
from src.data.pit import AVAILABLE_AT
from src.data.schema import Dataset, spec_for
from src.data.settings import DataSettings
from src.data.storage import DatasetArtifact, DataStore, RawPayload, UntrustedDatasetError

logger = logging.getLogger(__name__)

_CSV_COLUMNS: tuple[str, ...] = (
    "ticker",
    "date",
    "close_krw",
    "nav_krw",
    "distribution_krw",
    "distribution_pay_date",
    "split_factor",
    "volume",
)

_EXPECTED_PROXY: dict[str, str] = {"379800": "SPY", "379810": "QQQ", "469060": "SOXX"}
_EXPECTED_BENCHMARK: dict[str, str] = {
    "379800": "SP500",
    "379810": "NASDAQ100",
    "469060": "NYSE_SEMICONDUCTOR",
}
_RECORD_KEYS: frozenset[str] = frozenset(
    {
        "ticker",
        "listing_date",
        "benchmark_id",
        "proxy_ticker",
        "pension_eligible",
        "currency_hedged",
        "source_url",
        "source_checked_date",
    }
)
_SOXX_BREAK_DATE = date(2021, 6, 21)


@dataclass(frozen=True, slots=True)
class PensionEtfIdentity:
    """Source-backed Korean ETF identity used to enforce listing and proxy boundaries."""

    ticker: str
    listing_date: date
    benchmark_id: str
    proxy_ticker: str
    pension_eligible: bool
    currency_hedged: bool
    source_url: str
    source_checked_date: date


def _parse_identity_record(record: object) -> PensionEtfIdentity:
    if not isinstance(record, dict):
        raise ValueError(f"pension ETF identity record must be an object, got {type(record).__name__}")
    typed: dict[str, object] = dict(record)
    if set(typed.keys()) != set(_RECORD_KEYS):
        raise ValueError(f"pension ETF identity record must carry exactly keys {sorted(_RECORD_KEYS)}")
    ticker = typed["ticker"]
    benchmark_id = typed["benchmark_id"]
    proxy_ticker = typed["proxy_ticker"]
    source_url = typed["source_url"]
    if not isinstance(ticker, str) or not ticker:
        raise ValueError("pension ETF identity ticker must be a non-empty string")
    if ticker not in _EXPECTED_PROXY:
        raise ValueError(f"unsupported pension ETF ticker {ticker!r}")
    if proxy_ticker != _EXPECTED_PROXY[ticker]:
        raise ValueError(f"inconsistent proxy {proxy_ticker!r} for ticker {ticker!r}")
    if benchmark_id != _EXPECTED_BENCHMARK[ticker]:
        raise ValueError(f"inconsistent benchmark {benchmark_id!r} for ticker {ticker!r}")
    if typed["pension_eligible"] is not True:
        raise ValueError(f"pension ETF {ticker!r} must be pension eligible")
    if typed["currency_hedged"] is not False:
        raise ValueError(f"pension ETF {ticker!r} must be unhedged")
    try:
        listing_date = date.fromisoformat(str(typed["listing_date"]))
    except ValueError as exc:
        raise ValueError(f"pension ETF {ticker!r} has invalid listing_date: {exc}") from exc
    try:
        checked = date.fromisoformat(str(typed["source_checked_date"]))
    except ValueError as exc:
        raise ValueError(f"pension ETF {ticker!r} has invalid source_checked_date: {exc}") from exc
    if not isinstance(source_url, str) or not source_url.startswith("http"):
        raise ValueError(f"pension ETF {ticker!r} is undocumented without an http source_url")
    return PensionEtfIdentity(
        ticker=ticker,
        listing_date=listing_date,
        benchmark_id=str(benchmark_id),
        proxy_ticker=str(proxy_ticker),
        pension_eligible=True,
        currency_hedged=False,
        source_url=source_url,
        source_checked_date=checked,
    )


def load_pension_etf_identities(path: str | Path) -> tuple[PensionEtfIdentity, ...]:
    """Load dated, source-backed fund identities for pension research.

    Returns: Distinct eligible Korean-listed ETF identities.
    Raises: ValueError on unsupported, duplicated, hedged, undocumented, or inconsistent records.
    """
    source = Path(path)
    try:
        document: object = json.loads(source.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"pension ETF identity file unreadable at {source}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"pension ETF identity file is not valid JSON at {source}: {exc}") from exc
    if isinstance(document, dict):
        funds_raw: object = document.get("funds", document.get("etfs", document.get("records")))
        if funds_raw is None:
            raise ValueError(f"pension ETF identity file at {source} must carry a 'funds' array")
        caveats_raw: object = document.get("index_caveats", [])
        if not isinstance(caveats_raw, list) or not any(
            isinstance(item, dict)
            and item.get("proxy_ticker") == "SOXX"
            and str(item.get("break_date")) == _SOXX_BREAK_DATE.isoformat()
            for item in caveats_raw
        ):
            raise ValueError("pension ETF identity file must record the dated SOXX index-break caveat")
    elif isinstance(document, list):
        funds_raw = document
    else:
        raise ValueError(f"pension ETF identity file at {source} must be an object or array")
    if not isinstance(funds_raw, list) or len(funds_raw) != 3:
        raise ValueError(f"pension ETF identity file at {source} must carry exactly three fund records")
    parsed = tuple(_parse_identity_record(item) for item in funds_raw)
    tickers = [identity.ticker for identity in parsed]
    if len(set(tickers)) != len(tickers):
        raise ValueError(f"duplicated pension ETF ticker in {source}")
    return tuple(sorted(parsed, key=lambda item: item.ticker))


def _read_canonical_csv(content: bytes) -> pl.DataFrame:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"pension ETF CSV must be UTF-8: {exc}") from exc
    if "\x00" in text:
        raise ValueError("pension ETF CSV contains null bytes")
    try:
        frame = pl.read_csv(io.BytesIO(content), null_values=[""], try_parse_dates=False)
    except Exception as exc:  # pragma: no cover - polars rarely rejects valid UTF-8 bytes
        raise ValueError(f"pension ETF CSV is unreadable: {exc}") from exc
    if list(frame.columns) != list(_CSV_COLUMNS):
        raise ValueError(f"pension ETF CSV must carry exactly columns {list(_CSV_COLUMNS)}")
    if frame.is_empty():
        raise ValueError("pension ETF CSV must be a non-empty canonical export")
    try:
        parsed = frame.with_columns(
            pl.col("ticker").cast(pl.String, strict=True),
            pl.col("date").cast(pl.String, strict=True).str.to_date(strict=True),
            pl.col("close_krw").cast(pl.Float64, strict=True),
            pl.col("nav_krw").cast(pl.Float64, strict=True),
            pl.col("distribution_krw").cast(pl.Float64, strict=True),
            pl.col("distribution_pay_date").cast(pl.String, strict=False).str.to_date(strict=True),
            pl.col("split_factor").cast(pl.Float64, strict=True),
            pl.col("volume").cast(pl.Int64, strict=True),
        )
    except Exception as exc:
        raise ValueError(f"pension ETF CSV carries malformed market values: {exc}") from exc
    return parsed


def _enforce_identity_bounds(frame: pl.DataFrame, identities: tuple[PensionEtfIdentity, ...]) -> None:
    by_ticker = {identity.ticker: identity for identity in identities}
    unknown = sorted(set(frame.get_column("ticker").to_list()) - set(by_ticker))
    if unknown:
        raise ValueError(f"pension ETF CSV carries unknown tickers {unknown}")
    if frame.get_column("distribution_krw").null_count() > 0:
        raise ValueError("pension ETF CSV carries a missing distribution amount; nulls are never zero-filled")
    rows = frame.select("ticker", "date", "distribution_krw", "distribution_pay_date").to_dicts()
    for row in rows:
        identity = by_ticker[str(row["ticker"])]
        day = row["date"]
        assert isinstance(day, date)
        if day < identity.listing_date:
            raise ValueError(
                f"pension ETF {identity.ticker!r} observation {day.isoformat()} precedes listing {identity.listing_date.isoformat()}"
            )
        dist = row["distribution_krw"]
        pay = row["distribution_pay_date"]
        assert isinstance(dist, float)
        if dist > 0:
            if pay is None:
                raise ValueError(f"pension ETF {identity.ticker!r} positive distribution on {day.isoformat()} needs a pay date")
            assert isinstance(pay, date)
            if pay < day:
                raise ValueError(f"pension ETF {identity.ticker!r} pay date {pay.isoformat()} precedes observation {day.isoformat()}")
        elif dist == 0:
            if pay is not None:
                raise ValueError(f"pension ETF {identity.ticker!r} zero distribution on {day.isoformat()} must have a null pay date")
        else:
            raise ValueError(f"pension ETF {identity.ticker!r} negative distribution on {day.isoformat()}")


def _reject_conflicting_revision(frame: pl.DataFrame, settings: DataSettings) -> None:
    try:
        prior_artifact = latest_artifact(settings, Dataset.KR_ETF_PRICES)
    except UntrustedDatasetError:
        return
    prior = DataStore(settings).read_normalized(prior_artifact, spec_for(Dataset.KR_ETF_PRICES))
    if AVAILABLE_AT in prior.columns:
        prior = prior.drop(AVAILABLE_AT)
    market_cols = list(_CSV_COLUMNS)
    prior_keys = set(zip(prior.get_column("ticker").to_list(), prior.get_column("date").to_list(), strict=True))
    new_keys = set(zip(frame.get_column("ticker").to_list(), frame.get_column("date").to_list(), strict=True))
    if prior_keys - new_keys:
        raise ValueError(
            f"refusing to shrink {Dataset.KR_ETF_PRICES!s} history: {len(prior_keys - new_keys)} prior row(s) missing"
        )
    prior_map = dict(zip(prior_keys, prior.select(market_cols).to_dicts(), strict=True))
    new_map = dict(zip(new_keys, frame.select(market_cols).to_dicts(), strict=True))
    for key in sorted(prior_keys & new_keys):
        if prior_map[key] != new_map[key]:
            raise ValueError(f"conflicting revision for {Dataset.KR_ETF_PRICES!s} key {key!r}: refusing silent replacement")


def import_pension_etf_history(
    path: str | Path,
    settings: DataSettings,
    *,
    source_urls: tuple[str, ...],
    retrieved_at: datetime,
    identities: tuple[PensionEtfIdentity, ...],
) -> DatasetArtifact:
    """Archive and import a canonical KRX/issuer-reconciled CSV without inventing missing corporate actions.

    Args: path is the immutable source CSV; source_urls identify its official origins;
        retrieved_at is an aware capture time; identities constrain investable dates.
    Returns: A manifest-bound Korean ETF price partition.
    Raises: ValueError or DataQualityError for format, lineage, date, or price defects.
    """
    if not source_urls:
        raise ValueError("source_urls must carry at least one official origin")
    for url in source_urls:
        if not isinstance(url, str) or not url.startswith("http"):
            raise ValueError(f"source_urls must be http(s) origins, got {url!r}")
    if retrieved_at.tzinfo is None or retrieved_at.utcoffset() is None:
        raise ValueError("retrieved_at must be timezone-aware")
    if not identities:
        raise ValueError("identities must carry the loaded pension ETF identities")
    csv_path = Path(path)
    try:
        content = csv_path.read_bytes()
    except OSError as exc:
        raise ValueError(f"pension ETF CSV unreadable at {csv_path}: {exc}") from exc
    if not content:
        raise ValueError(f"pension ETF CSV is empty at {csv_path}")
    parsed = _read_canonical_csv(content)
    _enforce_identity_bounds(parsed, identities)
    spec = spec_for(Dataset.KR_ETF_PRICES)
    source_label = ",".join(source_urls)
    frame = parsed.with_columns(
        pl.lit(source_label, dtype=pl.String).alias("source"),
        pl.lit(retrieved_at, dtype=pl.Datetime("us", "UTC")).alias("retrieved_at"),
    ).select(list(spec.columns))
    _reject_conflicting_revision(frame, settings)
    payload = RawPayload(
        provider="krx",
        endpoint=source_urls[0],
        request_params={"source_urls": list(source_urls), "filename": csv_path.name},
        retrieved_at=retrieved_at,
        extension="csv",
        content=content,
    )
    artifact = persist_ingest(frame, Dataset.KR_ETF_PRICES, payload, settings, calendar_name="XKRX")
    logger.info("[DATA] event=pension_etf_import rows=%d source=%s", artifact.manifest.row_count, source_label)
    return artifact
