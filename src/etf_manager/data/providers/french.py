"""Kenneth French factor file client."""

from __future__ import annotations

import calendar as _calendar
import io
import logging
import zipfile
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING, Final

import httpx
import polars as pl
from tenacity import Retrying, retry_if_exception, stop_after_attempt

from src.etf_manager.data.providers.base import MAX_ATTEMPTS, ProviderError
from src.etf_manager.data.schema import TS_DTYPE, Dataset, spec_for
from src.etf_manager.data.storage import RawPayload

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = logging.getLogger(__name__)

_FIVE_FACTOR_URL: Final[str] = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip"
)
_DAILY_FIVE_FACTOR_URL: Final[str] = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
)
_MOMENTUM_URL: Final[str] = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_CSV.zip"
)
_PROVIDER: Final[str] = "ken_french"
_RESEARCH_SERIES_ID: Final[str] = "us_mkt_ff_daily"
_RESEARCH_LABEL: Final[str] = "research_proxy"
_FACTOR_NAMES: Final[tuple[str, ...]] = ("mkt_rf", "smb", "hml", "rmw", "cma")
_COLUMN_NAMES: Final[tuple[str, ...]] = ("period_end", *_FACTOR_NAMES, "rf")
# Ken French marks unavailable cells with these sentinel percentages.
_MISSING_SENTINELS: Final[frozenset[float]] = frozenset({-99.99, -999.0})


class FrenchClient:
    """Maps Ken French 5F+Mom monthly ZIP files onto Dataset.FACTORS."""

    def __init__(self, client: httpx.Client) -> None:
        self._client = client

    def fetch_factors(self, start: date, end: date) -> tuple[RawPayload, pl.DataFrame]:
        """Download the 5F 2x3 and Momentum monthly ZIPs into Dataset.FACTORS rows.

        Percent values become decimals; the momentum file is left-joined on
        ``period_end`` so months without momentum stay null (EXPLICIT_GAP).

        Raises:
            ProviderError: On HTTP failure or when no monthly row parses.
        """
        retrieved_at = datetime.now(UTC)
        five_bytes = _get_zip(self._client, _FIVE_FACTOR_URL)
        mom_bytes = _get_zip(self._client, _MOMENTUM_URL)
        five_rows = _parse_monthly_rows(_csv_text(five_bytes), len(_COLUMN_NAMES) - 1)
        mom_by_month = {month: values[0] for month, values in _parse_monthly_rows(_csv_text(mom_bytes), 1)}
        records: list[dict[str, object]] = [
            {
                **dict(zip(_COLUMN_NAMES, (month_end, *values), strict=True)),
                "mom": mom_by_month.get(month_end),
            }
            for month_end, values in five_rows
            if start <= month_end <= end
        ]
        spec = spec_for(Dataset.FACTORS)
        frame = (
            pl.DataFrame(records)
            .with_columns(
                pl.lit(_PROVIDER, dtype=pl.String()).alias("source"),
                pl.lit(retrieved_at, dtype=TS_DTYPE).alias("retrieved_at"),
            )
            .select(*spec.columns)
            .cast(pl.Schema(dict(spec.columns)))
        )
        logger.info("[DATA] event=fetch dataset=%s provider=%s rows=%d", str(Dataset.FACTORS), _PROVIDER, frame.height)
        return (
            RawPayload(
                provider=_PROVIDER,
                endpoint="ken_french/monthly_factors",
                request_params={"start": start.isoformat(), "end": end.isoformat()},
                retrieved_at=retrieved_at,
                extension="zip",
                content=five_bytes + b"\n" + mom_bytes,
            ),
            frame,
        )

    def fetch_daily_market_returns(self, start: date, end: date) -> tuple[RawPayload, pl.DataFrame]:
        """Download the daily 5F ZIP into Dataset.RESEARCH_RETURNS rows.

        Percent values become decimals and ``simple_return = mkt_rf + rf``; the
        frame carries the fixed research identity (no ticker column), so it can
        never splice onto PRICES.

        Raises:
            ProviderError: On HTTP failure or when no daily row parses.
        """
        retrieved_at = datetime.now(UTC)
        zip_bytes = _get_zip(self._client, _DAILY_FIVE_FACTOR_URL)
        rows = _parse_daily_rows(_csv_text(zip_bytes), start, end)
        spec = spec_for(Dataset.RESEARCH_RETURNS)
        frame = (
            pl.DataFrame(rows)
            .with_columns(
                pl.lit(_PROVIDER, dtype=pl.String()).alias("source"),
                pl.lit(retrieved_at, dtype=TS_DTYPE).alias("retrieved_at"),
            )
            .select(*spec.columns)
            .cast(pl.Schema(dict(spec.columns)))
        )
        logger.info("[DATA] event=fetch dataset=%s provider=%s rows=%d", str(Dataset.RESEARCH_RETURNS), _PROVIDER, frame.height)
        return (
            RawPayload(
                provider=_PROVIDER,
                endpoint="ken_french/daily_market_returns",
                request_params={"start": start.isoformat(), "end": end.isoformat()},
                retrieved_at=retrieved_at,
                extension="zip",
                content=zip_bytes,
            ),
            frame,
        )


def _get_zip(client: httpx.Client, url: str) -> bytes:
    """GET one ZIP document with the shared vendor retry policy (429/5xx only)."""

    def _attempt() -> httpx.Response:
        response = client.get(url)
        if response.status_code == 429 or response.status_code >= 500:
            # Raised so the surrounding tenacity policy can retry it.
            response.raise_for_status()
        if response.status_code >= 400:
            raise ProviderError(f"provider returned HTTP {response.status_code}")
        return response

    try:
        retrier = Retrying(
            stop=stop_after_attempt(MAX_ATTEMPTS),
            retry=retry_if_exception(lambda exc: isinstance(exc, httpx.HTTPStatusError)),
            reraise=True,
        )
        return retrier(_attempt).content
    except httpx.HTTPStatusError as exc:
        raise ProviderError(f"provider returned HTTP {exc.response.status_code}") from exc
    except httpx.HTTPError as exc:
        raise ProviderError(f"transport failure ({type(exc).__name__})") from exc


def _csv_text(zip_bytes: bytes) -> str:
    """Extract the single CSV member of a Ken French ZIP archive."""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(members) != 1:
            raise ProviderError(f"ken_french zip must hold exactly one CSV member, got {len(members)}")
        return archive.read(members[0]).decode("latin-1")


def _parse_monthly_rows(text: str, value_count: int) -> list[tuple[date, list[float | None]]]:
    """Parse ``YYYYMM`` percent rows into ``(month-end, decimals)`` pairs.

    Header, blank, and annual-section lines are skipped; sentinel percentages
    become ``None``; an empty parse fails closed.
    """
    rows: list[tuple[date, list[float | None]]] = []
    for parts in _numeric_lines(text):
        if len(parts) != value_count + 1:
            continue
        month = parts[0]
        values = [_decimal(raw) for raw in parts[1:]]
        rows.append((_month_end(int(month[:4]), int(month[4:])), values))
    if not rows:
        raise ProviderError("ken_french payload contains no monthly rows")
    return rows


def _parse_daily_rows(text: str, start: date, end: date) -> list[dict[str, object]]:
    """Parse ``YYYYMMDD`` percent rows into research-return records within the window.

    Header, blank, and footer lines are skipped; sentinel percentages yield no
    row; an empty parse fails closed.
    """
    rows: list[dict[str, object]] = []
    for line in text.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3 or not parts[0].isdigit() or len(parts[0]) != 8:
            continue
        try:
            day = date(int(parts[0][:4]), int(parts[0][4:6]), int(parts[0][6:]))
        except ValueError as exc:
            raise ProviderError(f"ken_french daily date {parts[0]!r} is invalid") from exc
        if not (start <= day <= end):
            continue
        mkt_rf = _decimal(parts[1])
        rf = _decimal(parts[-1])
        if mkt_rf is None or rf is None:
            continue
        rows.append(
            {
                "series_id": _RESEARCH_SERIES_ID,
                "date": day,
                "simple_return": mkt_rf + rf,
                "label": _RESEARCH_LABEL,
            }
        )
    if not rows:
        raise ProviderError("ken_french payload contains no daily rows")
    return rows


def _numeric_lines(text: str) -> Iterator[list[str]]:
    for line in text.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 2 and parts[0].isdigit() and len(parts[0]) == 6:
            yield parts


def _decimal(raw: str) -> float | None:
    try:
        value = float(raw)
    except ValueError as exc:
        raise ProviderError(f"ken_french cell {raw!r} is not numeric") from exc
    return None if value in _MISSING_SENTINELS else value / 100.0


def _month_end(year: int, month: int) -> date:
    return date(year, month, _calendar.monthrange(year, month)[1])
