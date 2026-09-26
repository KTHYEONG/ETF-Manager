"""Kenneth French factor file client."""

from __future__ import annotations

import calendar as _calendar
import io
import logging
import zipfile
from datetime import UTC, date, datetime
from itertools import pairwise as _pairwise
from typing import TYPE_CHECKING, Final

import httpx
import polars as pl
from tenacity import Retrying, retry_if_exception, stop_after_attempt

from src.data.providers.base import MAX_ATTEMPTS, ProviderError
from src.data.schema import TS_DTYPE, Dataset, spec_for
from src.data.storage import RawPayload

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = logging.getLogger(__name__)

_KEN_FRENCH_BASE_URL: Final[str] = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
_FACTOR_FILE_NAME: Final[str] = "F-F_Research_Data_Factors_CSV.zip"
_INDUSTRY_FILE_NAME: Final[str] = "10_Industry_Portfolios_CSV.zip"
_FIVE_FACTOR_URL: Final[str] = (
    f"{_KEN_FRENCH_BASE_URL}F-F_Research_Data_5_Factors_2x3_CSV.zip"
)
_DAILY_FIVE_FACTOR_URL: Final[str] = (
    f"{_KEN_FRENCH_BASE_URL}F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
)
_MOMENTUM_URL: Final[str] = f"{_KEN_FRENCH_BASE_URL}F-F_Momentum_Factor_CSV.zip"
_MONTHLY_FACTOR_URL: Final[str] = f"{_KEN_FRENCH_BASE_URL}{_FACTOR_FILE_NAME}"
_INDUSTRY_URL: Final[str] = f"{_KEN_FRENCH_BASE_URL}{_INDUSTRY_FILE_NAME}"
_PROVIDER: Final[str] = "ken_french"
_RESEARCH_SERIES_ID: Final[str] = "us_mkt_ff_daily"
_RESEARCH_LABEL: Final[str] = "research_proxy"
_MONTHLY_MARKET_SERIES_ID: Final[str] = "ff_mkt_monthly"
_MONTHLY_HITEC_SERIES_ID: Final[str] = "ff_hitec_monthly"
_MONTHLY_DIVIDEND_HI30_SERIES_ID: Final[str] = "ff_dp_hi30_monthly"
_MONTHLY_DEV_EX_US_SERIES_ID: Final[str] = "ff_dev_ex_us_mkt_monthly"
_MONTHLY_EMERGING_SERIES_ID: Final[str] = "ff_em_mkt_monthly"
_DIVIDEND_FILE_NAME: Final[str] = "Portfolios_Formed_on_D-P_CSV.zip"
_DEV_EX_US_FILE_NAME: Final[str] = "Developed_ex_US_3_Factors_CSV.zip"
_EMERGING_FILE_NAME: Final[str] = "Emerging_5_Factors_CSV.zip"
_DIVIDEND_URL: Final[str] = f"{_KEN_FRENCH_BASE_URL}{_DIVIDEND_FILE_NAME}"
_DEV_EX_US_URL: Final[str] = f"{_KEN_FRENCH_BASE_URL}{_DEV_EX_US_FILE_NAME}"
_EMERGING_URL: Final[str] = f"{_KEN_FRENCH_BASE_URL}{_EMERGING_FILE_NAME}"
_FACTOR_NAMES: Final[tuple[str, ...]] = ("mkt_rf", "smb", "hml", "rmw", "cma")
_COLUMN_NAMES: Final[tuple[str, ...]] = ("period_end", *_FACTOR_NAMES, "rf")
# Ken French marks unavailable cells with these sentinel percentages.
_MISSING_SENTINELS: Final[frozenset[float]] = frozenset({-99.99, -999.0})


class FrenchClient:
    """Map Ken French archives onto registered research datasets."""

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

    def fetch_monthly_research_returns(self, start: date, end: date) -> tuple[RawPayload, pl.DataFrame]:
        """Download monthly research proxies used by the pension decision tiers.

        Series: ``ff_mkt_monthly`` (Mkt-RF plus RF) and ``ff_hitec_monthly`` (value-weighted
        HiTec industry) share one month set; ``ff_dp_hi30_monthly`` is the value-weighted top
        30% dividend-yield portfolio (a century proxy for a US dividend sleeve);
        ``ff_dev_ex_us_mkt_monthly`` and ``ff_em_mkt_monthly`` are USD developed ex-US and
        emerging market returns (Mkt-RF plus RF) that start around 1990 and only proxy a world
        sleeve before its ETF inception. Each extra series keeps its own coverage because the
        source files start in different years; none is forward-filled or back-extended.

        Args:
            start: First month-end to keep.
            end: Last month-end to keep.

        Returns:
            Raw payload and rows for all five series.

        Raises:
            ProviderError: On HTTP failure, a missing expected section or column, a sentinel in a
                used column, mismatched month sets between the market and HiTec series, a gap inside
                any series' coverage, or an extra series with no row in the window.
        """
        retrieved_at = datetime.now(UTC)
        factor_bytes = _get_zip(self._client, _MONTHLY_FACTOR_URL)
        industry_bytes = _get_zip(self._client, _INDUSTRY_URL)
        dividend_bytes = _get_zip(self._client, _DIVIDEND_URL)
        dev_ex_us_bytes = _get_zip(self._client, _DEV_EX_US_URL)
        emerging_bytes = _get_zip(self._client, _EMERGING_URL)
        market_by_month = _parse_named_monthly_block(
            _csv_text(factor_bytes),
            ("Mkt-RF", "RF"),
            section_title=None,
            stop_title="Annual Factors",
        )
        hitec_by_month = _parse_named_monthly_block(
            _csv_text(industry_bytes),
            ("HiTec",),
            section_title="Average Value Weighted Returns -- Monthly",
            stop_title="Average Equal Weighted Returns -- Monthly",
        )
        market_months = {month for month in market_by_month if start <= month <= end}
        hitec_months = {month for month in hitec_by_month if start <= month <= end}
        if market_months != hitec_months:
            market_only = sorted(market_months - hitec_months)
            hitec_only = sorted(hitec_months - market_months)
            raise ProviderError(
                "ken_french monthly research month sets differ: "
                f"market_only={market_only[:3]} hitec_only={hitec_only[:3]}"
            )
        dividend_by_month = _parse_named_monthly_block(
            _csv_text(dividend_bytes),
            ("Hi 30",),
            section_title="Value Weight Returns -- Monthly",
            stop_title="Equal Weight Returns -- Monthly",
        )
        dev_ex_us_by_month = _parse_named_monthly_block(
            _csv_text(dev_ex_us_bytes),
            ("Mkt-RF", "RF"),
            section_title=None,
            stop_title="Annual Factors",
        )
        emerging_by_month = _parse_named_monthly_block(
            _csv_text(emerging_bytes),
            ("Mkt-RF", "RF"),
            section_title=None,
            stop_title="Annual Factors",
        )
        dividend_months = _windowed_contiguous_months(
            _MONTHLY_DIVIDEND_HI30_SERIES_ID, dividend_by_month, start, end
        )
        dev_ex_us_months = _windowed_contiguous_months(
            _MONTHLY_DEV_EX_US_SERIES_ID, dev_ex_us_by_month, start, end
        )
        emerging_months = _windowed_contiguous_months(
            _MONTHLY_EMERGING_SERIES_ID, emerging_by_month, start, end
        )
        records: list[dict[str, object]] = []
        for month_end in sorted(market_months):
            market = market_by_month[month_end]
            records.extend(
                (
                    {
                        "series_id": _MONTHLY_MARKET_SERIES_ID,
                        "period_end": month_end,
                        "simple_return": market["Mkt-RF"] + market["RF"],
                        "label": _RESEARCH_LABEL,
                    },
                    {
                        "series_id": _MONTHLY_HITEC_SERIES_ID,
                        "period_end": month_end,
                        "simple_return": hitec_by_month[month_end]["HiTec"],
                        "label": _RESEARCH_LABEL,
                    },
                )
            )
        records.extend(
            {
                "series_id": _MONTHLY_DIVIDEND_HI30_SERIES_ID,
                "period_end": month_end,
                "simple_return": dividend_by_month[month_end]["Hi 30"],
                "label": _RESEARCH_LABEL,
            }
            for month_end in dividend_months
        )
        records.extend(
            {
                "series_id": _MONTHLY_DEV_EX_US_SERIES_ID,
                "period_end": month_end,
                "simple_return": dev_ex_us_by_month[month_end]["Mkt-RF"]
                + dev_ex_us_by_month[month_end]["RF"],
                "label": _RESEARCH_LABEL,
            }
            for month_end in dev_ex_us_months
        )
        records.extend(
            {
                "series_id": _MONTHLY_EMERGING_SERIES_ID,
                "period_end": month_end,
                "simple_return": emerging_by_month[month_end]["Mkt-RF"]
                + emerging_by_month[month_end]["RF"],
                "label": _RESEARCH_LABEL,
            }
            for month_end in emerging_months
        )
        if not records:
            raise ProviderError("ken_french payload contains no monthly rows in the requested window")
        records.sort(key=lambda record: (str(record["series_id"]), str(record["period_end"])))
        spec = spec_for(Dataset.RESEARCH_MONTHLY)
        frame = (
            pl.DataFrame(records)
            .with_columns(
                pl.lit(_PROVIDER, dtype=pl.String()).alias("source"),
                pl.lit(retrieved_at, dtype=TS_DTYPE).alias("retrieved_at"),
            )
            .select(*spec.columns)
            .cast(pl.Schema(dict(spec.columns)))
        )
        logger.info(
            "[DATA] event=fetch dataset=%s provider=%s rows=%d",
            str(Dataset.RESEARCH_MONTHLY),
            _PROVIDER,
            frame.height,
        )
        return (
            RawPayload(
                provider=_PROVIDER,
                endpoint="ken_french/monthly_research_returns",
                request_params={
                    "factor_file": _FACTOR_FILE_NAME,
                    "industry_file": _INDUSTRY_FILE_NAME,
                    "dividend_file": _DIVIDEND_FILE_NAME,
                    "dev_ex_us_file": _DEV_EX_US_FILE_NAME,
                    "emerging_file": _EMERGING_FILE_NAME,
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                },
                retrieved_at=retrieved_at,
                extension="zip",
                content=factor_bytes + b"\n" + industry_bytes + b"\n" + dividend_bytes + b"\n" + dev_ex_us_bytes + b"\n" + emerging_bytes,
            ),
            frame,
        )


def _windowed_contiguous_months(
    series_id: str,
    by_month: dict[date, dict[str, float]],
    start: date,
    end: date,
) -> list[date]:
    """Filter an extra series to ``[start, end]`` and require month contiguity."""
    months = sorted(month for month in by_month if start <= month <= end)
    if not months:
        raise ProviderError(
            f"ken_french {series_id} payload contains no monthly rows in the requested window "
            f"start={start.isoformat()} end={end.isoformat()}"
        )
    for current, nxt in _pairwise(months):
        expected = _next_month_end(current)
        if nxt != expected:
            raise ProviderError(
                f"ken_french {series_id} has a gap: missing month {expected.isoformat()}"
            )
    return months


def _next_month_end(month_end: date) -> date:
    """Return the month-end following ``month_end``."""
    year = month_end.year + (1 if month_end.month == 12 else 0)
    month = 1 if month_end.month == 12 else month_end.month + 1
    return date(year, month, _calendar.monthrange(year, month)[1])


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


def _parse_named_monthly_block(
    text: str,
    required_columns: tuple[str, ...],
    *,
    section_title: str | None,
    stop_title: str,
) -> dict[date, dict[str, float]]:
    lines = text.splitlines()
    start = 0
    if section_title is not None:
        section_index = next(
            (index for index, line in enumerate(lines) if section_title in line),
            None,
        )
        if section_index is None:
            raise ProviderError(f"ken_french payload is missing section {section_title!r}")
        start = section_index + 1
    stop_index = next(
        (index for index in range(start, len(lines)) if stop_title in lines[index]),
        len(lines),
    )
    header_index = next(
        (
            index
            for index in range(start, stop_index)
            if set(required_columns).issubset(_csv_parts(lines[index]))
        ),
        None,
    )
    if header_index is None:
        raise ProviderError(f"ken_french payload is missing expected columns {required_columns!r}")
    header = _csv_parts(lines[header_index])
    column_indices = {column: header.index(column) for column in required_columns}
    values_by_month: dict[date, dict[str, float]] = {}
    for line in lines[header_index + 1 : stop_index]:
        parts = _csv_parts(line)
        if len(parts) < 2 or len(parts[0]) != 6 or not parts[0].isdigit():
            if values_by_month:
                break
            continue
        month_end = _month_end(int(parts[0][:4]), int(parts[0][4:]))
        values_by_month[month_end] = {
            column: _required_decimal(parts[column_index], period_end=month_end, column=column)
            for column, column_index in column_indices.items()
        }
    if not values_by_month:
        raise ProviderError("ken_french payload contains no monthly rows")
    return values_by_month


def _required_decimal(raw: str, *, period_end: date, column: str) -> float:
    value = _decimal(raw)
    if value is None:
        raise ProviderError(
            f"ken_french sentinel cell period_end={period_end.isoformat()} column={column!r} raw={raw!r}"
        )
    return value


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


def _csv_parts(line: str) -> list[str]:
    return [part.strip() for part in line.split(",")]


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
