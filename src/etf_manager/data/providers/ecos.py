"""Bank of Korea ECOS OpenAPI client."""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING, Final

import polars as pl

from src.etf_manager.data.providers.base import ProviderError, ProviderResponse, get_json
from src.etf_manager.data.schema import TS_DTYPE, Dataset, spec_for
from src.etf_manager.data.storage import RawPayload

if TYPE_CHECKING:
    from collections.abc import Callable

    import httpx

    from src.etf_manager.data.storage import JSONValue

logger = logging.getLogger(__name__)

_API_URL: Final[str] = "https://ecos.bok.or.kr/api/StatisticSearch"
_REQUEST_BEGIN: Final[str] = "1"
_REQUEST_COUNT: Final[str] = "100000"
_FX_STAT: Final[tuple[str, str, str]] = ("731Y001", "D", "0000001")
_CPI_STAT: Final[tuple[str, str, str]] = ("901Y009", "M", "0")


class EcosClient:
    """Maps ECOS USD/KRW to FX and CPI to Dataset.CPI."""

    def __init__(self, api_key: str, client: httpx.Client) -> None:
        self._api_key = api_key
        self._client = client

    def fetch_fx(self, start: date, end: date) -> tuple[RawPayload, pl.DataFrame]:
        """StatisticSearch 731Y001 / 0000001 daily as Dataset.FX (source=ecos)."""
        stat_code, cycle, item = _FX_STAT
        return self._fetch(_fx_record, Dataset.FX, stat_code, cycle, item, start, end)

    def fetch_cpi(self, start: date, end: date) -> tuple[RawPayload, pl.DataFrame]:
        """StatisticSearch 901Y009 monthly as Dataset.CPI; period_end is month-end."""
        stat_code, cycle, item = _CPI_STAT
        return self._fetch(_cpi_record, Dataset.CPI, stat_code, cycle, item, start, end)

    def _fetch(
        self,
        record: Callable[[JSONValue], dict[str, object]],
        dataset: Dataset,
        stat_code: str,
        cycle: str,
        item: str,
        start: date,
        end: date,
    ) -> tuple[RawPayload, pl.DataFrame]:
        spec = spec_for(dataset)
        retrieved_at = datetime.now(UTC)
        response, rows = self._search(stat_code, cycle, item, start, end)
        records = [record(row) for row in rows]
        frame = (
            pl.DataFrame(records)
            .with_columns(
                pl.lit("ecos", dtype=pl.String()).alias("source"),
                pl.lit(retrieved_at, dtype=TS_DTYPE).alias("retrieved_at"),
            )
            .select(*spec.columns)
            .cast(pl.Schema(dict(spec.columns)))
        )
        logger.info("[DATA] event=fetch dataset=%s provider=ecos stat=%s rows=%d", str(dataset), stat_code, frame.height)
        return _payload(dataset, stat_code, cycle, item, start, end, retrieved_at, response.content), frame

    def _search(
        self, stat_code: str, cycle: str, item: str, start: date, end: date
    ) -> tuple[ProviderResponse, list[JSONValue]]:
        url = (
            f"{_API_URL}/{self._api_key}/json/kr/{_REQUEST_BEGIN}/{_REQUEST_COUNT}"
            f"/{stat_code}/{cycle}/{_format_date(start, cycle)}/{_format_date(end, cycle)}/{item}"
        )
        response = get_json(self._client, url)
        body = response.body
        if not isinstance(body, dict):
            raise ProviderError(f"ecos payload for {stat_code} is not an object")
        result = body.get("RESULT")
        if isinstance(result, dict) and result.get("CODE") != "SUCCESS":
            raise ProviderError(f"ecos request failed with code {result.get('CODE')!r}")
        section = body.get("StatisticSearch")
        if not isinstance(section, dict):
            raise ProviderError(f"ecos response misses StatisticSearch data for {stat_code}")
        rows = section.get("row")
        if isinstance(rows, dict):
            # Single-row responses arrive unwrapped.
            rows = [rows]
        if not isinstance(rows, list) or not rows:
            raise ProviderError(f"ecos returned no rows for {stat_code}")
        return response, rows


def _fx_record(row: JSONValue) -> dict[str, object]:
    return {"date": _period(row), "usdkrw": _value(row)}


def _cpi_record(row: JSONValue) -> dict[str, object]:
    return {"period_end": _period(row), "value": _value(row)}


def _payload(
    dataset: Dataset,
    stat_code: str,
    cycle: str,
    item: str,
    start: date,
    end: date,
    retrieved_at: datetime,
    content: bytes,
) -> RawPayload:
    """Lineage keeps only keyless route facts; the API key never enters storage."""
    return RawPayload(
        provider="ecos",
        endpoint=f"StatisticSearch/{stat_code}/{cycle}",
        request_params={
            "stat_code": stat_code,
            "cycle": cycle,
            "begin_date": start.isoformat(),
            "end_date": end.isoformat(),
            "item": item,
        },
        retrieved_at=retrieved_at,
        extension="json",
        content=content,
    )


def _format_date(day: date, cycle: str) -> str:
    """Format date matching ECOS cycle requirement ('YYYYMM' for 'M', 'YYYYMMDD' for 'D')."""
    if cycle == "M":
        return f"{day.year:04d}{day.month:02d}"
    if cycle == "D":
        return f"{day.year:04d}{day.month:02d}{day.day:02d}"
    raise ProviderError(f"unsupported ecos cycle {cycle!r}")


def _period(row: JSONValue) -> date:
    """TIME 'YYYYMMDD' maps to that day; 'YYYYMM' maps to the month-end day."""
    text = str(row.get("TIME", "")) if isinstance(row, dict) else ""
    if len(text) == 8 and text.isdigit():
        try:
            return date(int(text[:4]), int(text[4:6]), int(text[6:8]))
        except ValueError as exc:
            raise ProviderError(f"ecos TIME {text!r} is malformed") from exc
    if len(text) == 6 and text.isdigit():
        year, month = int(text[:4]), int(text[4:])
        first_next = date(year + month // 12, month % 12 + 1, 1)
        return first_next - timedelta(days=1)
    raise ProviderError(f"ecos TIME {text!r} is malformed")


def _value(row: JSONValue) -> float | None:
    raw = row.get("DATA_VALUE") if isinstance(row, dict) else None
    if raw is None or raw in (".", ""):
        return None
    if isinstance(raw, bool) or not isinstance(raw, int | float | str):
        raise ProviderError("ecos DATA_VALUE is not numeric")
    try:
        return float(raw)
    except ValueError as exc:
        raise ProviderError("ecos DATA_VALUE is not numeric") from exc
