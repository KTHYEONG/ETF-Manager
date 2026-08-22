"""FRED/ALFRED observation and vintage client."""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING, Final

import polars as pl

from src.etf_manager.data.providers.base import ProviderError, ProviderResponse, get_json
from src.etf_manager.data.schema import TS_DTYPE, Dataset, spec_for
from src.etf_manager.data.storage import RawPayload

if TYPE_CHECKING:
    import httpx

    from src.etf_manager.data.storage import JSONValue

logger = logging.getLogger(__name__)

_OBSERVATIONS_URL: Final[str] = "https://api.stlouisfed.org/fred/series/observations"
_FX_SERIES: Final[str] = "DEXKOUS"


class FredClient:
    """Maps FRED DEXKOUS to FX and ALFRED vintage observations to MACRO."""

    def __init__(self, api_key: str, client: httpx.Client) -> None:
        self._api_key = api_key
        self._client = client

    def fetch_fx(self, start: date, end: date) -> tuple[RawPayload, pl.DataFrame]:
        """Fetch DEXKOUS as Dataset.FX (source=fred); '.' values persist as null gap rows."""
        spec = spec_for(Dataset.FX)
        retrieved_at = datetime.now(UTC)
        lineage = {
            "series_id": _FX_SERIES,
            "file_type": "json",
            "observation_start": start.isoformat(),
            "observation_end": end.isoformat(),
        }
        response, observations = self._get(lineage)
        records: list[dict[str, object]] = []
        for row in observations:
            day, value = _observed(row)
            records.append({"date": day, "usdkrw": value})
        frame = (
            pl.DataFrame(records)
            .with_columns(
                pl.lit("fred", dtype=pl.String()).alias("source"),
                pl.lit(retrieved_at, dtype=TS_DTYPE).alias("retrieved_at"),
            )
            .select(*spec.columns)
            .cast(pl.Schema(dict(spec.columns)))
        )
        logger.info("[DATA] event=fetch dataset=fx provider=fred series=%s rows=%d", _FX_SERIES, frame.height)
        return _payload(_FX_SERIES, response.content, retrieved_at, lineage), frame

    def fetch_macro_vintages(self, series_id: str, start: date, end: date) -> tuple[RawPayload, pl.DataFrame]:
        """Fetch ALFRED vintage history as Dataset.MACRO; release_date comes from realtime_end."""
        spec = spec_for(Dataset.MACRO)
        retrieved_at = datetime.now(UTC)
        lineage = {
            "series_id": series_id,
            "file_type": "json",
            "realtime_start": start.isoformat(),
            "realtime_end": end.isoformat(),
        }
        response, observations = self._get(lineage)
        records: list[dict[str, object]] = []
        for row in observations:
            day, value = _observed(row)
            records.append(
                {
                    "series_id": series_id,
                    "observation_date": day,
                    "release_date": _vintage(row, series_id),
                    "value": value,
                }
            )
        frame = pl.DataFrame(records).select(*spec.columns).cast(pl.Schema(dict(spec.columns)))
        logger.info("[DATA] event=fetch dataset=macro provider=fred series=%s rows=%d", series_id, frame.height)
        return _payload(series_id, response.content, retrieved_at, lineage), frame

    def _get(self, lineage: dict[str, str]) -> tuple[ProviderResponse, list[JSONValue]]:
        """GET one observations document; api_key rides the wire only."""
        response = get_json(
            self._client,
            _OBSERVATIONS_URL,
            params={"api_key": self._api_key, **lineage},
        )
        body = response.body
        if not isinstance(body, dict):
            raise ProviderError(f"fred payload for {lineage['series_id']} is not an object")
        observations = body.get("observations")
        if not isinstance(observations, list) or not observations:
            raise ProviderError(f"fred returned no observations for {lineage['series_id']}")
        return response, observations


def _observed(row: JSONValue) -> tuple[date, float | None]:
    """Parse one observation into (date, value); '.' marks a published gap."""
    if not isinstance(row, dict) or "date" not in row or "value" not in row:
        raise ProviderError("fred observation misses date or value")
    try:
        day = date.fromisoformat(str(row["date"])[:10])
    except ValueError as exc:
        raise ProviderError(f"fred observation date {str(row['date'])[:10]!r} is malformed") from exc
    raw = row["value"]
    if raw is None or raw in (".", ""):
        return day, None
    if isinstance(raw, bool) or not isinstance(raw, int | float | str):
        raise ProviderError("fred observation value is not numeric")
    try:
        return day, float(raw)
    except ValueError as exc:
        raise ProviderError("fred observation value is not numeric") from exc


def _vintage(row: JSONValue, series_id: str) -> datetime:
    """Map realtime_end onto a timezone-aware UTC midnight release instant."""
    raw = row.get("realtime_end") if isinstance(row, dict) else None
    try:
        day = date.fromisoformat(str(raw)[:10])
    except ValueError as exc:
        raise ProviderError(f"fred vintage for {series_id} misses realtime_end") from exc
    return datetime(day.year, day.month, day.day, tzinfo=UTC)


def _payload(series_id: str, content: bytes, retrieved_at: datetime, lineage: dict[str, str]) -> RawPayload:
    """Lineage excludes api_key; the wire query carried it, storage never does."""
    return RawPayload(
        provider="fred",
        endpoint=f"series/observations/{series_id}",
        request_params=dict(lineage),
        retrieved_at=retrieved_at,
        extension="json",
        content=content,
    )
