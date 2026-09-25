"""Backfill non-revised MACRO series before their first ALFRED vintage.

ALFRED vintage history for market-observed daily series such as VIXCLS and BAA10Y
starts near 2012, which blocks pre-2012 walk-forward and crisis validation. Those
series are not revised, so their latest observations, released at the close of a
conservative session lag, are point-in-time safe. The allowlist keeps that judgement
explicit and versioned instead of inferred from the data.
"""

from __future__ import annotations

import json
import logging
from bisect import bisect_left
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, cast

import httpx
import polars as pl

from src.data.calendar import DEFAULT_CALENDAR_NAME, TradingCalendar, load_calendar
from src.data.merge import load_prior_partition, merge_incremental
from src.data.paths import MACRO_BACKFILL_PATH, resolve_input_path
from src.data.pipeline import persist_ingest
from src.data.providers.base import DEFAULT_TIMEOUT_S
from src.data.providers.fred import FredClient
from src.data.schema import TS_DTYPE, Dataset
from src.data.storage import DatasetArtifact, RawPayload

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from src.data.secrets import ProviderSecrets
    from src.data.settings import DataSettings
    from src.data.storage import JSONValue

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class UnrevisedMacroSeries:
    """A MACRO series eligible for pre-vintage backfill.

    Attributes:
        series_id: FRED identifier already present in MACRO with vintage history.
        release_lag_sessions: XNYS sessions after the observation session before the
            value is treated as released (>= 1).
    """

    series_id: str
    release_lag_sessions: int


def load_unrevised_macro_series(path: str | Path = MACRO_BACKFILL_PATH) -> tuple[UnrevisedMacroSeries, ...]:
    """Load the versioned allowlist of non-revised MACRO series.

    A relative path is anchored to the repository root, so every caller reads the one
    git-tracked allowlist regardless of the process working directory.

    Args:
        path: JSON document with a `series` list.

    Returns:
        Allowlisted series ordered by series id.

    Raises:
        ValueError: On duplicate ids, lags below one, or malformed JSON.
    """
    source = resolve_input_path(path)
    raw = source.read_bytes()
    try:
        document = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"macro backfill file {source.as_posix()!r} is not valid JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise ValueError(f"macro backfill file {source.as_posix()!r} is not a JSON object")
    entries = document.get("series")
    if not isinstance(entries, list):
        raise ValueError(f"macro backfill file {source.as_posix()!r} declares no 'series' list")
    allowlist: list[UnrevisedMacroSeries] = []
    seen: set[str] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"series {index} of {source.as_posix()!r} is not a JSON object")
        series_id = str(entry.get("series_id", "")).strip()
        if not series_id:
            raise ValueError(f"series {index} of {source.as_posix()!r} declares no 'series_id'")
        lag = entry.get("release_lag_sessions")
        if isinstance(lag, bool) or not isinstance(lag, int) or lag < 1:
            raise ValueError(
                f"series {series_id!r} needs an integer 'release_lag_sessions' of at least 1, got {lag!r}"
            )
        if series_id in seen:
            raise ValueError(f"duplicate macro backfill series {series_id!r}")
        seen.add(series_id)
        allowlist.append(UnrevisedMacroSeries(series_id=series_id, release_lag_sessions=lag))
    allowlist.sort(key=lambda item: item.series_id)
    return tuple(allowlist)


@contextmanager
def _session(injected: httpx.Client | None) -> Iterator[httpx.Client]:
    """Pass an injected client through unchanged or open a short-lived default."""
    if injected is not None:
        yield injected
        return
    with httpx.Client(timeout=httpx.Timeout(DEFAULT_TIMEOUT_S)) as owned:
        yield owned


def _cutoff(prior_frame: pl.DataFrame, series_id: str) -> date:
    """Earliest observation_date already stored for one series; the backfill boundary.

    Raises:
        ValueError: If the series has no stored MACRO row to extend.
    """
    stored = prior_frame.filter(pl.col("series_id") == series_id)
    if stored.is_empty():
        raise ValueError(f"series {series_id!r} has no MACRO rows to extend")
    return cast(date, stored.get_column("observation_date").min())


def _sessions_through_lag(calendar: TradingCalendar, start: date, end: date, max_lag: int) -> list[date]:
    """Ascending sessions in [start, end] plus `max_lag` trailing sessions for releases."""
    sessions = list(calendar.sessions(start, end))
    if not sessions:
        raise ValueError(f"no exchange session in [{start.isoformat()}, {end.isoformat()}]")
    last = sessions[-1]
    sessions.extend(calendar.next_session(last, offset) for offset in range(1, max_lag + 1))
    return sessions


def _release_close(day: date, lag: int, sessions: list[date], calendar: TradingCalendar) -> datetime:
    """Close of the lag-th session after the observation's own or next session.

    A vendor print dated on an exchange holiday rolls forward to the next session
    before the lag applies, so a backfill row is never released before the value
    could have been published.
    """
    return calendar.close_ts(sessions[bisect_left(sessions, day) + lag])


def _backfill_rows(
    latest: pl.DataFrame, cutoff: date, lag: int, sessions: list[date], calendar: TradingCalendar
) -> pl.DataFrame:
    """Keep observations strictly before the cutoff and stamp each release close."""
    rows = latest.filter(pl.col("observation_date") < cutoff)
    observed = cast(list[date], rows.get_column("observation_date").to_list())
    return rows.with_columns(
        pl.Series(
            "release_date",
            [_release_close(day, lag, sessions, calendar) for day in observed],
            dtype=TS_DTYPE,
        )
    )


def _raw_observations(content: bytes) -> list[JSONValue]:
    """Re-read the validated provider payload so the archive keeps the vendor rows verbatim."""
    document = cast("dict[str, JSONValue]", json.loads(content.decode("utf-8")))
    return cast("list[JSONValue]", document["observations"])


def fetch_and_persist_macro_backfill(
    series_ids: Sequence[str],
    start: date,
    *,
    secrets: ProviderSecrets,
    settings: DataSettings,
    client: httpx.Client | None = None,
) -> DatasetArtifact:
    """Extend allowlisted MACRO series backwards before their first ALFRED vintage.

    The cutoff for each series is the earliest observation_date already present in
    the latest trusted MACRO partition, so backfill rows never overlap vintage rows.
    Each backfilled value is released at the close of the observation session
    shifted forward by the series' lag, which is never earlier than a real
    publication of an unrevised market series.

    Args:
        series_ids: Allowlisted series to extend.
        start: First observation date to request.
        secrets: Provider credentials.
        settings: Catalog root.
        client: Optional injected HTTP client.

    Returns:
        The new MACRO partition containing prior rows plus backfill rows.

    Raises:
        ValueError: If a series is not allowlisted, absent from MACRO, or start is not
            before its cutoff.
        PriorPartitionUntrustedError: If the prior MACRO partition cannot be verified.
    """
    requested = tuple(series_ids)
    if not requested:
        raise ValueError("macro backfill requires at least one series id")
    if len(set(requested)) != len(requested):
        raise ValueError(f"macro backfill received duplicate series ids: {requested!r}")
    allowlist = {item.series_id: item for item in load_unrevised_macro_series()}
    unlisted = sorted(set(requested) - set(allowlist))
    if unlisted:
        raise ValueError(f"series {unlisted!r} are not allowlisted for unrevised macro backfill")
    prior = load_prior_partition(settings, Dataset.MACRO)
    if prior is None:
        raise ValueError("macro backfill requires an existing MACRO partition to extend")
    cutoffs = {series_id: _cutoff(prior.frame, series_id) for series_id in requested}
    for series_id in requested:
        if start >= cutoffs[series_id]:
            raise ValueError(
                f"macro backfill start {start.isoformat()} is not before the {series_id} "
                f"first vintage {cutoffs[series_id].isoformat()}"
            )
    lags = {series_id: allowlist[series_id].release_lag_sessions for series_id in requested}
    calendar = load_calendar(DEFAULT_CALENDAR_NAME)
    sessions = _sessions_through_lag(calendar, start, max(cutoffs.values()), max(lags.values()))
    retrieved_at = datetime.now(UTC)
    frames: list[pl.DataFrame] = []
    observations: list[JSONValue] = []
    with _session(client) as http:
        fred = FredClient(secrets.fred_api, http)
        for series_id in requested:
            payload, latest = fred.fetch_latest_observations(series_id, start, cutoffs[series_id])
            frames.append(_backfill_rows(latest, cutoffs[series_id], lags[series_id], sessions, calendar))
            observations.extend(_raw_observations(payload.content))
    lineage = RawPayload(
        provider="fred",
        endpoint=f"series/observations/{'+'.join(requested)}/latest",
        request_params={
            "series_ids": list(requested),
            "observation_start": start.isoformat(),
            "first_vintage_cutoff": {sid: cutoffs[sid].isoformat() for sid in requested},
            "release_lag_sessions": {sid: str(lags[sid]) for sid in requested},
            "file_type": "json",
        },
        retrieved_at=retrieved_at,
        extension="json",
        content=json.dumps({"observations": observations}).encode("utf-8"),
    )
    merged = merge_incremental(prior, pl.concat(frames, how="vertical"), Dataset.MACRO)
    artifact = persist_ingest(merged, Dataset.MACRO, lineage, settings, prior=prior)
    logger.info(
        "[DATA] event=macro_backfill dataset=macro series_ids=%s start=%s rows=%d",
        ",".join(requested),
        start.isoformat(),
        artifact.manifest.row_count,
    )
    return artifact
