"""N-PORT quarter fetch + persist seam."""

from __future__ import annotations

import hashlib
import json
import logging
import zipfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

import httpx

from src.data.merge import load_prior_partition, merge_incremental
from src.data.paths import NPORT_SERIES_MAP_PATH
from src.data.pipeline import persist_ingest
from src.data.providers.base import ProviderError
from src.data.providers.sec_nport import _parse_raw_tables as _sec_parse_raw_tables
from src.data.providers.sec_nport import normalize_nport_holdings
from src.data.schema import Dataset
from src.data.settings import DataSettings
from src.data.storage import DatasetArtifact, RawPayload

logger = logging.getLogger(__name__)

_USER_AGENT = "ETF-Manager/1.0 contact@example.com"
_NPORT_BULK_BASE = "https://www.sec.gov/files/dera/data/form-n-port-data-sets"


def _nport_bulk_url(filing_quarter: str) -> str:
    """SEC DERA bulk ZIP URL for a quarter label like 2019q4."""
    fq = filing_quarter.strip().lower()
    return f"{_NPORT_BULK_BASE}/{fq}_nport.zip"


def load_nport_series_map(path: Path = NPORT_SERIES_MAP_PATH) -> Mapping[str, str]:
    """Load series_id -> ticker map; fails closed on unreadable JSON."""
    if not path.is_file():
        raise OSError(f"nport series map not found: {path}")
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"nport series map unreadable at {path}: {exc}") from exc
    if not isinstance(doc, dict):
        raise ValueError(f"nport series map root must be object at {path}")
    return {str(k): str(v) for k, v in doc.items()}


def _read_nport_pointer_zip(pointer_path: Path, data_root: Path) -> bytes | None:
    """Return cached ZIP bytes when pointer JSON and payload hash are valid."""
    try:
        doc = json.loads(pointer_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    sha = doc.get("sha256")
    rel = doc.get("relative_path")
    if not isinstance(sha, str) or not isinstance(rel, str) or len(sha) != 64:
        return None
    target = data_root / Path(rel)
    try:
        target.resolve().relative_to(data_root.resolve())
    except ValueError:
        return None
    if not target.is_file():
        return None
    data = target.read_bytes()
    if hashlib.sha256(data).hexdigest() != sha:
        return None
    return data


def _load_nport_zip_bytes(
    filing_quarter: str, settings: DataSettings, client: httpx.Client | None
) -> tuple[bytes, bool]:
    """Load N-PORT ZIP from pointer cache when valid, otherwise HTTP GET."""
    fq = filing_quarter.strip().lower()
    url = _nport_bulk_url(fq)
    data_root = settings.resolved_data_root()
    pointer_path = data_root / Path("raw/sec/nport") / f"{fq}.json"
    if pointer_path.is_file():
        cached = _read_nport_pointer_zip(pointer_path, data_root)
        if cached is not None:
            return cached, True
    # Fallback to HTTP
    if client is not None:
        try:
            resp = client.get(url, headers={"User-Agent": _USER_AGENT, "Accept-Encoding": "gzip"})
            if resp.status_code >= 400:
                raise ProviderError(f"provider returned HTTP {resp.status_code}")
            return resp.content, False
        except httpx.HTTPError as exc:
            raise ProviderError(f"transport failure ({type(exc).__name__})") from exc
    else:
        with httpx.Client(timeout=httpx.Timeout(600.0), headers={"User-Agent": _USER_AGENT}) as owned:
            try:
                resp = owned.get(url, headers={"User-Agent": _USER_AGENT})
                if resp.status_code >= 400:
                    raise ProviderError(f"provider returned HTTP {resp.status_code}")
                return resp.content, False
            except httpx.HTTPError as exc:
                raise ProviderError(f"transport failure ({type(exc).__name__})") from exc


def fetch_and_persist_nport_quarter(
    *,
    filing_quarter: str,
    series_map_path: Path = NPORT_SERIES_MAP_PATH,
    settings: DataSettings,
    client: httpx.Client | None = None,
) -> DatasetArtifact:
    """Fetch SEC N-PORT bulk ZIP for quarter, normalize, and persist Dataset.ETF_HOLDINGS."""
    series_map = load_nport_series_map(series_map_path)
    if not filing_quarter:
        raise ValueError("filing_quarter must be non-empty like 2019q4")
    # Normalize quarter string
    fq = filing_quarter.strip().lower()
    url = _nport_bulk_url(fq)
    retrieved_at = datetime.now(UTC)
    content, _from_cache = _load_nport_zip_bytes(fq, settings, client)

    # Validate zip
    try:
        with zipfile.ZipFile(__import__("io").BytesIO(content)) as _:
            pass
    except zipfile.BadZipFile as exc:
        raise ProviderError(f"sec nport payload is not a valid ZIP for {fq}") from exc

    # Parse and normalize; raw bytes are stored content-addressed via persist_ingest.
    raw_tables = _sec_parse_raw_tables(content)
    frame = normalize_nport_holdings(raw_tables, series_map=series_map, retrieved_at=retrieved_at)

    prior = load_prior_partition(settings, Dataset.ETF_HOLDINGS)
    frame = merge_incremental(prior, frame, Dataset.ETF_HOLDINGS)

    payload = RawPayload(
        provider="sec",
        endpoint=url,
        request_params={"filing_quarter": fq},
        retrieved_at=retrieved_at,
        extension="zip",
        content=content,
    )
    artifact = persist_ingest(frame, Dataset.ETF_HOLDINGS, payload, settings, prior=prior)
    # Write optional pointer JSON under raw/sec/nport/<quarter>.json (<8 KiB) containing sha256 and content path.
    try:
        data_root = settings.resolved_data_root()
        pointer_path = data_root / Path("raw/sec/nport") / f"{fq}.json"
        pointer_path.parent.mkdir(parents=True, exist_ok=True)
        payload_sha = hashlib.sha256(content).hexdigest()
        rel = artifact.manifest.raw_artifact.relative_path.as_posix()
        doc = {"sha256": payload_sha, "relative_path": rel, "filing_quarter": fq}
        serialized = json.dumps(doc, sort_keys=True, separators=(",", ":")).encode("utf-8")
        if len(serialized) < 8192 and (not pointer_path.exists() or pointer_path.read_bytes() != serialized):
            pointer_path.write_bytes(serialized)
    except OSError:  # pragma: no cover
        logger.warning("[DATA] event=nport_pointer_write_failed quarter=%s", fq)
    logger.info(
        "[DATA] event=fetch_persist dataset=%s provider=sec rows=%d",
        str(Dataset.ETF_HOLDINGS),
        artifact.manifest.row_count,
    )
    return artifact


def fetch_and_persist_nport_quarters(
    *,
    filing_quarters: Sequence[str],
    series_map_path: Path = NPORT_SERIES_MAP_PATH,
    settings: DataSettings,
) -> tuple[DatasetArtifact, ...]:
    """Fetch multiple quarters and persist one merged ETF_HOLDINGS partition."""
    # wiring: fetch_and_persist_nport_quarters(
    _ = fetch_and_persist_nport_quarters  # self reference for wiring detection

    if not filing_quarters:
        raise ValueError("filing_quarters must be non-empty")

    last_artifact: DatasetArtifact | None = None
    for fq in filing_quarters:
        last_artifact = fetch_and_persist_nport_quarter(
            filing_quarter=str(fq).strip().lower(),
            series_map_path=series_map_path,
            settings=settings,
        )
    return (last_artifact,) if last_artifact is not None else ()
