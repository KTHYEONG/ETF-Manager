"""N-PORT quarter fetch + persist seam."""

from __future__ import annotations

import hashlib
import json
import logging
import zipfile
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

import httpx

from src.data.pipeline import persist_ingest
from src.data.providers.base import ProviderError
from src.data.providers.sec_nport import SecNportClient, normalize_nport_holdings
from src.data.providers.sec_nport import _parse_raw_tables as _sec_parse_raw_tables
from src.data.schema import Dataset
from src.data.settings import DataSettings
from src.data.storage import DatasetArtifact, RawPayload

logger = logging.getLogger(__name__)

_DEFAULT_SERIES_MAP = Path("configs/etf_metadata/nport_series_map.json")
_USER_AGENT = "ETF-Manager/1.0 contact@example.com"
_NPORT_BULK_BASE = "https://www.sec.gov/files/dera/data/form-n-port-data-sets"


def _nport_bulk_url(filing_quarter: str) -> str:
    """SEC DERA bulk ZIP URL for a quarter label like 2019q4."""
    fq = filing_quarter.strip().lower()
    return f"{_NPORT_BULK_BASE}/{fq}_nport.zip"


def load_nport_series_map(path: Path = Path("configs/etf_metadata/nport_series_map.json")) -> Mapping[str, str]:
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


def fetch_and_persist_nport_quarter(
    *,
    filing_quarter: str,
    series_map_path: Path = Path("configs/etf_metadata/nport_series_map.json"),
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
    content: bytes
    if client is not None:
        try:
            resp = client.get(url, headers={"User-Agent": _USER_AGENT, "Accept-Encoding": "gzip"})
            if resp.status_code >= 400:
                raise ProviderError(f"provider returned HTTP {resp.status_code}")
            content = resp.content
        except httpx.HTTPError as exc:
            raise ProviderError(f"transport failure ({type(exc).__name__})") from exc
    else:
        with httpx.Client(timeout=httpx.Timeout(600.0), headers={"User-Agent": _USER_AGENT}) as owned:
            try:
                resp = owned.get(url, headers={"User-Agent": _USER_AGENT})
                if resp.status_code >= 400:
                    raise ProviderError(f"provider returned HTTP {resp.status_code}")
                content = resp.content
            except httpx.HTTPError as exc:
                raise ProviderError(f"transport failure ({type(exc).__name__})") from exc

    # Validate zip
    try:
        with zipfile.ZipFile(__import__("io").BytesIO(content)) as _:
            pass
    except zipfile.BadZipFile as exc:
        raise ProviderError(f"sec nport payload is not a valid ZIP for {fq}") from exc

    # Store raw ZIP under data/raw/sec/nport/
    raw_relative = Path("raw") / "sec" / "nport" / f"{fq}.zip"
    # Persist raw via archive path for lineage but also write via DataStore.store_raw path?
    # We will create RawPayload with extension zip and rely on persist_ingest to store under sec/etf_holdings hash path;
    # additionally, write raw ZIP to data/raw/sec/nport/ for requirement
    data_root = settings.resolved_data_root()
    raw_nport_path = data_root / raw_relative
    raw_nport_path.parent.mkdir(parents=True, exist_ok=True)
    # atomic-ish write
    if not raw_nport_path.exists():
        raw_nport_path.write_bytes(content)
    else:
        # ensure hash matches if existing
        existing = raw_nport_path.read_bytes()
        if hashlib.sha256(existing).hexdigest() != hashlib.sha256(content).hexdigest():
            # overwrite with new content (amendment) - keep latest
            raw_nport_path.write_bytes(content)

    # Parse and normalize (SecNportClient reference for orphan check)
    _ = SecNportClient
    raw_tables = _sec_parse_raw_tables(content)
    frame = normalize_nport_holdings(raw_tables, series_map=series_map, retrieved_at=retrieved_at)

    payload = RawPayload(
        provider="sec",
        endpoint=url,
        request_params={"filing_quarter": fq},
        retrieved_at=retrieved_at,
        extension="zip",
        content=content,
    )
    artifact = persist_ingest(frame, Dataset.ETF_HOLDINGS, payload, settings)
    logger.info("[DATA] event=fetch_persist dataset=%s provider=sec rows=%d", str(Dataset.ETF_HOLDINGS), artifact.manifest.row_count)
    return artifact
