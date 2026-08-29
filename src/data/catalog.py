"""Trusted-partition catalog: list manifests and load PIT-visible frames."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path, PurePosixPath

import polars as pl

from src.data.quality import FindingSeverity, QualityFinding
from src.data.query import load_as_of
from src.data.schema import Dataset, spec_for
from src.data.settings import DataSettings
from src.data.storage import (
    DatasetArtifact,
    DatasetManifest,
    DataStore,
    JSONValue,
    RawArtifact,
    UntrustedDatasetError,
)

logger = logging.getLogger(__name__)

_CATALOG_FRAME_CACHE: dict[tuple[Dataset, str], pl.DataFrame] = {}


def clear_catalog_frame_cache() -> None:
    _CATALOG_FRAME_CACHE.clear()


def latest_artifact(settings: DataSettings, dataset: Dataset) -> DatasetArtifact:
    """Return the newest verified partition for ``dataset`` under the data root.

    Every manifest under ``manifests/<dataset>/`` is reconstructed into a
    :class:`DatasetManifest`; the winner by ``(retrieved_at, normalized_sha256)``
    is re-verified through ``DataStore.read_normalized`` before returning.

    Raises:
        UntrustedDatasetError: If no readable manifest exists or files fail lineage checks.
    """
    root = settings.resolved_data_root()
    manifests_dir = root / "manifests" / str(dataset)
    candidates = sorted(manifests_dir.glob("*.json")) if manifests_dir.is_dir() else []
    if not candidates:
        raise UntrustedDatasetError(f"no trusted manifest partitions under {manifests_dir.as_posix()}")

    best_key: tuple[datetime, str] | None = None
    best_artifact: DatasetArtifact | None = None
    for path in candidates:
        document = _load_manifest_document(path)
        manifest = _reconstruct_manifest(document, path)
        artifact = DatasetArtifact(
            normalized_path=root.joinpath(*manifest.normalized_relative_path.parts),
            manifest_path=path,
            manifest=manifest,
        )
        key = (manifest.retrieved_at, manifest.normalized_sha256)
        if best_key is None or key > best_key:
            best_key, best_artifact = key, artifact

    assert best_artifact is not None
    cache_key = (dataset, best_artifact.manifest.normalized_sha256)
    cached = _CATALOG_FRAME_CACHE.get(cache_key)
    if cached is None:
        frame = DataStore(settings).read_normalized(best_artifact, spec_for(dataset))
        _CATALOG_FRAME_CACHE[cache_key] = frame
    logger.info(
        "[DATA] event=catalog_latest dataset=%s sha=%s retrieved_at=%s",
        str(dataset),
        best_artifact.manifest.normalized_sha256,
        best_artifact.manifest.retrieved_at.isoformat(),
    )
    return best_artifact


def load_visible(settings: DataSettings, dataset: Dataset, decision_ts: datetime) -> pl.DataFrame:
    """Read the latest partition and apply ``load_as_of`` at ``decision_ts``.

    Raises:
        UntrustedDatasetError: When the latest partition fails any lineage check.
        ValueError: On a naive ``decision_ts``.
    """
    artifact = latest_artifact(settings, dataset)
    cache_key = (dataset, artifact.manifest.normalized_sha256)
    frame = _CATALOG_FRAME_CACHE.get(cache_key)
    if frame is None:
        frame = DataStore(settings).read_normalized(artifact, spec_for(dataset))
        _CATALOG_FRAME_CACHE[cache_key] = frame
    visible = load_as_of(frame, dataset, decision_ts)
    logger.info("[DATA] event=catalog_visible dataset=%s rows=%d", str(dataset), visible.height)
    return visible


def _load_manifest_document(path: Path) -> dict[str, JSONValue]:
    try:
        document: JSONValue = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise UntrustedDatasetError(f"manifest unreadable at {path.as_posix()}: {exc}") from exc
    if not isinstance(document, dict):
        raise UntrustedDatasetError(f"manifest root must be an object: {path.as_posix()}")
    return document


def _reconstruct_manifest(document: dict[str, JSONValue], path: Path) -> DatasetManifest:
    """Rebuild the in-memory lineage record from its credential-free projection."""
    try:
        raw_section = document["raw_artifact"]
        request_params = document["request_params"]
        quality_items = document["quality_findings"]
        if not isinstance(raw_section, dict) or not isinstance(request_params, dict):
            raise TypeError("raw_artifact and request_params must be objects")
        if not isinstance(quality_items, list):
            raise TypeError("quality_findings must be an array")
        retrieved_at = datetime.fromisoformat(str(document["retrieved_at"]))
        raw_retrieved_at = datetime.fromisoformat(str(raw_section["retrieved_at"]))
        if retrieved_at.tzinfo is None or raw_retrieved_at.tzinfo is None:
            raise ValueError("retrieved_at timestamps must be timezone-aware")
        return DatasetManifest(
            dataset=Dataset(str(document["dataset"])),
            provider=str(document["provider"]),
            endpoint=str(document["endpoint"]),
            request_params=request_params,
            retrieved_at=retrieved_at,
            raw_artifact=RawArtifact(
                relative_path=PurePosixPath(str(raw_section["relative_path"])),
                sha256=str(raw_section["sha256"]),
                retrieved_at=raw_retrieved_at,
            ),
            normalized_relative_path=PurePosixPath(str(document["normalized_relative_path"])),
            normalized_sha256=str(document["normalized_sha256"]),
            row_count=_required_int(document, "row_count"),
            schema_version=str(document["schema_version"]),
            normalization_version=str(document["normalization_version"]),
            quality_findings=tuple(_finding_from_item(item) for item in quality_items),
        )
    except UntrustedDatasetError:
        raise
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise UntrustedDatasetError(f"manifest malformed at {path.as_posix()}: {exc}") from exc


def _finding_from_item(item: JSONValue) -> QualityFinding:
    if not isinstance(item, dict):
        raise TypeError("quality finding must be an object")
    return QualityFinding(
        code=str(item["code"]),
        severity=FindingSeverity(str(item["severity"])),
        message=str(item["message"]),
        row_count=_required_int(item, "row_count"),
    )


def _required_int(source: dict[str, JSONValue], key: str) -> int:
    value = source[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{key} must be a number")
    return int(value)
