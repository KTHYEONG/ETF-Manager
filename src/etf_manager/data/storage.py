"""Immutable raw archive, canonical frame hashing, and manifest-bound reads."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import logging
import os
import re
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Final, TypeAlias

import polars as pl

from src.etf_manager.data.pit import AVAILABLE_AT, assert_no_lookahead
from src.etf_manager.data.quality import DataQualityError, QualityFinding, QualityReport
from src.etf_manager.data.schema import Dataset, DatasetSpec
from src.etf_manager.data.settings import DataSettings

logger = logging.getLogger(__name__)

JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]

_PROVIDER_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_EXTENSION_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[a-z0-9]+$")
_HEX64_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class RawPayload:
    """Provider response body plus exact lineage of how it was obtained."""

    provider: str
    endpoint: str
    request_params: Mapping[str, JSONValue]
    retrieved_at: datetime
    extension: str
    content: bytes

    def __post_init__(self) -> None:
        if self.retrieved_at.tzinfo is None or self.retrieved_at.utcoffset() is None:
            raise ValueError("retrieved_at must be timezone-aware")
        if _EXTENSION_PATTERN.fullmatch(self.extension) is None:
            raise ValueError(f"extension {self.extension!r} must match ^[a-z0-9]+$")


@dataclass(frozen=True, slots=True)
class RawArtifact:
    """Content-addressed reference to one immutable archived payload."""

    relative_path: PurePosixPath
    sha256: str
    retrieved_at: datetime


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    """Lineage record binding a normalized partition to its raw source."""

    dataset: Dataset
    provider: str
    endpoint: str
    request_params: Mapping[str, JSONValue]
    retrieved_at: datetime
    raw_artifact: RawArtifact
    normalized_relative_path: PurePosixPath
    normalized_sha256: str
    row_count: int
    schema_version: str
    normalization_version: str
    quality_findings: tuple[QualityFinding, ...]


@dataclass(frozen=True, slots=True)
class DatasetArtifact:
    """Filesystem handles plus provenance for one written dataset partition."""

    normalized_path: Path
    manifest_path: Path
    manifest: DatasetManifest


class UntrustedDatasetError(RuntimeError):
    """Raised when a stored partition fails any lineage or content check."""


def canonical_frame_sha256(frame: pl.DataFrame, spec: DatasetSpec) -> str:
    """Hash schema version, canonical column order, and key-sorted rows.

    Equal frames with different provider row ordering produce identical
    digests; changing one cell or the ``schema_version`` changes the digest.

    Raises:
        ValueError: When the frame misses columns declared by the spec.
    """
    ordered_columns = [*spec.key, *(name for name in sorted(frame.columns) if name not in spec.key)]
    missing = [name for name in ordered_columns if name not in frame.columns]
    if missing:
        raise ValueError(f"frame misses canonical columns {sorted(missing)}")
    digest = hashlib.sha256()
    digest.update(f"schema_version={spec.schema_version}\n".encode())
    digest.update(("columns=" + ",".join(ordered_columns) + "\n").encode())
    canonical_rows = frame.select(ordered_columns).sort(ordered_columns)
    for row in canonical_rows.iter_rows():
        digest.update(json.dumps(row, default=str, separators=(",", ":")).encode())
    return digest.hexdigest()


def _atomic_create(path: Path, data: bytes) -> None:
    """Create-only durable write; an existing file is never truncated/replaced."""
    descriptor, temp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.tmp-")
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        with contextlib.suppress(FileExistsError):
            os.link(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def _manifest_document(manifest: DatasetManifest) -> dict[str, JSONValue]:
    """Relative-path, credential-free JSON projection of a manifest."""
    return {
        "dataset": str(manifest.dataset),
        "endpoint": manifest.endpoint,
        "normalization_version": manifest.normalization_version,
        "normalized_relative_path": manifest.normalized_relative_path.as_posix(),
        "normalized_sha256": manifest.normalized_sha256,
        "provider": manifest.provider,
        "quality_findings": [
            {
                "code": finding.code,
                "message": finding.message,
                "row_count": finding.row_count,
                "severity": str(finding.severity),
            }
            for finding in manifest.quality_findings
        ],
        "raw_artifact": {
            "relative_path": manifest.raw_artifact.relative_path.as_posix(),
            "retrieved_at": manifest.raw_artifact.retrieved_at.isoformat(),
            "sha256": manifest.raw_artifact.sha256,
        },
        "request_params": dict(manifest.request_params),
        "retrieved_at": manifest.retrieved_at.isoformat(),
        "row_count": manifest.row_count,
        "schema_version": manifest.schema_version,
    }


class DataStore:
    """Owns every filesystem write beneath the configured data root."""

    def __init__(self, settings: DataSettings) -> None:
        self._settings = settings

    def _resolve_under_root(self, relative: PurePosixPath) -> Path:
        if any(part in ("..", "") for part in relative.parts):
            raise ValueError(f"path {relative.as_posix()!r} would escape data_root")
        root = self._settings.resolved_data_root().resolve()
        target = (root / Path(*relative.parts)).resolve()
        if target != root and root not in target.parents:
            raise ValueError(f"path {relative.as_posix()!r} resolves outside data_root")
        return target

    def store_raw(self, dataset: Dataset, payload: RawPayload) -> RawArtifact:
        """Archive payload bytes once at their content address; never overwrite."""
        if _PROVIDER_PATTERN.fullmatch(payload.provider) is None:
            raise ValueError(f"provider name {payload.provider!r} is not path-safe")
        sha256 = hashlib.sha256(payload.content).hexdigest()
        relative = PurePosixPath("raw", payload.provider, str(dataset), sha256, f"payload.{payload.extension}")
        target = self._resolve_under_root(relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            _atomic_create(target, payload.content)
        stored_sha256 = hashlib.sha256(target.read_bytes()).hexdigest()
        if stored_sha256 != sha256:
            raise UntrustedDatasetError(f"existing raw artifact hash mismatch at {relative.as_posix()}")
        logger.info(
            "[DATA] event=raw_stored dataset=%s path=%s bytes=%d",
            str(dataset),
            relative.as_posix(),
            len(payload.content),
        )
        return RawArtifact(relative_path=relative, sha256=sha256, retrieved_at=payload.retrieved_at)

    def write_normalized(
        self,
        frame: pl.DataFrame,
        spec: DatasetSpec,
        raw_artifact: RawArtifact,
        payload: RawPayload,
        report: QualityReport,
        normalization_version: str = "1",
    ) -> DatasetArtifact:
        """Persist one validated, PIT-stamped frame with its bound manifest.

        Raises:
            DataQualityError: When ``report`` carries ERROR findings.
            LookAheadError: When any row becomes available after the frame's
                own latest ``available_at`` instant.
            ValueError: On empty frames, missing availability stamps, or paths
                escaping the data root.
        """
        if report.has_errors:
            raise DataQualityError(report)
        if frame.is_empty():
            raise ValueError("refusing to persist an empty normalized frame")
        if AVAILABLE_AT not in frame.columns:
            raise ValueError(f"normalized frame misses {AVAILABLE_AT!r} column")
        latest_available_at = frame.get_column(AVAILABLE_AT).max()
        if not isinstance(latest_available_at, datetime):
            raise ValueError(f"{AVAILABLE_AT!r} column does not carry timestamps")
        assert_no_lookahead(frame, latest_available_at)

        frame_sha256 = canonical_frame_sha256(frame, spec)
        normalized_relative = PurePosixPath(
            "normalized",
            str(spec.dataset),
            f"schema_version={spec.schema_version}",
            f"{frame_sha256}.parquet",
        )
        manifest_relative = PurePosixPath("manifests", str(spec.dataset), f"{frame_sha256}.json")
        parquet_path = self._resolve_under_root(normalized_relative)
        manifest_path = self._resolve_under_root(manifest_relative)

        manifest = DatasetManifest(
            dataset=spec.dataset,
            provider=payload.provider,
            endpoint=payload.endpoint,
            request_params=dict(payload.request_params),
            retrieved_at=payload.retrieved_at,
            raw_artifact=raw_artifact,
            normalized_relative_path=normalized_relative,
            normalized_sha256=frame_sha256,
            row_count=frame.height,
            schema_version=spec.schema_version,
            normalization_version=normalization_version,
            quality_findings=report.findings,
        )

        buffer = io.BytesIO()
        frame.write_parquet(buffer)
        if parquet_path.exists():
            if not self._parquet_matches(parquet_path, spec, frame.height, frame_sha256):
                _atomic_create(parquet_path, buffer.getvalue())
        else:
            parquet_path.parent.mkdir(parents=True, exist_ok=True)
            _atomic_create(parquet_path, buffer.getvalue())

        document = _manifest_document(manifest)
        serialized = json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")
        if manifest_path.exists():
            if not self._manifest_matches(manifest_path, document):
                _atomic_create(manifest_path, serialized)
        else:
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            _atomic_create(manifest_path, serialized)
        logger.info(
            "[DATA] event=normalized_written dataset=%s rows=%d frame_sha256=%s",
            str(spec.dataset),
            frame.height,
            frame_sha256,
        )
        return DatasetArtifact(normalized_path=parquet_path, manifest_path=manifest_path, manifest=manifest)

    def _parquet_matches(self, path: Path, spec: DatasetSpec, row_count: int, frame_sha256: str) -> bool:
        try:
            existing = pl.read_parquet(path)
            return existing.height == row_count and canonical_frame_sha256(existing, spec) == frame_sha256
        except (OSError, ValueError, ArithmeticError):
            return False

    def _manifest_matches(self, path: Path, document: Mapping[str, JSONValue]) -> bool:
        try:
            existing: JSONValue = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return False
        return existing == dict(document)

    def read_normalized(self, artifact: DatasetArtifact, spec: DatasetSpec) -> pl.DataFrame:
        """Return the stored frame only when every lineage check verifies.

        Raises:
            UntrustedDatasetError: On missing files, undecodable manifests,
                dataset/schema-version mismatches, wrong row counts, or any
                canonical/raw hash divergence.
        """
        root = self._settings.resolved_data_root().resolve()
        parquet_path = Path(artifact.normalized_path).resolve()
        manifest_path = Path(artifact.manifest_path).resolve()
        for path in (parquet_path, manifest_path):
            if path != root and root not in path.parents:
                raise UntrustedDatasetError(f"{path.as_posix()} is outside data_root {root.as_posix()}")
            if not path.is_file():
                raise UntrustedDatasetError(f"required file missing: {path.as_posix()}")
        document = self._load_manifest(manifest_path)
        if document.get("dataset") != str(spec.dataset) or document.get("schema_version") != spec.schema_version:
            raise UntrustedDatasetError(f"manifest identity mismatch at {manifest_path.as_posix()}")
        frame = pl.read_parquet(parquet_path)
        row_count = document.get("row_count")
        if isinstance(row_count, bool) or not isinstance(row_count, int) or row_count != frame.height:
            raise UntrustedDatasetError(f"manifest row count {row_count!r} != parquet rows {frame.height}")
        actual_sha256 = canonical_frame_sha256(frame, spec)
        recorded_sha256 = document.get("normalized_sha256")
        if not isinstance(recorded_sha256, str) or _HEX64_PATTERN.fullmatch(recorded_sha256) is None:
            raise UntrustedDatasetError("manifest normalized_sha256 is malformed")
        if recorded_sha256 != actual_sha256 or recorded_sha256 != artifact.manifest.normalized_sha256:
            raise UntrustedDatasetError(f"canonical frame hash mismatch at {parquet_path.as_posix()}")
        raw_section = document.get("raw_artifact")
        raw_hash = raw_section.get("sha256") if isinstance(raw_section, dict) else None
        if raw_hash != artifact.manifest.raw_artifact.sha256:
            raise UntrustedDatasetError("manifest raw-payload hash binding failed")
        logger.info(
            "[DATA] event=normalized_read dataset=%s rows=%d frame_sha256=%s",
            str(spec.dataset),
            frame.height,
            actual_sha256,
        )
        return frame

    @staticmethod
    def _load_manifest(manifest_path: Path) -> dict[str, JSONValue]:
        try:
            document: JSONValue = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise UntrustedDatasetError(f"manifest unreadable at {manifest_path.as_posix()}: {exc}") from exc
        if not isinstance(document, dict):
            raise UntrustedDatasetError(f"manifest root must be an object: {manifest_path.as_posix()}")
        return document
