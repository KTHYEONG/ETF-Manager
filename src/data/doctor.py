"""Dry inspection, supported repair planning, and guarded maintenance for Silver and Bronze."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import TypeAlias

import polars as pl

from src.data.calendar import DEFAULT_CALENDAR_NAME, load_calendar
from src.data.catalog import (
    _load_manifest_document,
    _reconstruct_manifest,
    clear_catalog_frame_cache,
)
from src.data.merge import PriorPartition, assert_key_coverage, plan_history_recovery
from src.data.quality import validate_frame
from src.data.retention import apply_prune, plan_prune
from src.data.schema import AvailabilityKind, Dataset, spec_for
from src.data.settings import DataSettings
from src.data.storage import (
    DatasetArtifact,
    DatasetManifest,
    DataStore,
    RawPayload,
    UntrustedDatasetError,
)

logger = logging.getLogger(__name__)

BronzeFetcher: TypeAlias = Callable[[Dataset, DatasetManifest], bytes | None]


class UnsupportedSourceError(UntrustedDatasetError):
    """Recorded Bronze provenance has no supported re-fetch path for this repair."""


class SourceMismatchError(UntrustedDatasetError):
    """A re-fetched Bronze payload does not match the manifest's recorded SHA."""


class SilverCondition(StrEnum):
    """Latest-partition health for one dataset; every variant names its dataset and path."""

    HEALTHY = "healthy"
    MISSING_BRONZE = "missing_bronze"
    DAMAGED_SILVER = "damaged_silver"
    UNRECOVERABLE = "unrecoverable"


@dataclass(frozen=True, slots=True)
class DatasetHealth:
    """Inspection outcome for the latest partition of one dataset."""

    dataset: Dataset
    condition: SilverCondition
    manifest_path: Path
    detail: str


@dataclass(frozen=True, slots=True)
class DataHealthReport:
    """Per-dataset Silver status from a dry inspection without mutation."""

    datasets: tuple[DatasetHealth, ...]


@dataclass(frozen=True, slots=True)
class BronzeRepair:
    """Restore one missing Bronze payload from its recorded source."""

    dataset: Dataset
    manifest_path: Path
    manifest: DatasetManifest


@dataclass(frozen=True, slots=True)
class SilverRebuild:
    """Rebuild one damaged Silver partition from trusted history."""

    dataset: Dataset
    damaged_stem: str


@dataclass(frozen=True, slots=True)
class DataMaintenancePlan:
    """Immutable dry actions with the manifest snapshot they were derived from."""

    findings: DataHealthReport
    repairs: tuple[BronzeRepair | SilverRebuild, ...]
    snapshot: tuple[tuple[str, str], ...]


def _discover_datasets(root: Path) -> list[Dataset]:
    """List datasets that own a manifests directory, oldest discovery order."""
    manifests_root = root / "manifests"
    if not manifests_root.is_dir():
        return []
    found: list[Dataset] = []
    for child in sorted(manifests_root.iterdir()):
        if not child.is_dir():
            continue
        try:
            found.append(Dataset(child.name))
        except ValueError:
            continue
    return found


def _manifest_stems(root: Path) -> tuple[tuple[str, str], ...]:
    """Sorted manifest identity pairs used as the plan snapshot and staleness guard."""
    pairs: list[tuple[str, str]] = []
    for dataset in _discover_datasets(root):
        manifests_dir = root / "manifests" / str(dataset)
        pairs.extend((str(dataset), path.stem) for path in sorted(manifests_dir.glob("*.json")))
    return tuple(sorted(pairs))


def inspect_data(settings: DataSettings) -> DataHealthReport:
    """Inspect latest Silver and repairable Bronze conditions for every registered dataset.

    Args:
        settings: Repository data root to inspect without mutation.

    Returns:
        Per-dataset Silver status and actionable Bronze or history-recovery findings.

    Raises:
        UntrustedDatasetError: If catalog metadata cannot be interpreted safely.
    """
    root = settings.resolved_data_root()
    store = DataStore(settings)
    outcomes: list[DatasetHealth] = []
    for dataset in _discover_datasets(root):
        spec = spec_for(dataset)
        manifests_dir = root / "manifests" / str(dataset)
        candidates = sorted(manifests_dir.glob("*.json"))
        if not candidates:
            continue
        manifests = [(_reconstruct_manifest(_load_manifest_document(path), path), path) for path in candidates]
        manifests.sort(key=lambda item: (item[0].retrieved_at, item[0].normalized_sha256, item[1].name))
        latest, latest_path = manifests[-1]
        artifact = DatasetArtifact(
            normalized_path=root.joinpath(*latest.normalized_relative_path.parts),
            manifest_path=latest_path,
            manifest=latest,
        )
        try:
            store.read_normalized(artifact, spec)
        except UntrustedDatasetError as exc:
            trusted_older = False
            for older, older_path in reversed(manifests[:-1]):
                older_artifact = DatasetArtifact(
                    normalized_path=root.joinpath(*older.normalized_relative_path.parts),
                    manifest_path=older_path,
                    manifest=older,
                )
                try:
                    store.read_normalized(older_artifact, spec)
                except UntrustedDatasetError:
                    continue
                trusted_older = True
                break
            condition = SilverCondition.DAMAGED_SILVER if trusted_older else SilverCondition.UNRECOVERABLE
            outcomes.append(
                DatasetHealth(
                    dataset=dataset,
                    condition=condition,
                    manifest_path=latest_path,
                    detail=(
                        f"dataset={dataset!s} manifest={latest_path.as_posix()} "
                        f"condition={condition.value} reason={exc}"
                    ),
                )
            )
            continue
        raw_path = root.joinpath(*latest.raw_artifact.relative_path.parts)
        condition = SilverCondition.HEALTHY if raw_path.is_file() else SilverCondition.MISSING_BRONZE
        outcomes.append(
            DatasetHealth(
                dataset=dataset,
                condition=condition,
                manifest_path=latest_path,
                detail=(
                    f"dataset={dataset!s} manifest={latest_path.as_posix()} "
                    f"raw={raw_path.as_posix()} condition={condition.value}"
                ),
            )
        )
    return DataHealthReport(datasets=tuple(outcomes))


def plan_data_maintenance(settings: DataSettings) -> DataMaintenancePlan:
    """Combine verified health findings, supported repairs, and protected retention into one dry plan.

    Args:
        settings: Data root containing the stored source metadata and run references.

    Returns:
        Idempotent actions ordered so repairs and verification precede pruning.

    Raises:
        UntrustedDatasetError: If a required source cannot be recovered or a protected Silver pin is unresolved.
    """
    report = inspect_data(settings)
    repairs: list[BronzeRepair | SilverRebuild] = []
    for health in report.datasets:
        if health.condition is SilverCondition.HEALTHY:
            continue
        if health.condition is SilverCondition.MISSING_BRONZE:
            manifest = _reconstruct_manifest(
                _load_manifest_document(health.manifest_path), health.manifest_path
            )
            repairs.append(BronzeRepair(dataset=health.dataset, manifest_path=health.manifest_path, manifest=manifest))
        elif health.condition is SilverCondition.DAMAGED_SILVER:
            repairs.append(SilverRebuild(dataset=health.dataset, damaged_stem=health.manifest_path.stem))
        else:
            raise UntrustedDatasetError(
                f"unrecoverable silver for dataset {health.dataset.value!r} at "
                f"{health.manifest_path.as_posix()}; no trusted partition remains to rebuild from"
            )
    root = settings.resolved_data_root()
    return DataMaintenancePlan(findings=report, repairs=tuple(repairs), snapshot=_manifest_stems(root))


def apply_data_maintenance(
    plan: DataMaintenancePlan,
    settings: DataSettings,
    *,
    fetcher: BronzeFetcher | None = None,
) -> DataHealthReport:
    """Apply an inspected plan and verify its postconditions before any deletion.

    Args:
        plan: Immutable actions derived from a previous inspection.
        settings: Root whose current identity must still match the plan.

    Returns:
        Health report after repair, rebuilding where required, and safe pruning.

    Raises:
        UntrustedDatasetError: If source bytes, the plan snapshot, or rebuilt Silver fail verification.
    """
    root = settings.resolved_data_root()
    if _manifest_stems(root) != plan.snapshot:
        raise UntrustedDatasetError("maintenance plan is stale; re-inspect before applying")
    store = DataStore(settings)
    for repair in plan.repairs:
        if isinstance(repair, BronzeRepair):
            _restore_bronze(repair, settings, store, fetcher)
        else:
            _rebuild_silver(repair, settings, store)
            logger.info("[DATA] event=silver_rebuilt dataset=%s", str(repair.dataset))
    inspect_data(settings)
    apply_prune(plan_prune(settings), dry_run=False)
    return inspect_data(settings)


def _rebuild_silver(repair: SilverRebuild, settings: DataSettings, store: DataStore) -> None:
    """Publish the trusted-history union as the new latest with the damaged stem as prior."""
    recovery = plan_history_recovery(settings, repair.dataset)
    if recovery is None:
        raise UntrustedDatasetError(
            f"silver rebuild unavailable for dataset {repair.dataset.value!r}; "
            "recoverable inputs are incomplete"
        )
    root = settings.resolved_data_root()
    spec = spec_for(repair.dataset)
    manifests_dir = root / "manifests" / str(repair.dataset)
    trusted: list[tuple[DatasetArtifact, pl.DataFrame]] = []
    for stem in recovery.source_manifest_sha256s:
        path = manifests_dir / f"{stem}.json"
        manifest = _reconstruct_manifest(_load_manifest_document(path), path)
        artifact = DatasetArtifact(
            normalized_path=root.joinpath(*manifest.normalized_relative_path.parts),
            manifest_path=path,
            manifest=manifest,
        )
        trusted.append((artifact, store.read_normalized(artifact, spec)))
    trusted.sort(key=lambda item: (item[0].manifest.retrieved_at, item[0].manifest.normalized_sha256))
    baseline_artifact, baseline_frame = trusted[-1]
    assert_key_coverage(
        PriorPartition(artifact=baseline_artifact, frame=baseline_frame),
        recovery.recovered,
        repair.dataset,
    )
    damaged_path = manifests_dir / f"{repair.damaged_stem}.json"
    damaged_manifest = _reconstruct_manifest(_load_manifest_document(damaged_path), damaged_path)
    calendar = None
    if spec.availability.kind is AvailabilityKind.SESSION_CLOSE:
        calendar = load_calendar(spec.availability.calendar_name or DEFAULT_CALENDAR_NAME)
    report = validate_frame(recovery.recovered, spec, calendar)
    payload = RawPayload(
        provider=damaged_manifest.provider,
        endpoint=damaged_manifest.endpoint,
        request_params=dict(damaged_manifest.request_params),
        retrieved_at=datetime.now(UTC),
        extension="json",
        content=b"",
    )
    store.write_normalized(
        recovery.recovered,
        spec,
        damaged_manifest.raw_artifact,
        payload,
        report,
        damaged_manifest.normalization_version,
        repair.damaged_stem,
    )
    logger.info(
        "[DATA] event=silver_rebuilt dataset=%s rows=%d",
        str(repair.dataset),
        recovery.recovered_rows,
    )
    damaged_path.unlink(missing_ok=True)
    if all(
        artifact.manifest.normalized_relative_path != damaged_manifest.normalized_relative_path
        for artifact, _ in trusted
    ):
        root.joinpath(*damaged_manifest.normalized_relative_path.parts).unlink(missing_ok=True)
    clear_catalog_frame_cache()


def _restore_bronze(
    repair: BronzeRepair,
    settings: DataSettings,
    store: DataStore,
    fetcher: BronzeFetcher | None,
) -> None:
    """Fetch, hash-verify, and archive one Bronze payload at its recorded address."""
    manifest = repair.manifest
    response = fetcher(repair.dataset, manifest) if fetcher is not None else None
    if response is None:
        raise UnsupportedSourceError(
            f"bronze re-fetch unsupported for dataset {repair.dataset.value!r} "
            f"provider={manifest.provider!r} endpoint={manifest.endpoint!r}"
        )
    if hashlib.sha256(response).hexdigest() != manifest.raw_artifact.sha256:
        raise SourceMismatchError(
            f"re-fetched bronze for dataset {repair.dataset.value!r} does not match "
            f"recorded sha at {repair.manifest_path.as_posix()}"
        )
    raw_relative: PurePosixPath = manifest.raw_artifact.relative_path
    try:
        payload = RawPayload(
            provider=manifest.provider,
            endpoint=manifest.endpoint,
            request_params=dict(manifest.request_params),
            retrieved_at=manifest.raw_artifact.retrieved_at,
            extension=raw_relative.suffix.lstrip("."),
            content=response,
        )
        store.store_raw(repair.dataset, payload)
    except (ValueError, OSError) as exc:
        raise UntrustedDatasetError(
            f"bronze restore failed verification for dataset {repair.dataset.value!r}: {exc}"
        ) from exc
    logger.info(
        "[DATA] event=bronze_restored dataset=%s raw=%s",
        str(repair.dataset),
        raw_relative.as_posix(),
    )
