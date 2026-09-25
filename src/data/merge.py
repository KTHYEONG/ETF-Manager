"""Shared prior-partition loader and key-coverage merge for incremental ingest."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime

import polars as pl

from src.data.calendar import DEFAULT_CALENDAR_NAME, load_calendar
from src.data.catalog import (
    _load_manifest_document,
    _reconstruct_manifest,
    clear_catalog_frame_cache,
    latest_artifact,
)
from src.data.pit import AVAILABLE_AT
from src.data.quality import validate_frame
from src.data.schema import AvailabilityKind, Dataset, spec_for
from src.data.settings import DataSettings
from src.data.storage import DatasetArtifact, DataStore, RawPayload, UntrustedDatasetError

logger = logging.getLogger(__name__)


class PriorPartitionUntrustedError(UntrustedDatasetError):
    """Latest manifest exists but fails verification; ingest must stop."""


class PartitionKeyLossError(ValueError):
    """A prior key outside the refreshed set is absent from the merge result."""


@dataclass(frozen=True, slots=True)
class PriorPartition:
    """Verified latest artifact with its normalized frame."""

    artifact: DatasetArtifact
    frame: pl.DataFrame


def load_prior_partition(settings: DataSettings, dataset: Dataset) -> PriorPartition | None:
    """Load the latest trusted partition for an incremental merge.

    Returns None only when the dataset has no manifest at all (first ingest). If any manifest exists but the latest partition cannot be verified, ingest must stop: continuing would publish only the freshly fetched slice as the dataset's newest view.

    Args:
        settings: Data root of the catalog.
        dataset: Dataset being ingested.

    Returns:
        The verified latest artifact with its frame, or None for a dataset that has never been ingested.

    Raises:
        PriorPartitionUntrustedError: If manifests exist but the latest one fails verification. The message names the dataset and states that the underlying trust failure must be repaired (or recovered from history) before re-ingesting.
    """
    root = settings.resolved_data_root()
    manifests_dir = root / "manifests" / str(dataset)
    candidates = sorted(manifests_dir.glob("*.json")) if manifests_dir.is_dir() else []
    if not candidates:
        return None
    try:
        artifact = latest_artifact(settings, dataset)
        frame = DataStore(settings).read_normalized(artifact, spec_for(dataset))
    except UntrustedDatasetError as exc:
        raise PriorPartitionUntrustedError(
            f"prior partition for dataset {str(dataset)!r} is untrusted: {exc}; "
            "repair the underlying trust failure (or recover from history) before re-ingesting"
        ) from exc
    return PriorPartition(artifact=artifact, frame=frame)


def _drop_availability(frame: pl.DataFrame) -> pl.DataFrame:
    if AVAILABLE_AT in frame.columns:
        return frame.drop(AVAILABLE_AT)
    return frame


def _refreshed_mask(prior_keys: pl.DataFrame, refreshed: pl.DataFrame) -> pl.Series:
    join_cols = [c for c in refreshed.columns if c in prior_keys.columns]
    if not join_cols:
        raise ValueError("refreshed frame must share at least one key column with the dataset key")
    marker = "__refreshed__"
    tagged = refreshed.select(join_cols).unique().with_columns(pl.lit(True).alias(marker))
    joined = prior_keys.join(tagged, on=join_cols, how="left")
    return joined.get_column(marker).fill_null(False)


def assert_key_coverage(
    prior: PriorPartition,
    result: pl.DataFrame,
    dataset: Dataset,
    *,
    refreshed: pl.DataFrame | None = None,
) -> None:
    """Raise PartitionKeyLossError when a non-refreshed prior key is missing."""
    spec = spec_for(dataset)
    key = list(spec.key)
    prior_clean = _drop_availability(prior.frame).select(key)
    result_keys = _drop_availability(result).select(key).unique()
    if refreshed is not None and refreshed.height > 0 and len(refreshed.columns) > 0:
        mask = _refreshed_mask(prior_clean, refreshed)
        owed = prior_clean.filter(~mask).unique()
    else:
        owed = prior_clean.unique()
    if owed.is_empty():
        return
    missing = owed.join(result_keys, on=key, how="anti")
    if not missing.is_empty():
        sample = missing.head(3).to_dicts()
        raise PartitionKeyLossError(
            f"merge for dataset {str(dataset)!r} would drop {missing.height} prior key(s) "
            f"outside refreshed; e.g. {sample}"
        )


def merge_incremental(
    prior: PriorPartition | None,
    incoming: pl.DataFrame,
    dataset: Dataset,
    *,
    refreshed: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Combine the prior partition with this ingest's rows without losing any prior key.

    Incoming rows win over prior rows that share a dataset key. Keys listed in `refreshed` (a frame of key columns) are dropped from the prior side before the union, which is how a dataset replaces a whole series or ticker.

    Args:
        prior: Verified prior partition, or None on first ingest.
        incoming: Newly fetched rows in the dataset's spec columns.
        dataset: Dataset whose spec key defines row identity.
        refreshed: Optional key-column frame of prior rows this ingest intentionally supersedes.

    Returns:
        Deduplicated frame in spec column order and dtypes; equals `incoming` when `prior` is None.

    Raises:
        PartitionKeyLossError: If a prior key outside `refreshed` is absent from the result.
    """
    spec = spec_for(dataset)
    key = list(spec.key)
    columns = list(spec.columns.keys())
    schema = pl.Schema(dict(spec.columns))
    incoming_clean = _drop_availability(incoming).select(columns).cast(schema)
    if prior is None:
        result = incoming_clean.unique(subset=key, keep="last", maintain_order=False).sort(key)
        logger.info("[DATA] event=merge_incremental dataset=%s rows=%d prior=none", str(dataset), result.height)
        return result
    prior_clean = _drop_availability(prior.frame).select(columns).cast(schema)
    if refreshed is not None and refreshed.height > 0 and len(refreshed.columns) > 0:
        join_cols = [c for c in refreshed.columns if c in prior_clean.columns]
        if not join_cols:
            raise ValueError("refreshed frame must share at least one key column with the dataset key")
        for col in join_cols:
            if col not in key:
                raise ValueError(f"refreshed column {col!r} is not part of dataset key {key!r}")
        prior_filtered = prior_clean.join(refreshed.select(join_cols).unique(), on=join_cols, how="anti")
    else:
        prior_filtered = prior_clean
    combined = pl.concat([prior_filtered, incoming_clean], how="vertical")
    result = combined.unique(subset=key, keep="last", maintain_order=False).sort(key)
    assert_key_coverage(prior, result, dataset, refreshed=refreshed)
    logger.info(
        "[DATA] event=merge_incremental dataset=%s rows=%d prior_rows=%d",
        str(dataset),
        result.height,
        prior.frame.height,
    )
    return result


@dataclass(frozen=True, slots=True)
class RecoveryPlan:
    dataset: Dataset
    latest_manifest_sha256: str
    source_manifest_sha256s: tuple[str, ...]
    latest_rows: int
    recovered_rows: int
    recovered: pl.DataFrame


def _verified_partitions(settings: DataSettings, dataset: Dataset) -> list[tuple[DatasetArtifact, pl.DataFrame]]:
    """Collect every manifest of a dataset that passes storage verification, oldest first."""
    spec = spec_for(dataset)
    root = settings.resolved_data_root()
    manifests_dir = root / "manifests" / str(dataset)
    candidates = sorted(manifests_dir.glob("*.json")) if manifests_dir.is_dir() else []
    store = DataStore(settings)
    trusted: list[tuple[DatasetArtifact, pl.DataFrame]] = []
    for path in candidates:
        try:
            manifest = _reconstruct_manifest(_load_manifest_document(path), path)
            artifact = DatasetArtifact(
                normalized_path=root.joinpath(*manifest.normalized_relative_path.parts),
                manifest_path=path,
                manifest=manifest,
            )
            frame = store.read_normalized(artifact, spec)
        except UntrustedDatasetError as exc:
            logger.warning("[DATA] event=recover_skip_untrusted manifest=%s reason=%s", path.name, exc)
            continue
        trusted.append((artifact, frame))
    trusted.sort(
        key=lambda item: (
            item[0].manifest.retrieved_at,
            item[0].manifest.normalized_sha256,
            item[0].manifest_path.name,
        )
    )
    return trusted


def plan_history_recovery(settings: DataSettings, dataset: Dataset) -> RecoveryPlan | None:
    """Union every trusted partition of a dataset and report whether the latest one lost keys.

    Manifests that fail verification are skipped with a `[DATA]` warning naming the file; they never contribute rows. Later `retrieved_at` wins on key collisions.

    Args:
        settings: Data root of the catalog.
        dataset: Dataset to inspect.

    Returns:
        A plan when the union has keys the latest trusted partition lacks; None when the latest partition already covers the union or when fewer than two trusted partitions exist.

    Raises:
        PriorPartitionUntrustedError: If no partition of the dataset can be verified.
    """
    spec = spec_for(dataset)
    key = list(spec.key)
    root = settings.resolved_data_root()
    manifests_dir = root / "manifests" / str(dataset)
    candidates = sorted(manifests_dir.glob("*.json")) if manifests_dir.is_dir() else []
    trusted = _verified_partitions(settings, dataset)
    if not trusted:
        if not candidates:
            return None
        raise PriorPartitionUntrustedError(
            f"no verified partition for dataset {str(dataset)!r}; "
            "repair the underlying trust failure (or recover from history) before re-ingesting"
        )
    if len(trusted) < 2:
        return None
    latest_verified, latest_frame = trusted[-1]
    union = (
        pl.concat([frame for _, frame in trusted], how="vertical")
        .unique(subset=key, keep="last", maintain_order=False)
        .sort(key)
    )
    latest_keys = latest_frame.select(key).unique()
    union_keys = union.select(key).unique()
    if union_keys.join(latest_keys, on=key, how="anti").is_empty():
        return None
    logger.info(
        "[DATA] event=recover_planned dataset=%s latest_rows=%d recovered_rows=%d sources=%d",
        str(dataset),
        latest_verified.manifest.row_count,
        union.height,
        len(trusted),
    )
    return RecoveryPlan(
        dataset=dataset,
        latest_manifest_sha256=latest_verified.manifest_path.stem,
        source_manifest_sha256s=tuple(artifact.manifest_path.stem for artifact, _ in trusted),
        latest_rows=latest_verified.manifest.row_count,
        recovered_rows=union.height,
        recovered=union,
    )


def apply_history_recovery(plan: RecoveryPlan, settings: DataSettings) -> DatasetArtifact:
    """Publish the recovered union as the dataset's new latest partition.

    Args:
        plan: Result of `plan_history_recovery` for the same data root.
        settings: Data root of the catalog.

    Returns:
        The written artifact; its manifest records the previous latest manifest as `prior_manifest_sha256`.

    Raises:
        PartitionKeyLossError: If the recovered frame misses any key of the latest partition.
        UntrustedDatasetError: If the latest manifest changed since planning.
    """
    spec = spec_for(plan.dataset)
    store = DataStore(settings)
    current = latest_artifact(settings, plan.dataset)
    if current.manifest_path.stem != plan.latest_manifest_sha256:
        raise UntrustedDatasetError(
            f"latest manifest for {str(plan.dataset)!r} changed since planning; re-plan recovery"
        )
    current_frame = store.read_normalized(current, spec)
    assert_key_coverage(PriorPartition(artifact=current, frame=current_frame), plan.recovered, plan.dataset)
    recovered_at = datetime.now(UTC)
    calendar = None
    if spec.availability.kind is AvailabilityKind.SESSION_CLOSE:
        calendar = load_calendar(spec.availability.calendar_name or DEFAULT_CALENDAR_NAME)
    report = validate_frame(plan.recovered, spec, calendar)
    payload = RawPayload(
        provider=current.manifest.provider,
        endpoint=current.manifest.endpoint,
        request_params=dict(current.manifest.request_params),
        retrieved_at=recovered_at,
        extension="json",
        content=b"",
    )
    artifact = store.write_normalized(
        plan.recovered,
        spec,
        current.manifest.raw_artifact,
        payload,
        report,
        current.manifest.normalization_version,
        plan.latest_manifest_sha256,
    )
    logger.info(
        "[DATA] event=recover_written dataset=%s rows=%d manifest=%s",
        str(plan.dataset),
        artifact.manifest.row_count,
        artifact.manifest_path.stem,
    )
    clear_catalog_frame_cache()
    return artifact
