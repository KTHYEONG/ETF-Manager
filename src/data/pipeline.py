"""Write-path seam: validate a normalized frame and persist it with lineage."""

from __future__ import annotations

import logging

import polars as pl

from src.data.calendar import DEFAULT_CALENDAR_NAME, TradingCalendar, load_calendar
from src.data.pit import stamp_availability
from src.data.quality import enforce, validate_frame
from src.data.schema import AvailabilityKind, Dataset, spec_for
from src.data.settings import DataSettings
from src.data.storage import DatasetArtifact, DataStore, RawPayload

logger = logging.getLogger(__name__)


def ingest(raw: pl.DataFrame, dataset: Dataset, *, calendar_name: str = DEFAULT_CALENDAR_NAME) -> pl.DataFrame:
    """Attach point-in-time availability, then validate and enforce quality.

    Args:
        raw: Normalized provider frame matching the dataset spec columns.
        dataset: Dataset identity resolved through the spec registry.
        calendar_name: Exchange calendar code used for session-close availability.

    Returns:
        The validated, availability-stamped frame unchanged.

    Raises:
        DataQualityError: When the stamped frame violates any ERROR predicate.
        ValueError: On schema-invalid arguments or naive timestamps.
    """
    spec = spec_for(dataset)
    calendar: TradingCalendar | None = None
    if spec.availability.kind is AvailabilityKind.SESSION_CLOSE:
        calendar = load_calendar(calendar_name)
    if spec.observation_column not in raw.columns:
        raise ValueError(f"frame for {dataset!r} is missing observation column {spec.observation_column!r}")
    stamped = stamp_availability(raw, spec, calendar)
    report = validate_frame(stamped, spec, calendar)
    enforce(report)
    logger.info("[DATA] event=ingest dataset=%s rows=%d", str(dataset), stamped.height)
    return stamped


def persist_ingest(
    raw: pl.DataFrame,
    dataset: Dataset,
    payload: RawPayload,
    settings: DataSettings,
    *,
    calendar_name: str = DEFAULT_CALENDAR_NAME,
    normalization_version: str = "1",
) -> DatasetArtifact:
    """Archive the raw payload, run :func:`ingest`, then persist the partition.

    The raw artifact is stored first so failed validation still leaves the
    immutable source bytes available for diagnosis; no normalized Parquet or
    manifest is written when validation raises.

    Raises:
        DataQualityError: When validation fails after the raw archive step.
        UntrustedDatasetError: When existing artifacts fail hash verification.
        ValueError: On invalid arguments or paths escaping the data root.
    """
    store = DataStore(settings)
    raw_artifact = store.store_raw(dataset, payload)
    stamped = ingest(raw, dataset, calendar_name=calendar_name)
    spec = spec_for(dataset)
    calendar = load_calendar(calendar_name) if spec.availability.kind is AvailabilityKind.SESSION_CLOSE else None
    report = validate_frame(stamped, spec, calendar)
    artifact = store.write_normalized(stamped, spec, raw_artifact, payload, report, normalization_version)
    logger.info(
        "[DATA] event=persist_ingest dataset=%s rows=%d frame_sha256=%s",
        str(dataset),
        stamped.height,
        artifact.manifest.normalized_sha256,
    )
    return artifact
