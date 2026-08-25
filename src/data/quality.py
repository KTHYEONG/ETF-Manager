"""Pure fail-closed data-quality gate: findings are reported, never repaired."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from typing import Final

import polars as pl

from src.data.calendar import TradingCalendar
from src.data.pit import AVAILABLE_AT
from src.data.schema import AvailabilityKind, Dataset, DatasetSpec, MissingPolicy

logger = logging.getLogger(__name__)

_RETURN_OUTLIER_THRESHOLD: Final[float] = math.log(2.0)
_OHLC_COLUMNS: Final[tuple[str, ...]] = ("open", "high", "low", "close")


class FindingSeverity(StrEnum):
    """Severity of a single validation finding."""

    ERROR = "error"
    WARN = "warn"


@dataclass(frozen=True, slots=True)
class QualityFinding:
    """One deterministic predicate result over a validated frame."""

    code: str
    severity: FindingSeverity
    message: str
    row_count: int


@dataclass(frozen=True, slots=True)
class QualityReport:
    """Aggregated validation outcome for one dataset frame."""

    dataset: Dataset
    findings: tuple[QualityFinding, ...]
    checked_rows: int

    @property
    def has_errors(self) -> bool:
        return any(finding.severity is FindingSeverity.ERROR for finding in self.findings)


class DataQualityError(ValueError):
    """Sole error boundary raised by ``enforce`` for ERROR-carrying reports."""

    report: QualityReport

    def __init__(self, report: QualityReport) -> None:
        errors = [finding for finding in report.findings if finding.severity is FindingSeverity.ERROR]
        detail = "; ".join(f"{finding.code}({finding.row_count}): {finding.message}" for finding in errors)
        super().__init__(
            f"data quality failed for {report.dataset!s} rows={report.checked_rows}: {detail}"
        )
        self.report = report


def _schema_findings(frame: pl.DataFrame, spec: DatasetSpec) -> list[QualityFinding]:
    expected_names = {*spec.columns, AVAILABLE_AT}
    actual_names = set(frame.columns)
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        unexpected = sorted(actual_names - expected_names)
        message = f"column set mismatch: missing={missing} unexpected={unexpected}"
        return [QualityFinding(code="SCHEMA_COLUMNS", severity=FindingSeverity.ERROR, message=message, row_count=0)]
    return [
        QualityFinding(
            code="SCHEMA_DTYPE",
            severity=FindingSeverity.ERROR,
            message=f"column {name!r} dtype {frame.schema[name]!r} != declared {dtype!r}",
            row_count=0,
        )
        for name, dtype in spec.columns.items()
        if frame.schema[name] != dtype
    ]


def _key_and_null_findings(frame: pl.DataFrame, spec: DatasetSpec) -> list[QualityFinding]:
    findings: list[QualityFinding] = []
    group_counts = frame.group_by(list(spec.key)).len()
    duplicated_rows = int(group_counts.filter(pl.col("len") > 1).get_column("len").sum() or 0)
    if duplicated_rows > 0:
        findings.append(
            QualityFinding(
                code="KEY_DUPLICATE",
                severity=FindingSeverity.ERROR,
                message=f"key {spec.key} repeats across {duplicated_rows} row(s)",
                row_count=duplicated_rows,
            )
        )
    required = [name for name in spec.columns if name not in spec.nullable_columns]
    null_mask = pl.any_horizontal([pl.col(name).is_null() for name in (*required, AVAILABLE_AT)])
    null_rows = frame.filter(null_mask).height
    if null_rows > 0:
        findings.append(
            QualityFinding(
                code="REQUIRED_NULL",
                severity=FindingSeverity.ERROR,
                message=f"nulls present in required columns over {null_rows} row(s)",
                row_count=null_rows,
            )
        )
    return findings


def _availability_order_finding(frame: pl.DataFrame, spec: DatasetSpec) -> QualityFinding | None:
    observation_dtype = frame.schema[spec.observation_column]
    if isinstance(observation_dtype, pl.Datetime):
        observation_ts = pl.col(spec.observation_column)
    else:
        observation_ts = pl.col(spec.observation_column).cast(pl.Datetime("us")).dt.replace_time_zone("UTC")
    violations = frame.filter(pl.col(AVAILABLE_AT) < observation_ts)
    if violations.height == 0:
        return None
    return QualityFinding(
        code="AVAILABILITY_ORDER",
        severity=FindingSeverity.ERROR,
        message=f"{violations.height} row(s) become available before their observation timestamp",
        row_count=violations.height,
    )


def _ohlc_findings(frame: pl.DataFrame) -> list[QualityFinding]:
    if not set(_OHLC_COLUMNS).issubset(frame.columns):
        return []
    open_, high, low, close = (pl.col(name) for name in _OHLC_COLUMNS)
    non_positive = (open_ <= 0) | (high <= 0) | (low <= 0) | (close <= 0)
    inverted_low = low > pl.min_horizontal(open_, close)
    inverted_high = high < pl.max_horizontal(open_, close)
    violations = frame.filter(non_positive | inverted_low | inverted_high)
    if violations.height == 0:
        return []
    return [
        QualityFinding(
            code="OHLC_INVALID",
            severity=FindingSeverity.ERROR,
            message=f"OHLC invariant violated on {violations.height} price row(s)",
            row_count=violations.height,
        )
    ]


def _observation_date_series(frame: pl.DataFrame, spec: DatasetSpec) -> pl.Series:
    column = frame.get_column(spec.observation_column)
    if isinstance(column.dtype, pl.Datetime):
        return column.dt.date()
    return column.cast(pl.Date)


def _session_missing_finding(frame: pl.DataFrame, spec: DatasetSpec, calendar: TradingCalendar) -> QualityFinding | None:
    identifiers = [name for name in spec.key if name != spec.observation_column]
    partitions = frame.partition_by(identifiers, as_dict=True) if identifiers else {(): frame}
    missing_dates: list[date] = []
    for partition in partitions.values():
        observed = set(_observation_date_series(partition, spec).to_list())
        if not observed:
            continue
        expected = set(calendar.sessions(min(observed), max(observed)))
        missing_dates.extend(sorted(expected - observed))
    if not missing_dates:
        return None
    preview = ", ".join(day.isoformat() for day in sorted(missing_dates)[:5])
    suffix = "" if len(missing_dates) <= 5 else ", ..."
    return QualityFinding(
        code="SESSION_MISSING",
        severity=FindingSeverity.ERROR,
        message=f"missing calendar session(s): {preview}{suffix}",
        row_count=len(missing_dates),
    )


def _return_outlier_finding(frame: pl.DataFrame, spec: DatasetSpec) -> QualityFinding | None:
    if "close" not in frame.columns or frame.height < 2:
        return None
    identifiers = [name for name in spec.key if name != spec.observation_column]
    ordered = frame.sort([*identifiers, spec.observation_column])
    previous_close = pl.col("close").shift(1).over(identifiers) if identifiers else pl.col("close").shift(1)

    outlier_mask = (
        ((pl.col("close") / previous_close).log().abs() > _RETURN_OUTLIER_THRESHOLD)
        & pl.col("close").is_not_null()
        & previous_close.is_not_null()
        & (previous_close > 0)
    )
    outliers = ordered.filter(outlier_mask)
    if outliers.height == 0:
        return None
    return QualityFinding(
        code="RETURN_OUTLIER",
        severity=FindingSeverity.WARN,
        message=f"abs(log return) exceeds log(2) on {outliers.height} consecutive-row transition(s)",
        row_count=outliers.height,
    )


def validate_frame(frame: pl.DataFrame, spec: DatasetSpec, calendar: TradingCalendar | None = None) -> QualityReport:
    """Run every quality predicate and aggregate deterministic findings.

    Purity contract: the input frame is never mutated, sorted in place, filled,
    coerced, or dropped; invalid data yields findings, not exceptions.

    Args:
        frame: Candidate frame carrying ``available_at`` plus the spec columns.
        spec: Immutable dataset contract resolved from the registry.
        calendar: Required trading calendar when availability is SESSION_CLOSE.

    Returns:
        QualityReport with findings in fixed rule order.

    Raises:
        ValueError: Only for invalid arguments (e.g. SESSION_CLOSE without a calendar).
    """
    if spec.availability.kind is AvailabilityKind.SESSION_CLOSE and calendar is None:
        raise ValueError(f"SESSION_CLOSE validation for dataset {str(spec.dataset)!r} requires a calendar")
    findings: list[QualityFinding] = [*_schema_findings(frame, spec)]
    if not any(finding.code.startswith("SCHEMA_") for finding in findings):
        findings.extend(_key_and_null_findings(frame, spec))
        availability_finding = _availability_order_finding(frame, spec)
        if availability_finding is not None:
            findings.append(availability_finding)
        findings.extend(_ohlc_findings(frame))
        if spec.availability.kind is AvailabilityKind.SESSION_CLOSE and spec.missing_policy is MissingPolicy.FAIL:
            assert calendar is not None
            session_finding = _session_missing_finding(frame, spec, calendar)
            if session_finding is not None:
                findings.append(session_finding)
        outlier_finding = _return_outlier_finding(frame, spec)
        if outlier_finding is not None:
            findings.append(outlier_finding)
    report = QualityReport(dataset=spec.dataset, findings=tuple(findings), checked_rows=frame.height)
    logger.info(
        "[DATA] event=frame_validated dataset=%s rows=%d errors=%d warnings=%d",
        str(spec.dataset),
        report.checked_rows,
        sum(1 for finding in findings if finding.severity is FindingSeverity.ERROR),
        sum(1 for finding in findings if finding.severity is FindingSeverity.WARN),
    )
    return report


def enforce(report: QualityReport) -> None:
    """Raise :class:`DataQualityError` iff the report carries ERROR findings."""
    if report.has_errors:
        raise DataQualityError(report)
