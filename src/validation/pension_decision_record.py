"""Frozen pension decision record and post-cutoff review."""

from __future__ import annotations

import calendar as _calendar
import json
import logging
import math
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Final, Literal

import polars as pl

from src.sim.pension_monthly import WeightSchedule
from src.validation.pension_decision import PensionDecisionReport
from src.validation.pension_decision_config import PensionDecisionSpec

logger = logging.getLogger(__name__)

__all__ = [
    "PensionDecisionRecord",
    "PensionReviewState",
    "PensionReviewStatus",
    "evaluate_pension_review",
    "freeze_pension_decision",
    "load_pension_decision_record",
]

PensionReviewState = Literal["HOLD", "REVIEW_DUE", "INSUFFICIENT_DATA"]

_RECORD_KEYS: Final[tuple[str, ...]] = (
    "record_id",
    "frozen_at",
    "git_commit",
    "config_sha256",
    "manifest_hashes",
    "seen_history_cutoff",
    "status",
    "incumbent_id",
    "incumbent_schedule",
    "equivalent_ids",
    "benchmark_id",
    "benchmark_schedule",
    "review_every_months",
    "previous_record_id",
)

_FREEZABLE_STATUSES: Final[tuple[str, ...]] = ("ADOPT_CANDIDATE", "KEEP_BENCHMARK", "KEEP_INCUMBENT")


@dataclass(frozen=True, slots=True)
class PensionDecisionRecord:
    """Frozen pension decision: which schedule is held, against what, and when it is re-evaluated.

    Attributes:
        record_id: Stable identity (config name plus run id).
        frozen_at: Timezone-aware freeze instant.
        git_commit: Code identity at freeze.
        config_sha256: SHA-256 of the decision config bytes.
        manifest_hashes: Dataset to manifest identity consumed by the decision run.
        seen_history_cutoff: Last month-end any evidence used.
        status: Decision status copied from the report.
        incumbent_id: Held candidate id (the benchmark when status keeps it).
        incumbent_schedule: Weight schedule of the incumbent.
        equivalent_ids: Candidates judged equal growth at freeze.
        benchmark_id: Immutable benchmark candidate id.
        benchmark_schedule: Weight schedule of the benchmark.
        review_every_months: Re-evaluation cadence copied from the config.
        previous_record_id: Record this one supersedes, if any.
    """

    record_id: str
    frozen_at: datetime
    git_commit: str
    config_sha256: str
    manifest_hashes: dict[str, str]
    seen_history_cutoff: date
    status: str
    incumbent_id: str
    incumbent_schedule: WeightSchedule
    equivalent_ids: tuple[str, ...]
    benchmark_id: str
    benchmark_schedule: WeightSchedule
    review_every_months: int
    previous_record_id: str | None


@dataclass(frozen=True, slots=True)
class PensionReviewStatus:
    """Post-cutoff tracking state of a frozen pension decision."""

    record_id: str
    as_of: date
    months_observed: int
    incumbent_over_benchmark: float | None
    state: PensionReviewState


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    document: dict[str, object] = {}
    for key, value in pairs:
        if key in document:
            raise ValueError(f"duplicate JSON key {key!r}")
        document[key] = value
    return document


def _nonblank_string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"pension record {name} must be a non-blank string")
    return value.strip()


def _schedule_from_json(value: object, *, name: str) -> WeightSchedule:
    if not isinstance(value, dict):
        raise ValueError(f"pension record {name} must be an object")
    try:
        return WeightSchedule(
            start_weights=value["start_weights"],
            end_weights=value["end_weights"],
            glide_years=value["glide_years"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"pension record {name} is not a valid weight schedule: {exc}") from exc


def _schedule_to_json(schedule: WeightSchedule) -> dict[str, object]:
    return {
        "start_weights": dict(schedule.start_weights),
        "end_weights": dict(schedule.end_weights),
        "glide_years": schedule.glide_years,
    }


def _require_month_end(day: date, label: str) -> None:
    if day.day != _calendar.monthrange(day.year, day.month)[1]:
        raise ValueError(f"pension record {label} {day.isoformat()} is not a month-end")


def _month_ends_after(cutoff: date, last: date) -> tuple[date, ...]:
    out: list[date] = []
    year, month = cutoff.year, cutoff.month
    while True:
        month += 1
        if month > 12:
            year += 1
            month = 1
        end = date(year, month, _calendar.monthrange(year, month)[1])
        if end > last:
            break
        out.append(end)
    return tuple(out)


def freeze_pension_decision(
    report: PensionDecisionReport,
    spec: PensionDecisionSpec,
    *,
    output_dir: Path,
    frozen_at: datetime,
    git_commit: str,
    config_sha256: str,
    previous_record_id: str | None = None,
) -> Path:
    """Write the immutable decision record; refuse to overwrite an existing record.

    Raises:
        ValueError: If status is NO_DECISION, `frozen_at` is naive, or the record path already exists.
    """
    if report.status == "NO_DECISION":
        raise ValueError("pension NO_DECISION reports cannot be frozen")
    if frozen_at.tzinfo is None or frozen_at.utcoffset() is None:
        raise ValueError(f"pension frozen_at must be timezone-aware, got {frozen_at!r}")
    commit = _nonblank_string(git_commit, name="git_commit")
    digest = _nonblank_string(config_sha256, name="config_sha256")
    previous: str | None = None
    if previous_record_id is not None:
        previous = _nonblank_string(previous_record_id, name="previous_record_id")
    if report.selected_id is None:
        raise ValueError("pension report carries no selected candidate to freeze")
    if report.selected_id not in spec.candidates:
        raise ValueError(f"pension selected id {report.selected_id!r} is not among candidates")
    cutoff = max(spec.modern_end, spec.century_end)
    record_id = f"{spec.name}__{cutoff.isoformat()}__{digest[:16]}"
    path = Path(output_dir) / f"{record_id}.json"
    if path.exists():
        raise ValueError(f"pension decision record already exists: {path.as_posix()}")
    document: dict[str, object] = {
        "record_id": record_id,
        "frozen_at": frozen_at.astimezone(UTC).isoformat(),
        "git_commit": commit,
        "config_sha256": digest,
        "manifest_hashes": dict(report.manifest_hashes),
        "seen_history_cutoff": cutoff.isoformat(),
        "status": report.status,
        "incumbent_id": report.selected_id,
        "incumbent_schedule": _schedule_to_json(spec.candidates[report.selected_id]),
        "equivalent_ids": list(report.equivalent_ids),
        "benchmark_id": spec.benchmark_id,
        "benchmark_schedule": _schedule_to_json(spec.candidates[spec.benchmark_id]),
        "review_every_months": spec.review_every_months,
        "previous_record_id": previous,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logger.info(
        "[PORTFOLIO] event=pension_decision_frozen record=%s status=%s incumbent=%s",
        record_id,
        report.status,
        report.selected_id,
    )
    return path


def load_pension_decision_record(path: str | Path) -> PensionDecisionRecord:
    """Load and validate a frozen record. Raises ValueError on malformed content."""
    try:
        document = json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=_unique_json_object,
        )
    except (OSError, ValueError) as exc:
        raise ValueError(f"pension decision record is unreadable: {path}") from exc
    if not isinstance(document, dict):
        raise ValueError(f"pension decision record must be an object: {path}")
    extra = sorted(set(document) - set(_RECORD_KEYS))
    if extra:
        raise ValueError(f"pension decision record has unknown fields: {extra}")
    missing = sorted(set(_RECORD_KEYS) - set(document))
    if missing:
        raise ValueError(f"pension decision record missing fields: {missing}")
    record_id = _nonblank_string(document["record_id"], name="record_id")
    try:
        frozen_at = datetime.fromisoformat(str(document["frozen_at"]))
    except ValueError as exc:
        raise ValueError("pension record frozen_at is not an ISO datetime") from exc
    if frozen_at.tzinfo is None or frozen_at.utcoffset() is None:
        raise ValueError("pension record frozen_at must be timezone-aware")
    try:
        cutoff = date.fromisoformat(str(document["seen_history_cutoff"]))
    except ValueError as exc:
        raise ValueError("pension record seen_history_cutoff is not an ISO date") from exc
    _require_month_end(cutoff, "seen_history_cutoff")
    status = _nonblank_string(document["status"], name="status")
    if status not in _FREEZABLE_STATUSES:
        raise ValueError(f"pension record status {status!r} is not freezable")
    raw_manifests = document["manifest_hashes"]
    if not isinstance(raw_manifests, dict):
        raise ValueError("pension record manifest_hashes must be an object")
    manifest_hashes = {
        _nonblank_string(key, name="manifest_hashes key"): _nonblank_string(
            value, name=f"manifest_hashes[{key!r}]"
        )
        for key, value in raw_manifests.items()
    }
    raw_equivalent = document["equivalent_ids"]
    if not isinstance(raw_equivalent, list):
        raise ValueError("pension record equivalent_ids must be an array")
    equivalent_ids = tuple(_nonblank_string(entry, name="equivalent_ids entry") for entry in raw_equivalent)
    review_every = document["review_every_months"]
    if isinstance(review_every, bool) or not isinstance(review_every, int) or review_every < 1:
        raise ValueError("pension record review_every_months must be a positive integer")
    raw_previous = document["previous_record_id"]
    previous: str | None = None
    if raw_previous is not None:
        previous = _nonblank_string(raw_previous, name="previous_record_id")
    return PensionDecisionRecord(
        record_id=record_id,
        frozen_at=frozen_at,
        git_commit=_nonblank_string(document["git_commit"], name="git_commit"),
        config_sha256=_nonblank_string(document["config_sha256"], name="config_sha256"),
        manifest_hashes=manifest_hashes,
        seen_history_cutoff=cutoff,
        status=status,
        incumbent_id=_nonblank_string(document["incumbent_id"], name="incumbent_id"),
        incumbent_schedule=_schedule_from_json(document["incumbent_schedule"], name="incumbent_schedule"),
        equivalent_ids=equivalent_ids,
        benchmark_id=_nonblank_string(document["benchmark_id"], name="benchmark_id"),
        benchmark_schedule=_schedule_from_json(document["benchmark_schedule"], name="benchmark_schedule"),
        review_every_months=review_every,
        previous_record_id=previous,
    )


def _month_close_map(prices: pl.DataFrame, sleeve: str) -> dict[tuple[int, int], float]:
    sub = prices.filter(pl.col("ticker") == sleeve).sort("date")
    closes: dict[tuple[int, int], tuple[date, float]] = {}
    for day, close in zip(
        sub.get_column("date").to_list(), sub.get_column("adjusted_close").to_list(), strict=True
    ):
        if close is None or isinstance(close, bool) or not isinstance(close, float | int):
            raise ValueError(f"pension adjusted_close for {sleeve!r} on {day!r} must be numeric")
        value = float(close)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"pension adjusted_close for {sleeve!r} on {day!r} must be finite and positive")
        key = (day.year, day.month)
        if key not in closes or day > closes[key][0]:
            closes[key] = (day, value)
    return {key: value for key, (_, value) in closes.items()}


def evaluate_pension_review(
    record: PensionDecisionRecord, prices: pl.DataFrame, as_of: datetime
) -> PensionReviewStatus:
    """Report how the incumbent tracks the benchmark after the cutoff and whether re-evaluation is due.

    Both schedules receive one unit at the first month after the cutoff and each
    later contribution year; the paired wealth ratio is informational and never
    triggers a switch by itself.

    Raises:
        ValueError: If `as_of` is naive or precedes the cutoff.
    """
    if not isinstance(as_of, datetime) or as_of.tzinfo is None or as_of.utcoffset() is None:
        raise ValueError(f"pension as_of must be timezone-aware, got {as_of!r}")
    cutoff = record.seen_history_cutoff
    if as_of.date() < cutoff:
        raise ValueError(f"pension as_of {as_of.date().isoformat()} precedes cutoff {cutoff.isoformat()}")
    for column in ("ticker", "date", "adjusted_close", "available_at"):
        if column not in prices.columns:
            raise ValueError(f"pension prices frame misses required column {column!r}")
    visible = prices.filter(pl.col("available_at") <= pl.lit(as_of.astimezone(UTC)))
    sleeves = sorted(set(record.incumbent_schedule.start_weights) | set(record.benchmark_schedule.start_weights))
    close_maps = {sleeve: _month_close_map(visible, sleeve) for sleeve in sleeves}
    observed: list[date] = []
    observed_closes: dict[str, list[float]] = {sleeve: [] for sleeve in sleeves}
    for month_end in _month_ends_after(cutoff, as_of.date()):
        if any((month_end.year, month_end.month) not in close_maps[sleeve] for sleeve in sleeves):
            break
        observed.append(month_end)
        for sleeve in sleeves:
            observed_closes[sleeve].append(close_maps[sleeve][(month_end.year, month_end.month)])
    months_observed = len(observed)
    if months_observed == 0:
        return PensionReviewStatus(
            record_id=record.record_id,
            as_of=as_of.date(),
            months_observed=0,
            incumbent_over_benchmark=None,
            state="INSUFFICIENT_DATA",
        )
    if record.incumbent_id == record.benchmark_id:
        ratio = 1.0
    else:
        ratio = _paired_wealth_ratio(record, observed_closes, months_observed)
    state: PensionReviewState = "REVIEW_DUE" if months_observed >= record.review_every_months else "HOLD"
    return PensionReviewStatus(
        record_id=record.record_id,
        as_of=as_of.date(),
        months_observed=months_observed,
        incumbent_over_benchmark=ratio,
        state=state,
    )


def _paired_wealth_ratio(
    record: PensionDecisionRecord, closes: dict[str, list[float]], months: int
) -> float:
    """Grow one unit per contribution year under each schedule and return the wealth ratio."""
    contributions = (months + 11) // 12
    total_years = max(
        contributions,
        record.incumbent_schedule.glide_years + 1,
        record.benchmark_schedule.glide_years + 1,
    )
    incumbent_weights = [
        record.incumbent_schedule.weights_for_year(year, total_years) for year in range(contributions)
    ]
    benchmark_weights = [
        record.benchmark_schedule.weights_for_year(year, total_years) for year in range(contributions)
    ]
    incumbent_holdings = dict.fromkeys(record.incumbent_schedule.start_weights, 0.0)
    benchmark_holdings = dict.fromkeys(record.benchmark_schedule.start_weights, 0.0)
    for year in range(contributions):
        incumbent_total = math.fsum(incumbent_holdings.values()) + 1.0
        weights = incumbent_weights[year]
        incumbent_holdings = {sleeve: incumbent_total * weights[sleeve] for sleeve in incumbent_holdings}
        benchmark_total = math.fsum(benchmark_holdings.values()) + 1.0
        bench_weights = benchmark_weights[year]
        benchmark_holdings = {sleeve: benchmark_total * bench_weights[sleeve] for sleeve in benchmark_holdings}
        for month in range(12):
            index = year * 12 + month
            if index + 1 >= months:
                break
            for sleeve in incumbent_holdings:
                growth = closes[sleeve][index + 1] / closes[sleeve][index]
                incumbent_holdings[sleeve] *= growth
            for sleeve in benchmark_holdings:
                growth = closes[sleeve][index + 1] / closes[sleeve][index]
                benchmark_holdings[sleeve] *= growth
    incumbent_wealth = math.fsum(incumbent_holdings.values())
    benchmark_wealth = math.fsum(benchmark_holdings.values())
    if not math.isfinite(incumbent_wealth) or not math.isfinite(benchmark_wealth) or benchmark_wealth <= 0.0:
        raise ValueError("pension review wealth is not finite and positive")  # pragma: no cover
    return incumbent_wealth / benchmark_wealth
