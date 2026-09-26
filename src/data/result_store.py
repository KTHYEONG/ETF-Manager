"""Per-experiment result layout writer entrypoint."""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from src.data.paths import results_root
from src.data.settings import DataSettings

logger = logging.getLogger(__name__)

_SLUG_INVALID_CHARS = re.compile(r"[^a-z0-9_.-]")


class ResultKind(StrEnum):
    """Artifact family written under one experiment directory; forms the filename prefix."""

    WALK_FORWARD = "walk_forward"
    COSTS = "costs"
    ROBUSTNESS = "robustness"
    SELECTION = "selection"
    ABLATION = "ablation"
    PROSPECTIVE_FREEZE = "prospective_freeze"
    ACCUMULATION = "accumulation"
    AFTER_TAX = "after_tax"
    PENSION = "pension"
    PENSION_DECISION = "pension_decision"
    FINAL_HISTORICAL = "final_historical"
    FEASIBILITY = "feasibility"
    THESIS_REPORT = "thesis_report"
    THESIS_WAVE = "thesis_wave"
    THESIS_INCREMENTAL = "thesis_incremental"
    WAVE_D_EXIT = "wave_d_exit"
    LEGACY = "legacy"


@dataclass(frozen=True, slots=True)
class ResultRef:
    """Deterministic location of one run artifact.

    ``json_path`` and ``markdown_path`` share the stem ``<kind>_<run_id>``; the markdown
    sidecar is optional and only exists when a writer emits a human summary.
    """

    experiment: str
    kind: ResultKind
    run_id: str
    json_path: Path
    markdown_path: Path


@dataclass(frozen=True, slots=True)
class RunLedgerEntry:
    """One line of ``runs.jsonl``: records that a run artifact was (re)written at ``written_at`` (UTC)."""

    experiment: str
    kind: ResultKind
    run_id: str
    written_at: datetime


def normalize_result_slug(text: str) -> str:
    """Normalize an experiment name or run id into a filesystem-safe slug.

    Strips surrounding whitespace, lower-cases, then replaces every character outside
    ``[a-z0-9_.-]`` with ``-`` (so ISO timestamps like ``2026-08-28T00:00:00+00:00``
    become ``2026-08-28t00-00-00-00-00``). Case-folding makes
    ``FINAL_HISTORICAL_CAMPAIGN_V1`` (campaign_id) and ``final_historical_campaign_v1``
    (config name) share one directory.

    Raises:
        ValueError: If the result is empty, starts with ``.`` or ``_`` (reserved for
            ``_legacy`` and hidden files), or contains ``..``.
    """
    slug = _SLUG_INVALID_CHARS.sub("-", text.strip().lower())
    if not slug or slug.startswith(".") or slug.startswith("_") or ".." in slug:
        raise ValueError(f"invalid result slug: {text!r}")
    return slug


def result_ref(
    settings: DataSettings, *, experiment: str, kind: ResultKind, run_id: str
) -> ResultRef:
    """Resolve (and create the directory for) the artifact paths of one run.

    Used directly by writers that render through an existing ``path``-taking function
    (thesis markdown/incremental writers); all others use :func:`write_result`.
    """
    experiment_slug = normalize_result_slug(experiment)
    run_slug = normalize_result_slug(run_id)
    directory = results_root(settings) / experiment_slug
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"{kind.value}_{run_slug}"
    json_path = directory / f"{stem}.json"
    markdown_path = directory / f"{stem}.md"
    return ResultRef(
        experiment=experiment_slug,
        kind=kind,
        run_id=run_slug,
        json_path=json_path,
        markdown_path=markdown_path,
    )


def _require_aware_utc(written_at: datetime | None) -> datetime:
    resolved = written_at if written_at is not None else datetime.now(UTC)
    if resolved.tzinfo is None:
        raise ValueError("written_at must be timezone-aware")
    return resolved


def record_run(
    settings: DataSettings, ref: ResultRef, *, written_at: datetime | None = None
) -> RunLedgerEntry:
    """Append one ledger line to ``<experiment>/runs.jsonl``.

    Args:
        written_at: Timezone-aware UTC timestamp; defaults to ``datetime.now(UTC)``.

    Raises:
        ValueError: If ``written_at`` is naive.
    """
    resolved = _require_aware_utc(written_at)
    ledger_path = ref.json_path.parent / "runs.jsonl"
    line = (
        json.dumps(
            {
                "experiment": ref.experiment,
                "kind": ref.kind.value,
                "run_id": ref.run_id,
                "written_at": resolved.isoformat(),
            }
        )
        + "\n"
    )
    with ledger_path.open("a", encoding="utf-8") as handle:
        handle.write(line)
    return RunLedgerEntry(
        experiment=ref.experiment, kind=ref.kind, run_id=ref.run_id, written_at=resolved
    )


def _atomic_write_text(path: Path, content: str) -> None:
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp_path.write_text(content, encoding="utf-8")
    os.replace(tmp_path, path)


def write_result(
    settings: DataSettings,
    *,
    experiment: str,
    kind: ResultKind,
    run_id: str,
    payload: Mapping[str, object],
    markdown: str | None = None,
    written_at: datetime | None = None,
) -> ResultRef:
    """Persist one run artifact atomically and record it in the run ledger.

    The JSON body is written exactly as ``json.dumps(payload, indent=2)`` so existing
    report schemas and downstream readers (e.g. the executed-hash census) are unchanged.

    Returns:
        The resolved :class:`ResultRef`; ``markdown_path`` exists only if ``markdown`` was given.

    Raises:
        ValueError: Invalid slug (see :func:`normalize_result_slug`) or naive ``written_at``.
        OSError: The artifact cannot be written.
    """
    if written_at is not None and written_at.tzinfo is None:
        raise ValueError("written_at must be timezone-aware")
    ref = result_ref(settings, experiment=experiment, kind=kind, run_id=run_id)
    _atomic_write_text(ref.json_path, json.dumps(payload, indent=2))
    if markdown is not None:
        _atomic_write_text(ref.markdown_path, markdown)
    record_run(settings, ref, written_at=written_at)
    logger.info(
        "[DATA] event=result_written experiment=%s kind=%s run_id=%s path=%s",
        ref.experiment,
        ref.kind.value,
        ref.run_id,
        ref.json_path.as_posix(),
    )
    return ref


def read_run_ledger(settings: DataSettings, experiment: str) -> tuple[RunLedgerEntry, ...]:
    """Return ledger entries in file (chronological append) order; empty tuple if no ledger.

    Raises:
        ValueError: A ledger line is not valid JSON or has an unknown ``kind``
            (fail-closed: a corrupted ledger must not silently hide runs).
    """
    experiment_slug = normalize_result_slug(experiment)
    ledger_path = results_root(settings) / experiment_slug / "runs.jsonl"
    if not ledger_path.exists():
        return ()
    entries: list[RunLedgerEntry] = []
    for raw_line in ledger_path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        try:
            doc = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"corrupted run ledger at {ledger_path}: {exc}") from exc
        if not isinstance(doc, dict):
            raise ValueError(f"corrupted run ledger at {ledger_path}: line must be an object")
        try:
            kind = ResultKind(str(doc["kind"]))
            written_at = datetime.fromisoformat(str(doc["written_at"]))
        except (KeyError, ValueError) as exc:
            raise ValueError(f"corrupted run ledger at {ledger_path}: {exc}") from exc
        try:
            entry_experiment = str(doc["experiment"])
            entry_run_id = str(doc["run_id"])
        except KeyError as exc:
            raise ValueError(f"corrupted run ledger at {ledger_path}: {exc}") from exc
        entries.append(
            RunLedgerEntry(
                experiment=entry_experiment,
                kind=kind,
                run_id=entry_run_id,
                written_at=written_at,
            )
        )
    return tuple(entries)
