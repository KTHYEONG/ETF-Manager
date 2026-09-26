"""Inspection, curation, pruning, and migration for per-experiment run artifacts."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from src.data.paths import LEGACY_FLAT_RESULT_SUBDIRS, legacy_results_dir, results_root
from src.data.result_store import (
    ResultKind,
    ResultRef,
    RunLedgerEntry,
    normalize_result_slug,
    read_run_ledger,
    record_run,
)
from src.data.settings import DataSettings

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ResultSummary:
    """Latest known state of one (experiment, kind, run_id) artifact."""

    experiment: str
    kind: ResultKind
    run_id: str
    last_written_at: datetime
    run_count: int
    json_path: Path
    has_markdown: bool


@dataclass(frozen=True, slots=True)
class ResultPrunePlan:
    """Run artifacts to delete, keeping ``keep`` most recent runs per (experiment, kind)."""

    keep: int
    to_delete: tuple[ResultRef, ...]


@dataclass(frozen=True, slots=True)
class ResultMigrationPlan:
    """Pre-layout flat outputs to move into the per-experiment layout."""

    moves: tuple[tuple[Path, Path], ...]
    ledger_entries: tuple[RunLedgerEntry, ...]


def _experiment_dirs(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted((d for d in root.iterdir() if d.is_dir() and d.name != "_legacy"), key=lambda d: d.name)


def list_results(settings: DataSettings, *, experiment: str | None = None) -> tuple[ResultSummary, ...]:
    """Summarize run artifacts from the run ledgers.

    One summary per distinct (experiment, kind, run_id) whose JSON still exists on disk;
    ``run_count`` counts ledger lines, ``last_written_at`` is the max ledger timestamp.
    Sorted by experiment ascending, then ``last_written_at`` descending.

    Raises:
        ValueError: Propagated from a corrupted ledger (fail-closed).
    """
    root = results_root(settings)
    slugs = [normalize_result_slug(experiment)] if experiment is not None else [d.name for d in _experiment_dirs(root)]
    summaries: list[ResultSummary] = []
    for slug in slugs:
        groups: dict[tuple[ResultKind, str], list[datetime]] = {}
        for entry in read_run_ledger(settings, slug):
            groups.setdefault((entry.kind, entry.run_id), []).append(entry.written_at)
        for (kind, run_id), stamps in groups.items():
            json_path = root / slug / f"{kind.value}_{run_id}.json"
            if not json_path.is_file():
                continue
            summaries.append(
                ResultSummary(
                    experiment=slug,
                    kind=kind,
                    run_id=run_id,
                    last_written_at=max(stamps),
                    run_count=len(stamps),
                    json_path=json_path,
                    has_markdown=json_path.with_suffix(".md").is_file(),
                )
            )
    summaries.sort(key=lambda s: (s.experiment, -s.last_written_at.timestamp()))
    return tuple(summaries)


def _copy_identical(src: Path, dest: Path) -> Path:
    if dest.exists():
        if dest.read_bytes() == src.read_bytes():
            return dest
        raise FileExistsError(f"curated destination differs, refusing overwrite: {dest.as_posix()}")
    shutil.copyfile(src, dest)
    return dest


def promote_result(
    settings: DataSettings,
    *,
    experiment: str,
    kind: ResultKind,
    run_id: str,
    dest_root: Path = Path("data/research"),
) -> tuple[Path, ...]:
    """Copy one run artifact (JSON and markdown sidecar if present) into promoted evidence.

    Destination: ``dest_root/<experiment_slug>/<kind>_<run_id_slug>.{json,md}``.
    Idempotent when the destination bytes already match.

    Returns:
        The destination paths written or already identical, JSON first.

    Raises:
        FileNotFoundError: The source JSON does not exist.
        FileExistsError: A destination exists with different bytes (promoted evidence is
            never silently overwritten).
    """
    exp_slug = normalize_result_slug(experiment)
    run_slug = normalize_result_slug(run_id)
    src_json = results_root(settings) / exp_slug / f"{kind.value}_{run_slug}.json"
    if not src_json.is_file():
        raise FileNotFoundError(f"no run artifact at {src_json.as_posix()}")
    dest_dir = dest_root / exp_slug
    dest_dir.mkdir(parents=True, exist_ok=True)
    written = [_copy_identical(src_json, dest_dir / src_json.name)]
    src_md = src_json.with_suffix(".md")
    if src_md.is_file():
        written.append(_copy_identical(src_md, dest_dir / src_md.name))
    return tuple(written)


def plan_result_prune(settings: DataSettings, *, keep: int) -> ResultPrunePlan:
    """Plan deletion of all but the ``keep`` most recently written runs per (experiment, kind).

    Recency is the latest ledger ``written_at`` per run_id. ``_legacy/`` and
    ``ResultKind.LEGACY`` groups are excluded.

    Raises:
        ValueError: ``keep < 1``.
    """
    if keep < 1:
        raise ValueError(f"keep must be >= 1, got {keep}")
    root = results_root(settings)
    to_delete: list[ResultRef] = []
    for exp_dir in _experiment_dirs(root):
        latest: dict[tuple[ResultKind, str], datetime] = {}
        for entry in read_run_ledger(settings, exp_dir.name):
            if entry.kind is ResultKind.LEGACY:
                continue
            key = (entry.kind, entry.run_id)
            if key not in latest or entry.written_at > latest[key]:
                latest[key] = entry.written_at
        by_kind: dict[ResultKind, list[tuple[str, datetime]]] = {}
        for (kind, run_id), written_at in latest.items():
            by_kind.setdefault(kind, []).append((run_id, written_at))
        for kind, runs in by_kind.items():
            runs.sort(key=lambda item: item[1], reverse=True)
            for run_id, _ in runs[keep:]:
                stem = f"{kind.value}_{run_id}"
                to_delete.append(
                    ResultRef(
                        experiment=exp_dir.name,
                        kind=kind,
                        run_id=run_id,
                        json_path=exp_dir / f"{stem}.json",
                        markdown_path=exp_dir / f"{stem}.md",
                    )
                )
    to_delete.sort(key=lambda ref: ref.json_path.as_posix())
    return ResultPrunePlan(keep=keep, to_delete=tuple(to_delete))


def _pruned_run_keys(plan: ResultPrunePlan, experiment: str) -> set[tuple[str, str]]:
    return {(ref.kind.value, ref.run_id) for ref in plan.to_delete if ref.experiment == experiment}


def _drop_pruned_ledger_lines(ledger_path: Path, pruned: set[tuple[str, str]]) -> bool:
    lines = ledger_path.read_text(encoding="utf-8").splitlines()
    kept: list[str] = []
    for raw in lines:
        if not raw.strip():
            kept.append(raw)
            continue
        try:
            doc = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"corrupted run ledger at {ledger_path}: {exc}") from exc
        if not isinstance(doc, dict):
            raise ValueError(f"corrupted run ledger at {ledger_path}: line must be an object")
        try:
            key = (str(doc["kind"]), str(doc["run_id"]))
        except KeyError as exc:
            raise ValueError(f"corrupted run ledger at {ledger_path}: {exc}") from exc
        if key not in pruned:
            kept.append(raw)
    if len(kept) == len(lines):
        return False
    tmp_path = ledger_path.with_name(f"{ledger_path.name}.{os.getpid()}.tmp")
    tmp_path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    os.replace(tmp_path, ledger_path)
    return True


def apply_result_prune(settings: DataSettings, plan: ResultPrunePlan, *, dry_run: bool) -> tuple[Path, ...]:
    """Delete planned artifacts and drop their ledger lines; return deleted (or would-delete) paths.

    Ledger rewrite is atomic (temp file + ``os.replace``) and preserves the relative order
    of surviving lines. ``dry_run=True`` touches nothing.
    """
    deleted: list[Path] = []
    for ref in plan.to_delete:
        deleted.append(ref.json_path)
        if ref.markdown_path.is_file():
            deleted.append(ref.markdown_path)
    if not dry_run:
        for ref in plan.to_delete:
            with contextlib.suppress(FileNotFoundError):
                ref.json_path.unlink()
            with contextlib.suppress(FileNotFoundError):
                ref.markdown_path.unlink()
        experiments = sorted({ref.experiment for ref in plan.to_delete})
        for experiment in experiments:
            ledger_path = results_root(settings) / experiment / "runs.jsonl"
            if ledger_path.is_file():
                _drop_pruned_ledger_lines(ledger_path, _pruned_run_keys(plan, experiment))
    logger.info("[DATA] event=results_prune dry_run=%s count=%d", dry_run, len(deleted))
    return tuple(deleted)


def _infer_experiment(doc: object) -> str | None:
    if not isinstance(doc, dict):
        return None
    for key in ("name", "campaign_id", "thesis_id"):
        value = doc.get(key)
        if isinstance(value, str) and value.strip():
            return f"thesis_{value}" if key == "thesis_id" else value
    return None


def _plan_single_migration(
    settings: DataSettings,
    *,
    flat: str,
    src: Path,
    seen: set[Path],
    moves: list[tuple[Path, Path]],
    entries: list[RunLedgerEntry],
) -> None:
    root = results_root(settings)
    try:
        doc = json.loads(src.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        doc = None
    raw_experiment = _infer_experiment(doc)
    dest_json: Path | None = None
    dest_md: Path | None = None
    entry: RunLedgerEntry | None = None
    if raw_experiment is not None:
        try:
            exp_slug = normalize_result_slug(raw_experiment)
            stem = src.stem
            prefix = f"{raw_experiment}_"
            if stem.lower().startswith(prefix.lower()):
                stem = stem[len(prefix):]
            run_slug = normalize_result_slug(stem)
        except ValueError:
            pass
        else:
            dest_json = root / exp_slug / f"legacy_{run_slug}.json"
            sidecar = src.with_suffix(".md")
            if sidecar.is_file():
                dest_md = dest_json.with_suffix(".md")
            entry = RunLedgerEntry(
                experiment=exp_slug,
                kind=ResultKind.LEGACY,
                run_id=run_slug,
                written_at=datetime.fromtimestamp(src.stat().st_mtime, tz=UTC),
            )
    if dest_json is None:
        dest_json = legacy_results_dir(settings) / flat / src.name
        sidecar = src.with_suffix(".md")
        if sidecar.is_file():
            dest_md = dest_json.with_suffix(".md")
        entry = None
    for dest in (dest_json, dest_md) if dest_md is not None else (dest_json,):
        if dest in seen or dest.exists():
            raise FileExistsError(f"migration destination collision, plan rejected: {dest.as_posix()}")
        seen.add(dest)
    moves.append((src, dest_json))
    if dest_md is not None:
        moves.append((src.with_suffix(".md"), dest_md))
    if entry is not None:
        entries.append(entry)


def plan_result_migration(settings: DataSettings) -> ResultMigrationPlan:
    """Plan moving pre-layout flat outputs into ``<experiment>/legacy_<stem>.json``.

    Scans ``results_root / d`` for ``d`` in ``LEGACY_FLAT_RESULT_SUBDIRS`` (``*.json`` only;
    a ``.md`` with the same stem follows its JSON). Experiment = first non-empty string
    among payload keys ``name``, ``campaign_id``, ``thesis_id`` (``thesis_id`` gets the
    ``thesis_`` prefix). ``stem`` is the original stem with a leading
    ``<experiment>_`` prefix removed case-insensitively. Files that are not JSON objects,
    lack all three keys, or yield an invalid slug go to ``legacy_results_dir / d / <name>``.
    One ``LEGACY`` ledger entry per migrated JSON with ``written_at`` = source mtime (UTC).

    Raises:
        FileExistsError: Two sources map to the same destination, or a destination
            already exists (plan is rejected as a whole; nothing is half-applied).
    """
    root = results_root(settings)
    moves: list[tuple[Path, Path]] = []
    entries: list[RunLedgerEntry] = []
    seen: set[Path] = set()
    if root.is_dir():
        for flat in LEGACY_FLAT_RESULT_SUBDIRS:
            flat_dir = root / flat
            if not flat_dir.is_dir():
                continue
            for src in sorted(flat_dir.glob("*.json"), key=lambda p: p.as_posix()):
                if not src.is_file():
                    continue
                _plan_single_migration(settings, flat=flat, src=src, seen=seen, moves=moves, entries=entries)
    return ResultMigrationPlan(moves=tuple(moves), ledger_entries=tuple(entries))


def apply_result_migration(settings: DataSettings, plan: ResultMigrationPlan, *, dry_run: bool) -> int:
    """Execute moves, append ledger entries, then remove emptied flat directories.

    Returns the number of files moved (or that would move). ``dry_run=True`` touches nothing.
    """
    if not dry_run:
        for src, dest in plan.moves:
            dest.parent.mkdir(parents=True, exist_ok=True)
            src.replace(dest)
        for entry in plan.ledger_entries:
            dest = results_root(settings) / entry.experiment / f"{entry.kind.value}_{entry.run_id}.json"
            record_run(
                settings,
                ResultRef(
                    experiment=entry.experiment,
                    kind=entry.kind,
                    run_id=entry.run_id,
                    json_path=dest,
                    markdown_path=dest.with_suffix(".md"),
                ),
                written_at=entry.written_at,
            )
        root = results_root(settings)
        for flat in LEGACY_FLAT_RESULT_SUBDIRS:
            with contextlib.suppress(OSError):
                (root / flat).rmdir()
    logger.info("[DATA] event=results_migrate dry_run=%s count=%d", dry_run, len(plan.moves))
    return len(plan.moves)
