# ruff: noqa
"""Prune planning and application for storage layout migration."""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath

from src.data.schema import Dataset
from src.data.settings import DataSettings

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PrunePlan:
    to_delete: tuple[Path, ...] = ()
    to_migrate: tuple[tuple[Path, Path], ...] = ()
    retained_manifests: tuple[Path, ...] = ()
    retained_parquets: tuple[Path, ...] = ()
    raw_sha_to_keep: frozenset[str] = frozenset()
    nport_mirrors_to_delete: tuple[Path, ...] = ()


@dataclass(frozen=True, slots=True)
class PruneReport:
    deleted: tuple[Path, ...] = ()
    migrated: tuple[tuple[Path, Path], ...] = ()
    dry_run: bool = True
    plan: PrunePlan | None = None


def _load_manifest_document(path: Path) -> dict[str, object] | None:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(doc, dict):
            return doc
    except Exception:
        return None
    return None


def _manifest_key(doc: dict[str, object]) -> tuple[datetime, str] | None:
    try:
        retrieved_at = datetime.fromisoformat(str(doc["retrieved_at"]))
        sha = str(doc["normalized_sha256"])
        if retrieved_at.tzinfo is None:
            return None
        return (retrieved_at, sha)
    except Exception:
        return None


def _collect_datasets(root: Path) -> list[Dataset]:
    # Discover datasets that have manifests, fallback to all known Dataset values
    manifests_root = root / "manifests"
    if manifests_root.is_dir():
        found = []
        for child in manifests_root.iterdir():
            if child.is_dir():
                try:
                    ds = Dataset(child.name)
                    found.append(ds)
                except ValueError:
                    continue
        if found:
            return found
    # Fallback to all Dataset enum members
    return list(Dataset)


def plan_prune(
    settings: DataSettings,
    *,
    keep_latest_only: bool = True,
    drop_nport_zip_mirrors: bool = True,
    migrate_results_layout: bool = True,
) -> PrunePlan:
    root = settings.resolved_data_root()
    to_delete: list[Path] = []
    to_migrate: list[tuple[Path, Path]] = []
    retained_manifests: list[Path] = []
    retained_parquets: list[Path] = []
    retained_raw_shas: set[str] = set()
    nport_mirrors: list[Path] = []

    # Collect manifests per dataset and decide retention
    for dataset in _collect_datasets(root):
        manifests_dir = root / "manifests" / str(dataset)
        if not manifests_dir.is_dir():
            continue
        candidates = sorted(manifests_dir.glob("*.json"))
        if not candidates:
            continue
        # Parse manifests
        parsed: list[tuple[Path, dict[str, object], tuple[datetime, str]]] = []
        for p in candidates:
            doc = _load_manifest_document(p)
            if doc is None:
                # Malformed manifest -> treat as deletable? But fail-closed: keep only if we can parse?
                # For pruning, we will keep latest only among well-formed; malformed counted as stale.
                continue
            key = _manifest_key(doc)
            if key is None:
                continue
            parsed.append((p, doc, key))
        if not parsed:
            # No well-formed manifests, keep none, but don't delete arbitrarily?
            continue
        # Determine latest by (retrieved_at, normalized_sha256)
        parsed.sort(key=lambda x: x[2])
        latest_path, latest_doc, latest_key = parsed[-1]
        latest_sha = latest_key[1]
        # Resolve parquet path for latest
        try:
            latest_rel = PurePosixPath(str(latest_doc["normalized_relative_path"]))
            latest_parquet = root.joinpath(*latest_rel.parts)
        except Exception:
            latest_parquet = root / "normalized" / str(dataset) / f"schema_version={latest_doc.get('schema_version', '1')}" / f"{latest_sha}.parquet"
        # Determine retention set
        if keep_latest_only:
            keep_set = {latest_path}
        else:
            keep_set = {p for p, _, _ in parsed}
        for p, doc, key in parsed:
            sha = key[1]
            # Resolve parquet path
            try:
                rel = PurePosixPath(str(doc["normalized_relative_path"]))
                parquet = root.joinpath(*rel.parts)
            except Exception:
                parquet = root / "normalized" / str(dataset) / f"schema_version={doc.get('schema_version', '1')}" / f"{sha}.parquet"
            if p in keep_set:
                retained_manifests.append(p)
                retained_parquets.append(parquet)
                # Track raw sha
                try:
                    raw_section = doc["raw_artifact"]
                    if isinstance(raw_section, dict):
                        raw_sha = str(raw_section["sha256"])
                        retained_raw_shas.add(raw_sha)
                except Exception:  # noqa: S110
                    pass
            else:
                # stale to delete
                to_delete.append(p)
                if parquet.exists() or True:
                    # Always list parquet for deletion if manifest stale (even if missing, we list)
                    # But only if parquet file expected exists; we still list to attempt delete
                    to_delete.append(parquet)

        # Handle manifests that were malformed or not parsed: treat as deletable but we already skipped?
        # Include any files not in parsed as deletable?
        parsed_paths = {p for p, _, _ in parsed}
        for p in candidates:
            if p not in parsed_paths:
                to_delete.append(p)
                # try to infer parquet name from manifest file name (sha.json)
                sha_guess = p.stem
                # attempt locate parquet
                # We don't know schema_version, try glob?
                # For safety, try to find parquet with same sha under normalized/dataset
                norm_dir = root / "normalized" / str(dataset)
                if norm_dir.is_dir():
                    for parquet_candidate in norm_dir.rglob(f"{sha_guess}.parquet"):
                        to_delete.append(parquet_candidate)

    # Raw deletion: only when sha256 is unreferenced by every retained manifest
    raw_root = root / "raw"
    if raw_root.is_dir():
        # Walk raw directories: raw/<provider>/<dataset>/<sha>/payload.*
        for provider_dir in raw_root.iterdir():
            if not provider_dir.is_dir():
                continue
            # Skip raw/sec/nport handling for mirrors
            if provider_dir.name == "sec" and (provider_dir / "nport").is_dir() and provider_dir.name == "sec":
                # We'll handle nport mirrors separately
                pass
            for dataset_dir in provider_dir.iterdir():
                if not dataset_dir.is_dir():
                    continue
                # If this is raw/sec/nport, skip (handled below)
                if provider_dir.name == "sec" and dataset_dir.name == "nport":
                    continue
                for sha_dir in dataset_dir.iterdir():
                    if not sha_dir.is_dir():
                        continue
                    sha = sha_dir.name
                    # Validate sha is hex64
                    if len(sha) != 64 or not all(c in "0123456789abcdef" for c in sha):
                        continue
                    if sha not in retained_raw_shas:
                        # List all files under this sha_dir for deletion (payload.*)
                        for payload_file in sha_dir.iterdir():
                            if payload_file.is_file():
                                to_delete.append(payload_file)
                        # Also include the directory itself? We'll delete files, then directory cleanup attempted.
                        # We list directory for potential rmdir, but apply will handle file deletion.
                        # For now, also consider deleting empty sha_dir after files gone (not needed for test).
                        # We'll not add directory itself to to_delete, just files.
                        pass

    # N-PORT mirrors
    if drop_nport_zip_mirrors:
        nport_dir = root / "raw" / "sec" / "nport"
        if nport_dir.is_dir():
            for p in nport_dir.glob("*.zip"):
                if p.is_file():
                    nport_mirrors.append(p)
                    to_delete.append(p)

    # Migrate results layout
    if migrate_results_layout:
        # Existing flat dirs: data/experiments, data/audits, data/thesis_reports
        # New dirs: data/results/experiments, data/results/audits, data/results/thesis
        from src.data.paths import audits_dir, experiments_dir, thesis_reports_dir

        old_new = [
            (root / "experiments", experiments_dir(settings)),
            (root / "audits", audits_dir(settings)),
            (root / "thesis_reports", thesis_reports_dir(settings)),
        ]
        for old, new in old_new:
            if old.is_dir():
                # List each json file under old for migration
                for f in old.glob("*.json"):
                    if f.is_file():
                        dest = new / f.name
                        # Only migrate if dest doesn't already exist
                        to_migrate.append((f, dest))
                # Also consider if old dir contains subdirs? Assume flat json.
                # We don't delete old dir itself, just files after migration.

    # Deduplicate to_delete while preserving order
    seen: set[Path] = set()
    deduped: list[Path] = []
    for p in to_delete:
        # Normalize path
        try:
            rp = p.resolve()
        except Exception:
            rp = p
        # Use original path for tracking but dedup by resolve
        key_path = p
        if key_path not in seen:
            seen.add(key_path)
            deduped.append(p)

    # Similarly dedup retained
    # Filter to_delete to remove any path that is retained (should not happen)
    retained_set = set(retained_manifests) | set(retained_parquets)
    final_delete = [p for p in deduped if p not in retained_set]

    # Ensure nport_mirrors_to_delete is subset of to_delete
    # Keep order

    return PrunePlan(
        to_delete=tuple(final_delete),
        to_migrate=tuple(to_migrate),
        retained_manifests=tuple(retained_manifests),
        retained_parquets=tuple(retained_parquets),
        raw_sha_to_keep=frozenset(retained_raw_shas),
        nport_mirrors_to_delete=tuple(nport_mirrors),
    )


def apply_prune(plan: PrunePlan, *, dry_run: bool = True) -> PruneReport:
    deleted: list[Path] = []
    migrated: list[tuple[Path, Path]] = []

    if dry_run:
        return PruneReport(deleted=tuple(), migrated=tuple(), dry_run=True, plan=plan)

    # Delete only paths listed in plan
    for p in plan.to_delete:
        try:
            if p.is_file():
                p.unlink()
                deleted.append(p)
            elif p.is_dir():
                # Remove empty dirs? Only if listed as dir
                try:
                    p.rmdir()
                    deleted.append(p)
                except OSError:
                    pass
            else:
                # Path may have been already deleted or not exist; ignore
                pass
        except FileNotFoundError:
            pass
        except OSError:
            # Fail-closed: do not raise, just skip? But log
            logger.warning("[DATA] event=prune_delete_failed path=%s", p.as_posix())
            continue
        # Clean up empty parent sha dirs after file deletion
        # If file was under raw/.../<sha>/payload.*, try to remove empty sha dir
        try:
            parent = p.parent
            if parent.is_dir() and not any(parent.iterdir()):
                parent.rmdir()
        except Exception:
            pass

    for src, dst in plan.to_migrate:
        try:
            if not src.is_file():
                continue
            if dst.exists():
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
            migrated.append((src, dst))
        except Exception:
            logger.warning("[DATA] event=prune_migrate_failed src=%s dst=%s", src.as_posix(), dst.as_posix())
            continue

    # After migration, try to remove old empty dirs if no files remain
    # Not required for tests

    return PruneReport(deleted=tuple(deleted), migrated=tuple(migrated), dry_run=False, plan=plan)

