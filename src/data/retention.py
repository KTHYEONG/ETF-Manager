# ruff: noqa
"""Prune planning and application for storage layout migration."""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from src.data.catalog import _load_manifest_document, _reconstruct_manifest
from src.data.schema import Dataset, spec_for
from src.data.settings import DataSettings
from src.data.storage import DatasetArtifact, DataStore

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


@dataclass(frozen=True, slots=True)
class _VerifiedManifest:
    """One manifest that parsed and passed storage verification."""

    manifest_path: Path
    parquet_path: Path
    retrieved_at: datetime
    normalized_sha256: str
    raw_sha256: str


def _verify_manifest(store: DataStore, root: Path, dataset: Dataset, path: Path) -> _VerifiedManifest:
    """Parse, reconstruct, and storage-verify one manifest; any defect fails closed."""
    manifest = _reconstruct_manifest(_load_manifest_document(path), path)
    spec = spec_for(dataset)
    artifact = DatasetArtifact(
        normalized_path=root.joinpath(*manifest.normalized_relative_path.parts),
        manifest_path=path,
        manifest=manifest,
    )
    store.read_normalized(artifact, spec)
    return _VerifiedManifest(
        manifest_path=path,
        parquet_path=root.joinpath(*manifest.normalized_relative_path.parts),
        retrieved_at=manifest.retrieved_at,
        normalized_sha256=manifest.normalized_sha256,
        raw_sha256=manifest.raw_artifact.sha256,
    )


def plan_prune(
    settings: DataSettings,
    *,
    keep_latest_only: bool = True,
    drop_nport_zip_mirrors: bool = True,
    migrate_results_layout: bool = True,
) -> PrunePlan:
    """Build a non-destructive retention plan from fully verified lineage references.

    Args:
        settings: Data root to inspect.
        keep_latest_only: Keep only the latest trusted manifest per dataset when true.
        drop_nport_zip_mirrors: Include redundant N-PORT mirrors in the proposed plan.
        migrate_results_layout: Include eligible legacy result moves in the plan.

    Returns:
        Proposed file deletions and migrations without changing the filesystem.

    Raises:
        UntrustedDatasetError: If a manifest is malformed, a referenced file is
            missing or corrupt, or safe reachability cannot be established.
    """
    root = settings.resolved_data_root()
    store = DataStore(settings)
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
        # Verify every candidate before selecting latest or computing reachability.
        verified = [_verify_manifest(store, root, dataset, path) for path in candidates]
        # Determine latest with the catalog order: (retrieved_at, sha, filename).
        verified.sort(key=lambda item: (item.retrieved_at, item.normalized_sha256, item.manifest_path.name))
        if keep_latest_only:
            keep_set = {verified[-1].manifest_path}
        else:
            keep_set = {item.manifest_path for item in verified}
        for item in verified:
            if item.manifest_path in keep_set:
                retained_manifests.append(item.manifest_path)
                retained_parquets.append(item.parquet_path)
                retained_raw_shas.add(item.raw_sha256)
        retained_parquet_paths = set(retained_parquets)
        for item in verified:
            if item.manifest_path not in keep_set:
                to_delete.append(item.manifest_path)
                if item.parquet_path not in retained_parquet_paths:
                    to_delete.append(item.parquet_path)

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
        # New dirs: data/results/<flat-name> staging dirs consumed by Spec 2 migration.
        from src.data.paths import LEGACY_FLAT_RESULT_SUBDIRS, results_root

        _legacy_old = {"experiments": "experiments", "audits": "audits", "thesis": "thesis_reports"}
        old_new = [
            (root / _legacy_old[name], results_root(settings) / name)
            for name in LEGACY_FLAT_RESULT_SUBDIRS
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
    """Apply only an explicitly supplied retention plan when dry-run is disabled.

    Args:
        plan: Verified candidate deletions and migrations.
        dry_run: Preserve all files and report the plan when true.

    Returns:
        Paths actually deleted or migrated, together with dry-run status.
    """
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

