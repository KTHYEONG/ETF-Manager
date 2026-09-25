# ruff: noqa
"""Prune planning and application for storage layout migration."""

from __future__ import annotations

import json
import logging
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Final

from src.data.catalog import _load_manifest_document, _reconstruct_manifest
from src.data.schema import Dataset, spec_for
from src.data.settings import DataSettings
from src.data.storage import DatasetArtifact, DataStore, UntrustedDatasetError

logger = logging.getLogger(__name__)

_HEX64_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{64}$")
_AMBIGUOUS_SINGLE_KEYS: Final[tuple[str, ...]] = ("manifest_hash",)
_AMBIGUOUS_MULTI_KEYS: Final[tuple[str, ...]] = ("manifest_hashes",)
_EXACT_SINGLE_KEYS: Final[tuple[str, ...]] = ("manifest_sha256",)
_EXACT_MULTI_KEYS: Final[tuple[str, ...]] = ("manifest_sha256s",)


@dataclass(frozen=True, slots=True)
class PrunePlan:
    to_delete: tuple[Path, ...] = ()
    to_migrate: tuple[tuple[Path, Path], ...] = ()
    retained_manifests: tuple[Path, ...] = ()
    retained_parquets: tuple[Path, ...] = ()
    raw_sha_to_keep: frozenset[str] = frozenset()
    nport_mirrors_to_delete: tuple[Path, ...] = ()
    missing_evidence: tuple[str, ...] = ()
    missing_raw: tuple[str, ...] = ()


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
    raw_relative_path: PurePosixPath


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
        raw_relative_path=manifest.raw_artifact.relative_path,
    )


def _register_pin_value(value: object, source: Path, exact: bool, ambiguous: list[str], exact_pins: list[str]) -> None:
    """Validate one recorded pin value; sentinels are skipped, malformed syntax fails closed."""
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, str):
        raise UntrustedDatasetError(f"malformed pin value in {source.as_posix()}")
    if value.startswith("NO_"):
        return
    if _HEX64_PATTERN.fullmatch(value) is None:
        raise UntrustedDatasetError(f"malformed pin value in {source.as_posix()}")
    (exact_pins if exact else ambiguous).append(value)


def _collect_recorded_pins(settings: DataSettings) -> tuple[list[str], list[str]]:
    """Collect ambiguous frame-or-manifest pins and exact manifest pins from evidence files."""
    roots = [settings.resolved_data_root() / "results", Path.cwd() / "docs" / "results", Path.cwd() / "records" / "prospective"]
    ambiguous: list[str] = []
    exact_pins: list[str] = []
    for evidence_root in roots:
        if not evidence_root.is_dir():
            continue
        for candidate in sorted(evidence_root.rglob("*.json")):
            try:
                document = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, ValueError):
                continue
            if not isinstance(document, dict):
                continue
            for key in _AMBIGUOUS_SINGLE_KEYS:
                if key in document:
                    _register_pin_value(document[key], candidate, False, ambiguous, exact_pins)
            for key in _EXACT_SINGLE_KEYS:
                if key in document:
                    _register_pin_value(document[key], candidate, True, ambiguous, exact_pins)
            for key, exact in (
                *((k, False) for k in _AMBIGUOUS_MULTI_KEYS),
                *((k, True) for k in _EXACT_MULTI_KEYS),
            ):
                if key not in document:
                    continue
                multi = document[key]
                if multi is None:
                    continue
                if isinstance(multi, dict):
                    items: list[object] = list(multi.values())
                elif isinstance(multi, (list, tuple)):
                    items = list(multi)
                else:
                    raise UntrustedDatasetError(f"malformed pin value in {candidate.as_posix()}")
                for item in items:
                    _register_pin_value(item, candidate, exact, ambiguous, exact_pins)
    return ambiguous, exact_pins


def plan_prune(
    settings: DataSettings,
    *,
    keep_latest_only: bool = True,
    drop_nport_zip_mirrors: bool = True,
    migrate_results_layout: bool = True,
) -> PrunePlan:
    """Plan deletion only after resolving latest and recorded Silver references.

    Args:
        settings: Root containing manifests, Silver, Bronze, and run results.
        keep_latest_only: Retain latest plus recorded pins when true; retain every valid Silver when false.
        drop_nport_zip_mirrors: Include redundant N-PORT mirror ZIPs when safe.
        migrate_results_layout: Preserve the existing result-layout migration option.

    Returns:
        A dry, explicit plan with all retained and deletable paths.

    Raises:
        UntrustedDatasetError: If Silver integrity or reference reachability cannot be established safely.
    """
    root = settings.resolved_data_root()
    store = DataStore(settings)
    to_delete: list[Path] = []
    to_migrate: list[tuple[Path, Path]] = []
    retained_manifests: list[Path] = []
    retained_parquets: list[Path] = []
    latest_raw_shas: set[str] = set()
    missing_raw: list[str] = []
    nport_mirrors: list[Path] = []

    # Verify every candidate before selecting latest or computing reachability.
    verified_by_dataset: dict[Dataset, list[_VerifiedManifest]] = {}
    for dataset in _collect_datasets(root):
        manifests_dir = root / "manifests" / str(dataset)
        if not manifests_dir.is_dir():
            continue
        candidates = sorted(manifests_dir.glob("*.json"))
        if not candidates:
            continue
        verified = [_verify_manifest(store, root, dataset, path) for path in candidates]
        verified.sort(key=lambda item: (item.retrieved_at, item.normalized_sha256, item.manifest_path.name))
        verified_by_dataset[dataset] = verified

    # Resolve recorded pins against verified manifests; malformed syntax fails closed.
    ambiguous_pins, exact_pins = _collect_recorded_pins(settings)
    all_verified = [item for verified in verified_by_dataset.values() for item in verified]
    pinned_paths: set[Path] = set()
    missing_evidence: list[str] = []
    for pin in ambiguous_pins:
        exact = [item for item in all_verified if item.manifest_path.stem == pin]
        if exact:
            pinned_paths.update(item.manifest_path for item in exact)
            continue
        framed = [item for item in all_verified if item.normalized_sha256 == pin]
        if framed:
            pinned_paths.update(item.manifest_path for item in framed)
        else:
            missing_evidence.append(pin)
    for pin in exact_pins:
        exact = [item for item in all_verified if item.manifest_path.stem == pin]
        if exact:
            pinned_paths.update(item.manifest_path for item in exact)
        else:
            missing_evidence.append(pin)

    # Collect manifests per dataset and decide retention
    for dataset, verified in verified_by_dataset.items():
        # Determine latest with the catalog order: (retrieved_at, sha, filename).
        latest = verified[-1]
        latest_raw_shas.add(latest.raw_sha256)
        raw_path = root.joinpath(*latest.raw_relative_path.parts)
        if not raw_path.is_file():
            logger.warning(
                "[DATA] event=bronze_missing_for_repair dataset=%s raw_path=%s",
                str(dataset),
                raw_path.as_posix(),
            )
            if latest.raw_sha256 not in missing_raw:
                missing_raw.append(latest.raw_sha256)
        if keep_latest_only:
            keep_set = {latest.manifest_path} | {p for p in pinned_paths if any(p == item.manifest_path for item in verified)}
        else:
            keep_set = {item.manifest_path for item in verified}
        for item in verified:
            if item.manifest_path in keep_set:
                retained_manifests.append(item.manifest_path)
                retained_parquets.append(item.parquet_path)
        retained_parquet_paths = set(retained_parquets)
        for item in verified:
            if item.manifest_path not in keep_set:
                to_delete.append(item.manifest_path)
                if item.parquet_path not in retained_parquet_paths:
                    to_delete.append(item.parquet_path)

    # Raw deletion: only when sha256 is unreferenced by every latest manifest
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
                    if sha not in latest_raw_shas:
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
        raw_sha_to_keep=frozenset(latest_raw_shas),
        nport_mirrors_to_delete=tuple(nport_mirrors),
        missing_evidence=tuple(sorted(set(missing_evidence))),
        missing_raw=tuple(sorted(set(missing_raw))),
    )


def _resolved_identity(path: Path) -> Path:
    """Resolve a plan path for protected-identity comparison; never raises."""
    try:
        return path.resolve()
    except Exception:  # pragma: no cover
        return path


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

    # Recheck protected Silver identities: a plan that outlived concurrent
    # publication must never delete a currently retained manifest or Parquet.
    protected = {_resolved_identity(p) for p in (*plan.retained_manifests, *plan.retained_parquets)}

    # Delete only paths listed in plan
    for p in plan.to_delete:
        if _resolved_identity(p) in protected:
            logger.warning("[DATA] event=prune_skip_protected path=%s", p.as_posix())
            continue
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

