"""Mirror the data root to Google Drive with no delete propagation and remove local Bronze payloads only after remote verification."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

from src.data.settings import DataSettings

logger = logging.getLogger(__name__)

DEFAULT_REMOTE = "gdrive:quant-lake/projects/ETF-Manager"
DEFAULT_RCLONE = "rclone"
REMOTE_ENV_VAR = "ETF_MANAGER_BACKUP_REMOTE"
RCLONE_ENV_VAR = "RCLONE"

_EXCLUDE_PATTERNS = (".env*", "*.key", "*_key.txt")


class BackupZone(StrEnum):
    """One data-root-relative directory family with its own overwrite policy."""

    NORMALIZED = "normalized"
    MANIFESTS = "manifests"
    FROZEN = "frozen"
    PROSPECTIVE_REGISTRY = "prospective_registry"
    RAW = "raw"


_PUSH_ORDER = (
    BackupZone.NORMALIZED,
    BackupZone.MANIFESTS,
    BackupZone.FROZEN,
    BackupZone.PROSPECTIVE_REGISTRY,
    BackupZone.RAW,
)
_PULL_ZONES = (
    BackupZone.NORMALIZED,
    BackupZone.MANIFESTS,
    BackupZone.FROZEN,
    BackupZone.PROSPECTIVE_REGISTRY,
)
_IMMUTABLE_ZONES = frozenset(
    {BackupZone.NORMALIZED, BackupZone.MANIFESTS, BackupZone.FROZEN, BackupZone.RAW}
)


@dataclass(frozen=True, slots=True)
class BackupConfig:
    """Resolved endpoints for one backup invocation.

    Attributes:
        data_root: Absolute local data root; every zone is a direct child of it.
        remote: Drive folder holding the mirrored zones (``<remote>/<zone>``).
        rclone: rclone executable name or path.

    Raises:
        ValueError: If ``remote`` is blank or does not contain a ``:`` remote separator, or ``data_root``
            is not an absolute directory path.
    """

    data_root: Path
    remote: str
    rclone: str

    def __post_init__(self) -> None:
        if not self.remote.strip() or ":" not in self.remote:
            raise ValueError(f"backup remote {self.remote!r} must look like <remote>:<path>")
        if not self.data_root.is_absolute():
            raise ValueError(f"data_root {str(self.data_root)!r} must be an absolute directory path")


def resolve_backup_config(environ: Mapping[str, str], settings: DataSettings) -> BackupConfig:
    """Build the configuration from environment overrides and the repository data root.

    ``ETF_MANAGER_BACKUP_REMOTE`` overrides the default remote ``gdrive:quant-lake/projects/ETF-Manager``;
    ``RCLONE`` overrides the executable (default ``rclone``). The data root always comes from settings,
    never from a caller-supplied path, so a mistyped argument cannot point the tool at another directory.
    """
    return BackupConfig(
        data_root=settings.resolved_data_root(),
        remote=environ.get(REMOTE_ENV_VAR, DEFAULT_REMOTE),
        rclone=environ.get(RCLONE_ENV_VAR, DEFAULT_RCLONE) or DEFAULT_RCLONE,
    )


def _log_zone(zone: BackupZone, action: str, status: str) -> None:
    logger.info("[SYS] event=backup_zone zone=%s action=%s status=%s", zone.value, action, status)


def _log_done(command: str, status: str) -> None:
    logger.info("[SYS] event=backup_done command=%s status=%s", command, status)


def _transfer_args(config: BackupConfig, zone: BackupZone, verb: str, local_first: bool) -> list[str]:
    """Copy/move command for one zone; content-addressed zones are never silently overwritten."""
    local = str(config.data_root / zone.value)
    remote_path = f"{config.remote}/{zone.value}"
    source, dest = (local, remote_path) if local_first else (remote_path, local)
    args = [config.rclone, verb, source, dest, "--checksum"]
    if zone in _IMMUTABLE_ZONES:
        args.append("--immutable")
    for pattern in _EXCLUDE_PATTERNS:
        args.extend(["--exclude", pattern])
    args.extend(["--retries", "3", "--low-level-retries", "10"])
    return args


def _verify_args(config: BackupConfig, zone: BackupZone) -> list[str]:
    """One-way read-only content comparison for one zone; never writes to the remote."""
    args = [
        config.rclone,
        "check",
        str(config.data_root / zone.value),
        f"{config.remote}/{zone.value}",
        "--one-way",
    ]
    for pattern in _EXCLUDE_PATTERNS:
        args.extend(["--exclude", pattern])
    args.extend(["--retries", "3", "--low-level-retries", "10"])
    return args


def _run(args: Sequence[str]) -> int:
    """Run one rclone command, translating a missing executable into a failure code."""
    try:
        completed = subprocess.run(args, check=False)
    except OSError:
        return 1
    return completed.returncode


def _present_local_dir(config: BackupConfig, zone: BackupZone) -> bool:
    """True only when the zone is a real directory inside the data root."""
    path = config.data_root / zone.value
    return path.is_dir() and not path.is_symlink()


def _safe_raw_dir(config: BackupConfig) -> Path | None:
    """Resolve the Bronze source only when it is a real directory inside the data root."""
    raw = config.data_root / BackupZone.RAW.value
    if raw.is_symlink() or not raw.is_dir():
        return None
    if config.data_root.resolve() not in raw.resolve().parents:
        return None
    return raw


def push(config: BackupConfig) -> int:
    """Upload every existing local zone, then move verified Bronze payloads off the local disk.

    Zone order is fixed: normalized, manifests, frozen, prospective_registry, raw. Only data that cannot be regenerated is backed up; run outputs and promoted evidence are rebuilt by rerunning the decisions. Content-addressed and
    write-once zones (normalized, manifests, frozen) are copied with ``--immutable`` so a changed file
    is an error rather than a silent overwrite; the prospective registry is a plain copy because its
    observation log appends. Raw is uploaded last with a move that verifies checksums and deletes the local file only
    after the remote copy is confirmed.

    Returns: 0 when every present zone succeeded; 1 as soon as any zone fails, and the raw move is then
        never attempted so a partial failure cannot delete the only copy.
    """
    for zone in _PUSH_ORDER:
        if zone is BackupZone.RAW:
            if _safe_raw_dir(config) is None:
                logger.warning(
                    "[SYS] event=backup_zone zone=%s action=%s status=%s",
                    zone.value,
                    "move",
                    "skipped",
                )
                _log_zone(zone, "move", "skipped")
                continue
            args = _transfer_args(config, zone, "move", local_first=True)
        else:
            if not _present_local_dir(config, zone):
                _log_zone(zone, "copy", "skipped")
                continue
            args = _transfer_args(config, zone, "copy", local_first=True)
        if _run(args) != 0:
            _log_zone(zone, args[1], "error")
            _log_done("push", "error")
            return 1
        _log_zone(zone, args[1], "ok")
    _log_done("push", "ok")
    return 0


def pull(config: BackupConfig) -> int:
    """Restore normalized, manifests, frozen, and prospective_registry from Drive into an existing or empty data root.

    Raw is never pulled: Bronze is refetched on demand by the data-maintenance repair path.
    Returns: 0 on success, 1 on the first failing zone.
    """
    for zone in _PULL_ZONES:
        args = _transfer_args(config, zone, "copy", local_first=False)
        if _run(args) != 0:
            _log_zone(zone, "copy", "error")
            _log_done("pull", "error")
            return 1
        _log_zone(zone, "copy", "ok")
    _log_done("pull", "ok")
    return 0


def verify(config: BackupConfig) -> int:
    """Check that every local zone file exists remotely with matching content, one-way and read-only.

    Returns: 0 when no difference is reported, 1 otherwise.
    """
    for zone in _PUSH_ORDER:
        if not _present_local_dir(config, zone):
            _log_zone(zone, "check", "skipped")
            continue
        args = _verify_args(config, zone)
        if _run(args) != 0:
            _log_zone(zone, "check", "error")
            _log_done("verify", "error")
            return 1
        _log_zone(zone, "check", "ok")
    _log_done("verify", "ok")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry: ``push`` (default), ``pull``, or ``verify``; returns the process exit code."""
    raw_args = list(argv) if argv is not None else sys.argv[1:]
    command = raw_args[0] if raw_args else "push"
    if command not in ("push", "pull", "verify"):
        return 2
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
    config = resolve_backup_config(os.environ, DataSettings())
    if command == "push":
        return push(config)
    if command == "pull":
        return pull(config)
    return verify(config)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
