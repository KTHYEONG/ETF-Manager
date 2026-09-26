"""Invariant guards for the Drive backup tool: no delete propagation, raw-last safety."""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pytest

from src.data.settings import DataSettings

_REPO_ROOT = Path(__file__).resolve().parents[3]
_BACKUP_PATH = _REPO_ROOT / "tools" / "devops" / "backup.py"


def _load_backup_module() -> Any:
    name = "etf_manager_backup_tool"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, _BACKUP_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


backup = _load_backup_module()

_ALL_ZONES = ("normalized", "manifests", "frozen", "prospective_registry", "raw")
_FORBIDDEN_VERBS = ("sync", "delete", "deletefile", "purge", "cleanup")

_STUB_SCRIPT = "\n".join(
    [
        "#!/usr/bin/env bash",
        'printf "%s\\n" "$*" >> "$RCLONE_LOG"',
        'fail_zone="${RCLONE_FAIL_ZONE:-}"',
        'if [[ -n "$fail_zone" && "$*" == *"$fail_zone"* ]]; then',
        "  exit 1",
        "fi",
        'exit "${RCLONE_EXIT:-0}"',
        "",
    ]
)


class _Stub:
    def __init__(self, rclone: str, log: Path) -> None:
        self.rclone = rclone
        self.log = log


@pytest.fixture
def stub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Stub:
    """Executable rclone stub logging each argv line; failure driven by env vars."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True)
    log = tmp_path / "rclone.log"
    stub_path = bin_dir / "rclone"
    stub_path.write_text(_STUB_SCRIPT, encoding="utf-8")
    stub_path.chmod(0o755)
    monkeypatch.setenv("RCLONE_LOG", str(log))
    monkeypatch.delenv("RCLONE_EXIT", raising=False)
    monkeypatch.delenv("RCLONE_FAIL_ZONE", raising=False)
    return _Stub(str(stub_path), log)


def _make_root(base: Path, zones: Iterable[str] = _ALL_ZONES) -> Path:
    root = base / "data"
    root.mkdir(parents=True, exist_ok=True)
    for zone in zones:
        zone_dir = root / zone
        zone_dir.mkdir(parents=True, exist_ok=True)
        (zone_dir / "part.parquet").write_bytes(b"payload")
    return root


def _make_config(stub: _Stub, root: Path, remote: str = "testremote:bucket") -> Any:
    return backup.BackupConfig(data_root=root, remote=remote, rclone=stub.rclone)


def _logged_lines(stub: _Stub) -> list[str]:
    if not stub.log.exists():
        return []
    return stub.log.read_text(encoding="utf-8").splitlines()


def _dest_zones(lines: list[str], remote: str) -> list[str]:
    zones: list[str] = []
    for line in lines:
        zones.extend(token[len(remote) + 1 :] for token in line.split() if token.startswith(remote + "/"))
    return zones


def test_push_uploads_zones_in_fixed_order(stub: _Stub, tmp_path: Path) -> None:
    """Push visits every present zone in the contractual raw-last order."""
    root = _make_root(tmp_path)
    assert backup.push(_make_config(stub, root)) == 0
    assert _dest_zones(_logged_lines(stub), "testremote:bucket") == list(_ALL_ZONES)


def test_push_marks_content_addressed_zones_immutable(stub: _Stub, tmp_path: Path) -> None:
    """Write-once zones carry --immutable; append-only ledgers do not; all carry --checksum."""
    root = _make_root(tmp_path)
    assert backup.push(_make_config(stub, root)) == 0
    by_zone = dict(zip(_dest_zones(_logged_lines(stub), "testremote:bucket"), _logged_lines(stub), strict=True))
    for zone in ("normalized", "manifests", "frozen", "raw"):
        assert "--immutable" in by_zone[zone].split()
    assert "--immutable" not in by_zone["prospective_registry"].split()
    for line in _logged_lines(stub):
        assert "--checksum" in line.split()


def test_push_moves_raw_and_copies_other_zones(stub: _Stub, tmp_path: Path) -> None:
    """Only the Bronze step uses move; every other zone uses copy."""
    root = _make_root(tmp_path)
    assert backup.push(_make_config(stub, root)) == 0
    verbs = [line.split()[0] for line in _logged_lines(stub)]
    assert verbs == ["copy", "copy", "copy", "copy", "move"]


def test_push_failure_blocks_raw_removal(
    stub: _Stub, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A frozen failure stops the run before any later zone, keeping the only Bronze copy."""
    root = _make_root(tmp_path)
    monkeypatch.setenv("RCLONE_FAIL_ZONE", "frozen")
    assert backup.push(_make_config(stub, root)) == 1
    assert _dest_zones(_logged_lines(stub), "testremote:bucket") == ["normalized", "manifests", "frozen"]
    assert (root / "raw" / "part.parquet").exists()


def test_no_destructive_verbs_ever(stub: _Stub, tmp_path: Path) -> None:
    """Push, pull, and verify never issue a delete-propagating rclone verb."""
    root = _make_root(tmp_path)
    config = _make_config(stub, root)
    assert backup.push(config) == 0
    assert backup.pull(config) == 0
    assert backup.verify(config) == 0
    for line in _logged_lines(stub):
        assert line.split()[0] in ("copy", "move", "check")
        for forbidden in _FORBIDDEN_VERBS:
            assert forbidden not in line.split()


def test_push_skips_missing_zones(stub: _Stub, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Absent zone directories are skipped as INFO, not failures."""
    root = _make_root(tmp_path, ("normalized", "manifests"))
    with caplog.at_level(logging.INFO):
        assert backup.push(_make_config(stub, root)) == 0
    assert _dest_zones(_logged_lines(stub), "testremote:bucket") == ["normalized", "manifests"]


def test_push_refuses_symlinked_raw(
    stub: _Stub, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A symlinked Bronze source never reaches rclone and warns instead."""
    root = _make_root(tmp_path, ("normalized", "manifests", "frozen", "prospective_registry"))
    outside = tmp_path / "outside-raw"
    outside.mkdir()
    (outside / "part.parquet").write_bytes(b"payload")
    (root / "raw").symlink_to(outside, target_is_directory=True)
    assert backup.push(_make_config(stub, root)) == 0
    assert "raw" not in _dest_zones(_logged_lines(stub), "testremote:bucket")
    assert "raw" in caplog.text
    assert any(record.levelno >= logging.WARNING for record in caplog.records)


def test_pull_restores_without_raw(stub: _Stub, tmp_path: Path) -> None:
    """Pull restores the Silver/ledger zones with normalized first and never touches Bronze."""
    root = tmp_path / "data"
    root.mkdir()
    assert backup.pull(_make_config(stub, root)) == 0
    zones = _dest_zones(_logged_lines(stub), "testremote:bucket")
    assert zones == ["normalized", "manifests", "frozen", "prospective_registry"]
    assert zones.index("normalized") < zones.index("manifests")
    by_zone = dict(zip(zones, _logged_lines(stub), strict=True))
    assert "--immutable" not in by_zone["prospective_registry"].split()


def test_verify_checks_one_way_read_only(
    stub: _Stub, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify compares with one-way check and reports any difference as failure."""
    root = _make_root(tmp_path)
    assert backup.verify(_make_config(stub, root)) == 0
    lines = _logged_lines(stub)
    assert len(lines) == len(_ALL_ZONES)
    for line in lines:
        tokens = line.split()
        assert tokens[0] == "check"
        assert "--one-way" in tokens
    monkeypatch.setenv("RCLONE_EXIT", "1")
    assert backup.verify(_make_config(stub, root)) == 1


def test_remote_and_executable_overrides(
    stub: _Stub, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Environment overrides steer every command; malformed remotes fail closed."""
    root = _make_root(tmp_path)
    monkeypatch.setenv("ETF_MANAGER_BACKUP_REMOTE", "x:y")
    monkeypatch.setenv("RCLONE", stub.rclone)
    config = backup.resolve_backup_config(dict(os.environ), DataSettings())
    assert config.remote == "x:y"
    assert config.rclone == stub.rclone
    config = backup.BackupConfig(data_root=root, remote=config.remote, rclone=config.rclone)
    assert backup.push(config) == 0
    assert _dest_zones(_logged_lines(stub), "x:y") == list(_ALL_ZONES)
    with pytest.raises(ValueError, match="remote"):
        backup.resolve_backup_config({"ETF_MANAGER_BACKUP_REMOTE": ""}, DataSettings())
    with pytest.raises(ValueError, match="remote"):
        backup.resolve_backup_config({"ETF_MANAGER_BACKUP_REMOTE": "no-separator"}, DataSettings())
    with pytest.raises(ValueError, match="absolute"):
        backup.BackupConfig(data_root=Path("relative"), remote="x:y", rclone="rclone")


def test_excludes_present_on_every_command(stub: _Stub, tmp_path: Path) -> None:
    """Every issued command carries the credential exclude patterns."""
    root = _make_root(tmp_path)
    config = _make_config(stub, root)
    assert backup.push(config) == 0
    assert backup.pull(config) == 0
    assert backup.verify(config) == 0
    lines = _logged_lines(stub)
    assert lines
    for line in lines:
        assert ".env*" in line
        assert "*.key" in line
        assert "*_key.txt" in line


def test_main_rejects_unknown_command(stub: _Stub) -> None:
    """An unknown CLI command returns 2 without issuing any rclone command."""
    assert backup.main(["frobnicate"]) == 2
    assert _logged_lines(stub) == []


def test_push_reports_missing_executable_as_failure(tmp_path: Path) -> None:
    """A missing rclone binary is a closed failure, never an exception."""
    root = _make_root(tmp_path)
    config = backup.BackupConfig(
        data_root=root, remote="testremote:bucket", rclone=str(tmp_path / "absent-rclone")
    )
    assert backup.push(config) == 1


def test_main_dispatches_known_commands(monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI entry routes push (default), pull, and verify to their handlers."""
    sentinel = object()
    calls: list[str] = []
    monkeypatch.setattr(backup, "resolve_backup_config", lambda environ, settings: sentinel)
    monkeypatch.setattr(backup, "push", lambda config: calls.append("push") or 0)
    monkeypatch.setattr(backup, "pull", lambda config: calls.append("pull") or 0)
    monkeypatch.setattr(backup, "verify", lambda config: calls.append("verify") or 0)
    assert backup.main(["push"]) == 0
    assert backup.main([]) == 0
    assert backup.main(["pull"]) == 0
    assert backup.main(["verify"]) == 0
    assert calls == ["push", "push", "pull", "verify"]
