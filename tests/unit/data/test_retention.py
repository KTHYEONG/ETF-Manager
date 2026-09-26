"""Retention prune planning and application."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from src.data.catalog import latest_artifact
from src.data.pipeline import persist_ingest
from src.data.retention import apply_prune, plan_prune
from src.data.schema import Dataset, spec_for
from src.data.settings import DataSettings
from src.data.storage import RawPayload, UntrustedDatasetError

_RETRIEVED_EARLY = datetime(2024, 2, 1, 5, 0, tzinfo=UTC)
_RETRIEVED_LATE = datetime(2024, 2, 2, 5, 0, tzinfo=UTC)


def _fx_frame(dates: list[date], rates: list[float], retrieved_at: datetime) -> pl.DataFrame:
    spec = spec_for(Dataset.FX)
    return pl.DataFrame(
        {
            "date": list(dates),
            "usdkrw": list(rates),
            "source": ["synthetic"] * len(dates),
            "retrieved_at": [retrieved_at] * len(dates),
        },
        schema=dict(spec.columns),
    )


def _payload(retrieved_at: datetime) -> RawPayload:
    return RawPayload(
        provider="synthetic",
        endpoint="usdkrw/daily",
        request_params={"interval": "daily"},
        retrieved_at=retrieved_at,
        extension="json",
        content=b"{}",
    )


def _payload_with(retrieved_at: datetime, content: bytes) -> RawPayload:
    return RawPayload(
        provider="synthetic",
        endpoint="usdkrw/daily",
        request_params={"interval": "daily"},
        retrieved_at=retrieved_at,
        extension="json",
        content=content,
    )


def _write_result_pin(settings: DataSettings, name: str, payload: dict[str, object]) -> Path:
    path = settings.resolved_data_root() / "runs" / "retention_probe" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_plan_prune_keeps_latest_drops_stale_and_nport_mirrors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "retention"
    root.mkdir()
    monkeypatch.chdir(root)
    settings = DataSettings(data_root="data")

    days = [date(2024, 1, 30), date(2024, 1, 31)]
    early_art = persist_ingest(
        _fx_frame(days, [1300.0, 1301.0], _RETRIEVED_EARLY),
        Dataset.FX,
        _payload(_RETRIEVED_EARLY),
        settings,
    )
    late_art = persist_ingest(
        _fx_frame(days, [1300.5, 1302.0], _RETRIEVED_LATE),
        Dataset.FX,
        _payload(_RETRIEVED_LATE),
        settings,
    )

    nport_zip = settings.resolved_data_root() / "raw" / "sec" / "nport" / "2019q4.zip"
    nport_zip.parent.mkdir(parents=True, exist_ok=True)
    nport_zip.write_bytes(b"fake nport zip mirror")

    plan = plan_prune(settings, migrate_results_layout=False)

    assert early_art.manifest_path in plan.to_delete
    assert early_art.normalized_path in plan.to_delete
    assert late_art.manifest_path in plan.retained_manifests
    assert late_art.normalized_path in plan.retained_parquets
    assert nport_zip in plan.to_delete

    dry_report = apply_prune(plan, dry_run=True)
    assert dry_report.dry_run is True
    assert early_art.manifest_path.is_file()
    assert nport_zip.is_file()

    apply_report = apply_prune(plan, dry_run=False)
    assert apply_report.dry_run is False
    assert not early_art.manifest_path.is_file()
    assert not nport_zip.is_file()
    assert late_art.manifest_path.is_file()

    latest = latest_artifact(settings, Dataset.FX)
    assert latest.manifest.normalized_sha256 == late_art.manifest.normalized_sha256


def test_plan_prune_stages_legacy_flat_results_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Ancient flat result dirs are staged into the per-layout staging dirs."""
    root = tmp_path / "retention"
    root.mkdir()
    monkeypatch.chdir(root)
    settings = DataSettings(data_root="data")
    legacy = root / "data" / "thesis_reports"
    legacy.mkdir(parents=True)
    (legacy / "wave_old.json").write_text("{}", encoding="utf-8")

    plan = plan_prune(settings, keep_latest_only=True, drop_nport_zip_mirrors=False)

    assert (legacy / "wave_old.json", root / "data" / "runs" / "thesis" / "wave_old.json") in plan.to_migrate


def test_plan_prune_rejects_malformed_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A malformed manifest fails planning without proposing deletions."""
    root = tmp_path / "malformed"
    root.mkdir()
    monkeypatch.chdir(root)
    settings = DataSettings(data_root="data")

    days = [date(2024, 1, 30), date(2024, 1, 31)]
    artifact = persist_ingest(
        _fx_frame(days, [1300.0, 1301.0], _RETRIEVED_EARLY),
        Dataset.FX,
        _payload(_RETRIEVED_EARLY),
        settings,
    )
    (artifact.manifest_path.parent / "broken.json").write_bytes(b"not json{{")

    with pytest.raises(UntrustedDatasetError):
        plan_prune(settings, migrate_results_layout=False)
    assert artifact.manifest_path.is_file()
    assert artifact.normalized_path.is_file()


def test_plan_prune_rejects_corrupt_parquet(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A Parquet whose hash no longer matches fails planning."""
    root = tmp_path / "corrupt"
    root.mkdir()
    monkeypatch.chdir(root)
    settings = DataSettings(data_root="data")

    days = [date(2024, 1, 30), date(2024, 1, 31)]
    artifact = persist_ingest(
        _fx_frame(days, [1300.0, 1301.0], _RETRIEVED_EARLY),
        Dataset.FX,
        _payload(_RETRIEVED_EARLY),
        settings,
    )
    tampered = pl.read_parquet(artifact.normalized_path).with_columns(pl.col("usdkrw") + 1.0)
    tampered.write_parquet(artifact.normalized_path)

    with pytest.raises(UntrustedDatasetError):
        plan_prune(settings, migrate_results_layout=False)


def test_plan_prune_keeps_shared_parquet(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Two manifests over one Parquet retire only the older manifest."""
    root = tmp_path / "shared_parquet"
    root.mkdir()
    monkeypatch.chdir(root)
    settings = DataSettings(data_root="data")

    days = [date(2024, 1, 30), date(2024, 1, 31)]
    frame = _fx_frame(days, [1300.0, 1301.0], _RETRIEVED_EARLY)
    early_art = persist_ingest(frame, Dataset.FX, _payload(_RETRIEVED_EARLY), settings)
    late_payload = RawPayload(
        provider="synthetic",
        endpoint="usdkrw/daily",
        request_params={"interval": "daily"},
        retrieved_at=_RETRIEVED_LATE,
        extension="json",
        content=b'{"rows": ["late"]}',
    )
    late_art = persist_ingest(frame, Dataset.FX, late_payload, settings)

    assert early_art.normalized_path == late_art.normalized_path
    assert early_art.manifest_path != late_art.manifest_path

    plan = plan_prune(settings, migrate_results_layout=False)
    assert early_art.manifest_path in plan.to_delete
    assert late_art.manifest_path in plan.retained_manifests
    assert early_art.normalized_path not in plan.to_delete
    assert early_art.normalized_path in plan.retained_parquets


def test_plan_prune_keeps_shared_raw(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """One raw payload stays while any retained manifest references it."""
    root = tmp_path / "shared_raw"
    root.mkdir()
    monkeypatch.chdir(root)
    settings = DataSettings(data_root="data")

    days = [date(2024, 1, 30), date(2024, 1, 31)]
    persist_ingest(_fx_frame(days, [1300.0, 1301.0], _RETRIEVED_EARLY), Dataset.FX, _payload(_RETRIEVED_EARLY), settings)
    persist_ingest(_fx_frame(days, [1300.5, 1302.0], _RETRIEVED_LATE), Dataset.FX, _payload(_RETRIEVED_LATE), settings)

    raw_files = sorted((root / "data" / "raw").rglob("payload.*"))
    assert len(raw_files) == 1

    plan = plan_prune(settings, keep_latest_only=False, migrate_results_layout=False)
    assert plan.to_delete == ()
    assert len(plan.retained_manifests) == 2
    assert raw_files[0] not in plan.to_delete


def test_apply_prune_dry_run_is_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Default application preserves every path in the plan."""
    root = tmp_path / "dry_run"
    root.mkdir()
    monkeypatch.chdir(root)
    settings = DataSettings(data_root="data")

    days = [date(2024, 1, 30), date(2024, 1, 31)]
    early_art = persist_ingest(
        _fx_frame(days, [1300.0, 1301.0], _RETRIEVED_EARLY),
        Dataset.FX,
        _payload(_RETRIEVED_EARLY),
        settings,
    )
    persist_ingest(
        _fx_frame(days, [1300.5, 1302.0], _RETRIEVED_LATE),
        Dataset.FX,
        _payload(_RETRIEVED_LATE),
        settings,
    )
    legacy = root / "data" / "thesis_reports"
    legacy.mkdir(parents=True)
    (legacy / "wave_old.json").write_text("{}", encoding="utf-8")

    plan = plan_prune(settings)
    assert plan.to_delete != ()
    assert plan.to_migrate != ()

    report = apply_prune(plan)
    assert report.dry_run is True
    assert report.deleted == ()
    assert report.migrated == ()
    assert early_art.manifest_path.is_file()
    assert early_art.normalized_path.is_file()
    assert (legacy / "wave_old.json").is_file()
    assert not (root / "data" / "runs" / "thesis" / "wave_old.json").exists()


def test_plan_prune_retains_latest_silver_with_missing_bronze(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A valid latest Silver stays retained while its missing Bronze is reported."""
    root = tmp_path / "missing_bronze"
    root.mkdir()
    monkeypatch.chdir(root)
    settings = DataSettings(data_root="data")

    days = [date(2024, 1, 30), date(2024, 1, 31)]
    artifact = persist_ingest(
        _fx_frame(days, [1300.0, 1301.0], _RETRIEVED_EARLY), Dataset.FX, _payload(_RETRIEVED_EARLY), settings
    )
    raw_path = root / "data" / Path(*artifact.manifest.raw_artifact.relative_path.parts)
    raw_path.unlink()

    plan = plan_prune(settings, migrate_results_layout=False)
    assert artifact.manifest_path in plan.retained_manifests
    assert artifact.normalized_path in plan.retained_parquets
    assert artifact.manifest_path not in plan.to_delete
    assert artifact.normalized_path not in plan.to_delete
    assert artifact.manifest.raw_artifact.sha256 in plan.missing_raw


def test_plan_prune_retains_pinned_old_silver(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A result-pinned older manifest survives while unrelated old Silver is prunable."""
    root = tmp_path / "pinned_old"
    root.mkdir()
    monkeypatch.chdir(root)
    settings = DataSettings(data_root="data")

    days = [date(2024, 1, 30), date(2024, 1, 31)]
    middle_ts = _RETRIEVED_EARLY + timedelta(hours=12)
    early_art = persist_ingest(
        _fx_frame(days, [1300.0, 1301.0], _RETRIEVED_EARLY),
        Dataset.FX,
        _payload_with(_RETRIEVED_EARLY, b'{"v": 1}'),
        settings,
    )
    middle_art = persist_ingest(
        _fx_frame(days, [1301.0, 1302.0], middle_ts),
        Dataset.FX,
        _payload_with(middle_ts, b'{"v": 2}'),
        settings,
    )
    late_art = persist_ingest(
        _fx_frame(days, [1302.0, 1303.0], _RETRIEVED_LATE),
        Dataset.FX,
        _payload_with(_RETRIEVED_LATE, b'{"v": 3}'),
        settings,
    )
    _write_result_pin(settings, "pinned.json", {"manifest_hash": early_art.manifest_path.stem})

    plan = plan_prune(settings, migrate_results_layout=False)
    assert early_art.manifest_path in plan.retained_manifests
    assert early_art.normalized_path in plan.retained_parquets
    assert late_art.manifest_path in plan.retained_manifests
    assert middle_art.manifest_path in plan.to_delete
    assert middle_art.normalized_path in plan.to_delete
    assert early_art.normalized_path not in plan.to_delete


def test_plan_prune_legacy_frame_hash_pin_preserves_all_sharing_manifests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A legacy frame-hash-only pin retains every manifest sharing that frame."""
    root = tmp_path / "legacy_pin"
    root.mkdir()
    monkeypatch.chdir(root)
    settings = DataSettings(data_root="data")

    days = [date(2024, 1, 30), date(2024, 1, 31)]
    frame = _fx_frame(days, [1300.0, 1301.0], _RETRIEVED_EARLY)
    first = persist_ingest(frame, Dataset.FX, _payload(_RETRIEVED_EARLY), settings)
    late_payload = RawPayload(
        provider="synthetic",
        endpoint="usdkrw/daily",
        request_params={"interval": "daily"},
        retrieved_at=_RETRIEVED_LATE,
        extension="json",
        content=b'{"rows": ["late"]}',
    )
    second = persist_ingest(frame, Dataset.FX, late_payload, settings)
    assert first.normalized_path == second.normalized_path
    assert first.manifest_path != second.manifest_path

    _write_result_pin(settings, "legacy.json", {"manifest_hash": first.manifest.normalized_sha256})

    plan = plan_prune(settings, migrate_results_layout=False)
    assert first.manifest_path in plan.retained_manifests
    assert second.manifest_path in plan.retained_manifests
    assert first.manifest_path not in plan.to_delete
    assert second.manifest_path not in plan.to_delete
    assert first.normalized_path not in plan.to_delete


def test_plan_prune_reports_missing_historical_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A well-formed pin with no candidate is reported without deleting matches."""
    root = tmp_path / "missing_pin"
    root.mkdir()
    monkeypatch.chdir(root)
    settings = DataSettings(data_root="data")

    days = [date(2024, 1, 30), date(2024, 1, 31)]
    early_art = persist_ingest(
        _fx_frame(days, [1300.0, 1301.0], _RETRIEVED_EARLY), Dataset.FX, _payload(_RETRIEVED_EARLY), settings
    )
    late_art = persist_ingest(
        _fx_frame(days, [1300.5, 1302.0], _RETRIEVED_LATE), Dataset.FX, _payload(_RETRIEVED_LATE), settings
    )
    ghost = "ab" * 32
    ghost_exact = "cd" * 32
    _write_result_pin(
        settings,
        "ghost.json",
        {"manifest_hashes": {"fx": ghost}, "manifest_sha256": ghost_exact},
    )

    plan = plan_prune(settings, migrate_results_layout=False)
    assert ghost in plan.missing_evidence
    assert ghost_exact in plan.missing_evidence
    assert late_art.manifest_path in plan.retained_manifests
    assert early_art.manifest_path in plan.to_delete


def test_plan_prune_skips_absent_and_unreadable_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Absent markers and unreadable evidence never pin, block, or vanish Silver."""
    root = tmp_path / "skipped_evidence"
    root.mkdir()
    monkeypatch.chdir(root)
    settings = DataSettings(data_root="data")

    days = [date(2024, 1, 30), date(2024, 1, 31)]
    early_art = persist_ingest(
        _fx_frame(days, [1300.0, 1301.0], _RETRIEVED_EARLY), Dataset.FX, _payload(_RETRIEVED_EARLY), settings
    )
    late_art = persist_ingest(
        _fx_frame(days, [1300.5, 1302.0], _RETRIEVED_LATE), Dataset.FX, _payload(_RETRIEVED_LATE), settings
    )
    ghost = "ef" * 32
    _write_result_pin(
        settings,
        "absent.json",
        {"manifest_hash": None, "manifest_hashes": {"fx": None, "cpi": "NO_TRUSTED_CPI"}, "manifest_sha256s": None},
    )
    _write_result_pin(settings, "listed.json", {"manifest_hashes": [ghost]})
    probe_dir = settings.resolved_data_root() / "runs" / "retention_probe"
    (probe_dir / "corrupt.json").write_bytes(b"not json{{")
    (probe_dir / "array.json").write_text("[1, 2]", encoding="utf-8")

    plan = plan_prune(settings, migrate_results_layout=False)
    assert ghost in plan.missing_evidence
    assert late_art.manifest_path in plan.retained_manifests
    assert early_art.manifest_path in plan.to_delete


def test_plan_prune_malformed_pin_blocks_planning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A malformed value in a recognized pin field returns no destructive plan."""
    root = tmp_path / "malformed_pin"
    root.mkdir()
    monkeypatch.chdir(root)
    settings = DataSettings(data_root="data")

    days = [date(2024, 1, 30), date(2024, 1, 31)]
    persist_ingest(
        _fx_frame(days, [1300.0, 1301.0], _RETRIEVED_EARLY), Dataset.FX, _payload(_RETRIEVED_EARLY), settings
    )
    _write_result_pin(settings, "bad.json", {"manifest_hash": "bad-value"})

    with pytest.raises(UntrustedDatasetError):
        plan_prune(settings, migrate_results_layout=False)

    (settings.resolved_data_root() / "runs" / "retention_probe" / "bad.json").unlink()
    _write_result_pin(settings, "bad_multi.json", {"manifest_hashes": "bad-value"})

    with pytest.raises(UntrustedDatasetError):
        plan_prune(settings, migrate_results_layout=False)

    (settings.resolved_data_root() / "runs" / "retention_probe" / "bad_multi.json").unlink()
    _write_result_pin(settings, "bad_type.json", {"manifest_hash": 42})

    with pytest.raises(UntrustedDatasetError):
        plan_prune(settings, migrate_results_layout=False)


def test_plan_prune_retains_only_latest_bronze(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The latest Bronze payload is retained while unreferenced older raw is prunable."""
    root = tmp_path / "latest_bronze"
    root.mkdir()
    monkeypatch.chdir(root)
    settings = DataSettings(data_root="data")

    days = [date(2024, 1, 30), date(2024, 1, 31)]
    early_art = persist_ingest(
        _fx_frame(days, [1300.0, 1301.0], _RETRIEVED_EARLY),
        Dataset.FX,
        _payload_with(_RETRIEVED_EARLY, b'{"v": 1}'),
        settings,
    )
    late_art = persist_ingest(
        _fx_frame(days, [1300.5, 1302.0], _RETRIEVED_LATE),
        Dataset.FX,
        _payload_with(_RETRIEVED_LATE, b'{"v": 2}'),
        settings,
    )
    early_raw = root / "data" / Path(*early_art.manifest.raw_artifact.relative_path.parts)
    late_raw = root / "data" / Path(*late_art.manifest.raw_artifact.relative_path.parts)
    assert early_raw.is_file()
    assert late_raw.is_file()

    plan = plan_prune(settings, migrate_results_layout=False)
    assert plan.raw_sha_to_keep == frozenset({late_art.manifest.raw_artifact.sha256})
    assert early_raw in plan.to_delete
    assert late_raw not in plan.to_delete


def test_plan_prune_retains_shared_parquet_for_pinned_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A Parquet shared by a pinned manifest and latest is never listed for deletion."""
    root = tmp_path / "shared_pinned"
    root.mkdir()
    monkeypatch.chdir(root)
    settings = DataSettings(data_root="data")

    days = [date(2024, 1, 30), date(2024, 1, 31)]
    frame = _fx_frame(days, [1300.0, 1301.0], _RETRIEVED_EARLY)
    early_art = persist_ingest(frame, Dataset.FX, _payload(_RETRIEVED_EARLY), settings)
    late_payload = RawPayload(
        provider="synthetic",
        endpoint="usdkrw/daily",
        request_params={"interval": "daily"},
        retrieved_at=_RETRIEVED_LATE,
        extension="json",
        content=b'{"rows": ["late"]}',
    )
    late_art = persist_ingest(frame, Dataset.FX, late_payload, settings)
    assert early_art.normalized_path == late_art.normalized_path
    _write_result_pin(settings, "pinned.json", {"manifest_sha256": early_art.manifest_path.stem})

    plan = plan_prune(settings, migrate_results_layout=False)
    assert early_art.manifest_path in plan.retained_manifests
    assert late_art.manifest_path in plan.retained_manifests
    assert early_art.normalized_path not in plan.to_delete
    assert early_art.normalized_path in plan.retained_parquets


def test_apply_prune_rechecks_protected_silver_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stale plan naming a retained Silver path must not delete it on apply."""
    root = tmp_path / "recheck"
    root.mkdir()
    monkeypatch.chdir(root)
    settings = DataSettings(data_root="data")

    days = [date(2024, 1, 30), date(2024, 1, 31)]
    persist_ingest(
        _fx_frame(days, [1300.0, 1301.0], _RETRIEVED_EARLY), Dataset.FX, _payload(_RETRIEVED_EARLY), settings
    )
    persist_ingest(
        _fx_frame(days, [1300.5, 1302.0], _RETRIEVED_LATE), Dataset.FX, _payload(_RETRIEVED_LATE), settings
    )
    plan = plan_prune(settings, migrate_results_layout=False)
    assert plan.retained_parquets != ()

    stale = replace(plan, to_delete=plan.to_delete + plan.retained_parquets)
    report = apply_prune(stale, dry_run=False)
    for kept in plan.retained_parquets:
        assert kept.is_file()
        assert kept not in report.deleted


def test_plan_prune_collects_pins_from_new_roots_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pins under runs, frozen, and research are honored; legacy docs/results pins are ignored."""
    root = tmp_path / "new_roots"
    root.mkdir()
    monkeypatch.chdir(root)
    settings = DataSettings(data_root="data")

    days = [date(2024, 1, 30), date(2024, 1, 31)]
    early_art = persist_ingest(
        _fx_frame(days, [1300.0, 1301.0], _RETRIEVED_EARLY),
        Dataset.FX,
        _payload(_RETRIEVED_EARLY),
        settings,
    )
    late_art = persist_ingest(
        _fx_frame(days, [1300.5, 1302.0], _RETRIEVED_LATE),
        Dataset.FX,
        _payload(_RETRIEVED_LATE),
        settings,
    )
    data = settings.resolved_data_root()
    for subdir in ("runs", "frozen", "research"):
        pin_dir = data / subdir / "pins"
        pin_dir.mkdir(parents=True, exist_ok=True)
        (pin_dir / "pin.json").write_text(
            json.dumps({"manifest_hash": early_art.manifest_path.stem}), encoding="utf-8"
        )
    ghost = "ab" * 32
    for legacy in ("docs/results", "records/prospective"):
        legacy_dir = root / legacy
        legacy_dir.mkdir(parents=True, exist_ok=True)
        (legacy_dir / "ghost.json").write_text(json.dumps({"manifest_hash": ghost}), encoding="utf-8")

    plan = plan_prune(settings, migrate_results_layout=False)
    assert early_art.manifest_path in plan.retained_manifests
    assert late_art.manifest_path in plan.retained_manifests
    assert ghost not in plan.missing_evidence


def test_plan_prune_malformed_pin_under_frozen_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A malformed pin value under the frozen root fails planning without deletions."""
    root = tmp_path / "frozen_pin"
    root.mkdir()
    monkeypatch.chdir(root)
    settings = DataSettings(data_root="data")

    days = [date(2024, 1, 30), date(2024, 1, 31)]
    persist_ingest(
        _fx_frame(days, [1300.0, 1301.0], _RETRIEVED_EARLY),
        Dataset.FX,
        _payload(_RETRIEVED_EARLY),
        settings,
    )
    frozen_dir = settings.resolved_data_root() / "frozen"
    frozen_dir.mkdir(parents=True, exist_ok=True)
    (frozen_dir / "bad.json").write_text(json.dumps({"manifest_hash": "bad-value"}), encoding="utf-8")

    with pytest.raises(UntrustedDatasetError):
        plan_prune(settings, migrate_results_layout=False)
