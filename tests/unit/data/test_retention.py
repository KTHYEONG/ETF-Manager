"""Retention prune planning and application."""

from __future__ import annotations

from datetime import UTC, date, datetime
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

    assert (legacy / "wave_old.json", root / "data" / "results" / "thesis" / "wave_old.json") in plan.to_migrate


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
    assert not (root / "data" / "results" / "thesis" / "wave_old.json").exists()
