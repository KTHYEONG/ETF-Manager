"""Doctor inspection, repair planning, and guarded maintenance."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path, PurePosixPath

import polars as pl
import pytest

from src.data.calendar import load_calendar
from src.data.catalog import latest_artifact, load_visible
from src.data.doctor import (
    SilverCondition,
    SourceMismatchError,
    UnsupportedSourceError,
    apply_data_maintenance,
    inspect_data,
    plan_data_maintenance,
)
from src.data.pipeline import persist_ingest
from src.data.schema import Dataset, spec_for
from src.data.settings import DataSettings
from src.data.storage import DatasetManifest, RawPayload, UntrustedDatasetError

_RETRIEVED_EARLY = datetime(2024, 2, 1, 5, 0, tzinfo=UTC)
_RETRIEVED_MIDDLE = datetime(2024, 2, 1, 17, 0, tzinfo=UTC)
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


def _payload_with(retrieved_at: datetime, content: bytes) -> RawPayload:
    return RawPayload(
        provider="synthetic",
        endpoint="usdkrw/daily",
        request_params={"interval": "daily"},
        retrieved_at=retrieved_at,
        extension="json",
        content=content,
    )


def _raw_path(root: Path, manifest: DatasetManifest) -> Path:
    return root / "data" / Path(*manifest.raw_artifact.relative_path.parts)


def _snapshot_files(root: Path) -> dict[str, bytes]:
    data_root = root / "data"
    if not data_root.is_dir():
        return {}
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(data_root.rglob("*"))
        if path.is_file()
    }


def _tamper_parquet(parquet_path: Path) -> None:
    tampered = pl.read_parquet(parquet_path).with_columns(pl.col("usdkrw") + 1.0)
    tampered.write_parquet(parquet_path)


def test_inspect_and_plan_are_dry_without_bronze(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing Bronze is reported without any fetch, write, or deletion."""
    root = tmp_path / "dry"
    root.mkdir()
    monkeypatch.chdir(root)
    settings = DataSettings(data_root="data")

    days = [date(2024, 1, 30), date(2024, 1, 31)]
    artifact = persist_ingest(
        _fx_frame(days, [1300.0, 1301.0], _RETRIEVED_EARLY),
        Dataset.FX,
        _payload_with(_RETRIEVED_EARLY, b'{"v": 1}'),
        settings,
    )
    _raw_path(root, artifact.manifest).unlink()
    before = _snapshot_files(root)

    report = inspect_data(settings)
    assert len(report.datasets) == 1
    assert report.datasets[0].condition is SilverCondition.MISSING_BRONZE

    plan = plan_data_maintenance(settings)
    assert len(plan.repairs) == 1
    assert _snapshot_files(root) == before
    assert not _raw_path(root, artifact.manifest).exists()


def test_apply_restores_bronze_from_matching_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A response matching the recorded SHA restores the content address with Silver unchanged."""
    root = tmp_path / "repair"
    root.mkdir()
    monkeypatch.chdir(root)
    settings = DataSettings(data_root="data")

    days = [date(2024, 1, 30), date(2024, 1, 31)]
    content = b'{"v": 7}'
    artifact = persist_ingest(
        _fx_frame(days, [1300.0, 1301.0], _RETRIEVED_EARLY),
        Dataset.FX,
        _payload_with(_RETRIEVED_EARLY, content),
        settings,
    )
    raw_path = _raw_path(root, artifact.manifest)
    raw_path.unlink()
    stem_before = artifact.manifest_path.stem
    parquet_before = artifact.normalized_path.read_bytes()
    seen: dict[str, object] = {}
    calls = {"count": 0}

    def fake_fetcher(dataset: Dataset, manifest: DatasetManifest) -> bytes | None:
        calls["count"] += 1
        seen["dataset"] = dataset
        seen["provider"] = manifest.provider
        seen["endpoint"] = manifest.endpoint
        seen["request_params"] = dict(manifest.request_params)
        return content

    plan = plan_data_maintenance(settings)
    report = apply_data_maintenance(plan, settings, fetcher=fake_fetcher)

    assert calls["count"] == 1
    assert seen["dataset"] is Dataset.FX
    assert seen["provider"] == "synthetic"
    assert seen["endpoint"] == "usdkrw/daily"
    assert seen["request_params"] == {"interval": "daily"}
    assert raw_path.read_bytes() == content
    assert artifact.manifest_path.stem == stem_before
    assert artifact.normalized_path.read_bytes() == parquet_before
    assert len(report.datasets) == 1
    assert report.datasets[0].condition is SilverCondition.HEALTHY


def test_apply_rejects_mismatched_vendor_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Different vendor bytes are refused without overwriting identity or pruning."""
    root = tmp_path / "mismatch"
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
    _raw_path(root, late_art.manifest).unlink()
    stems_before = sorted(p.stem for p in (root / "data" / "manifests" / "fx").glob("*.json"))

    def wrong_fetcher(dataset: Dataset, manifest: DatasetManifest) -> bytes | None:
        return b'{"v": "tampered"}'

    plan = plan_data_maintenance(settings)
    with pytest.raises(SourceMismatchError):
        apply_data_maintenance(plan, settings, fetcher=wrong_fetcher)

    assert not _raw_path(root, late_art.manifest).exists()
    assert sorted(p.stem for p in (root / "data" / "manifests" / "fx").glob("*.json")) == stems_before
    assert early_art.manifest_path.is_file()


def test_apply_without_fetcher_reports_unsupported_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A repair with no fetch capability raises the specific unsupported-source error."""
    root = tmp_path / "unsupported"
    root.mkdir()
    monkeypatch.chdir(root)
    settings = DataSettings(data_root="data")

    days = [date(2024, 1, 30), date(2024, 1, 31)]
    artifact = persist_ingest(
        _fx_frame(days, [1300.0, 1301.0], _RETRIEVED_EARLY),
        Dataset.FX,
        _payload_with(_RETRIEVED_EARLY, b'{"v": 1}'),
        settings,
    )
    _raw_path(root, artifact.manifest).unlink()

    plan = plan_data_maintenance(settings)
    with pytest.raises(UnsupportedSourceError):
        apply_data_maintenance(plan, settings)


def test_plan_rejects_unrecoverable_silver(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A lone damaged partition cannot be planned when no trusted history remains."""
    root = tmp_path / "unrecoverable"
    root.mkdir()
    monkeypatch.chdir(root)
    settings = DataSettings(data_root="data")

    days = [date(2024, 1, 30), date(2024, 1, 31)]
    artifact = persist_ingest(
        _fx_frame(days, [1300.0, 1301.0], _RETRIEVED_EARLY),
        Dataset.FX,
        _payload_with(_RETRIEVED_EARLY, b'{"v": 1}'),
        settings,
    )
    _tamper_parquet(artifact.normalized_path)

    report = inspect_data(settings)
    assert report.datasets[0].condition is SilverCondition.UNRECOVERABLE
    with pytest.raises(UntrustedDatasetError):
        plan_data_maintenance(settings)
    assert artifact.manifest_path.is_file()


def test_apply_rebuilds_damaged_silver_from_history(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A damaged latest is rebuilt from trusted history with full key coverage."""
    root = tmp_path / "rebuild"
    root.mkdir()
    monkeypatch.chdir(root)
    settings = DataSettings(data_root="data")

    first_days = [date(2024, 1, 30), date(2024, 1, 31)]
    later_days = [date(2024, 1, 31), date(2024, 2, 1)]
    persist_ingest(
        _fx_frame(first_days, [1300.0, 1301.0], _RETRIEVED_EARLY),
        Dataset.FX,
        _payload_with(_RETRIEVED_EARLY, b'{"v": 1}'),
        settings,
    )
    persist_ingest(
        _fx_frame(later_days, [1301.5, 1302.0], _RETRIEVED_MIDDLE),
        Dataset.FX,
        _payload_with(_RETRIEVED_MIDDLE, b'{"v": 2}'),
        settings,
    )
    latest_art = persist_ingest(
        _fx_frame(later_days, [1305.0, 1306.0], _RETRIEVED_LATE),
        Dataset.FX,
        _payload_with(_RETRIEVED_LATE, b'{"v": 3}'),
        settings,
    )
    _tamper_parquet(latest_art.normalized_path)

    report = inspect_data(settings)
    assert report.datasets[0].condition is SilverCondition.DAMAGED_SILVER
    plan = plan_data_maintenance(settings)
    assert len(plan.repairs) == 1

    result = apply_data_maintenance(plan, settings)
    assert result.datasets[0].condition is SilverCondition.HEALTHY

    selected = latest_artifact(settings, Dataset.FX)
    assert selected.manifest.row_count == 3
    cutoff = load_calendar("XNYS").close_ts(date(2024, 2, 1))
    visible = load_visible(settings, Dataset.FX, cutoff)
    assert sorted(visible.get_column("date").to_list()) == [
        date(2024, 1, 30),
        date(2024, 1, 31),
        date(2024, 2, 1),
    ]


def test_apply_fails_before_publication_without_complete_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A damaged latest with only one trusted ancestor cannot be rebuilt."""
    root = tmp_path / "incomplete"
    root.mkdir()
    monkeypatch.chdir(root)
    settings = DataSettings(data_root="data")

    days = [date(2024, 1, 30), date(2024, 1, 31)]
    persist_ingest(
        _fx_frame(days, [1300.0, 1301.0], _RETRIEVED_EARLY),
        Dataset.FX,
        _payload_with(_RETRIEVED_EARLY, b'{"v": 1}'),
        settings,
    )
    middle_art = persist_ingest(
        _fx_frame(days, [1301.0, 1302.0], _RETRIEVED_MIDDLE),
        Dataset.FX,
        _payload_with(_RETRIEVED_MIDDLE, b'{"v": 2}'),
        settings,
    )
    latest_art = persist_ingest(
        _fx_frame(days, [1302.0, 1303.0], _RETRIEVED_LATE),
        Dataset.FX,
        _payload_with(_RETRIEVED_LATE, b'{"v": 3}'),
        settings,
    )
    _tamper_parquet(middle_art.normalized_path)
    _tamper_parquet(latest_art.normalized_path)
    manifests_before = sorted((root / "data" / "manifests" / "fx").glob("*.json"))

    report = inspect_data(settings)
    assert report.datasets[0].condition is SilverCondition.DAMAGED_SILVER
    plan = plan_data_maintenance(settings)
    with pytest.raises(UntrustedDatasetError):
        apply_data_maintenance(plan, settings)
    assert sorted((root / "data" / "manifests" / "fx").glob("*.json")) == manifests_before


def test_completed_repair_plans_no_further_actions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A second planning pass after a successful apply finds nothing left to repair."""
    root = tmp_path / "idempotent"
    root.mkdir()
    monkeypatch.chdir(root)
    settings = DataSettings(data_root="data")

    days = [date(2024, 1, 30), date(2024, 1, 31)]
    content = b'{"v": 9}'
    artifact = persist_ingest(
        _fx_frame(days, [1300.0, 1301.0], _RETRIEVED_EARLY),
        Dataset.FX,
        _payload_with(_RETRIEVED_EARLY, content),
        settings,
    )
    _raw_path(root, artifact.manifest).unlink()

    def fake_fetcher(dataset: Dataset, manifest: DatasetManifest) -> bytes | None:
        return content

    apply_data_maintenance(plan_data_maintenance(settings), settings, fetcher=fake_fetcher)

    repeat = plan_data_maintenance(settings)
    assert repeat.repairs == ()
    assert len(repeat.findings.datasets) == 1
    assert repeat.findings.datasets[0].condition is SilverCondition.HEALTHY


def test_apply_rejects_stale_plan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A plan derived before a concurrent publication is refused before any deletion."""
    root = tmp_path / "stale"
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
    persist_ingest(
        _fx_frame(days, [1300.5, 1302.0], _RETRIEVED_LATE),
        Dataset.FX,
        _payload_with(_RETRIEVED_LATE, b'{"v": 2}'),
        settings,
    )
    plan = plan_data_maintenance(settings)
    assert plan.repairs == ()

    persist_ingest(
        _fx_frame(days, [1303.0, 1304.0], _RETRIEVED_LATE + timedelta(hours=1)),
        Dataset.FX,
        _payload_with(_RETRIEVED_LATE + timedelta(hours=1), b'{"v": 3}'),
        settings,
    )
    with pytest.raises(UntrustedDatasetError):
        apply_data_maintenance(plan, settings)
    assert early_art.manifest_path.is_file()


def test_apply_rejects_unexpected_bronze_extension(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A recorded raw filename outside the archive contract fails repair verification."""
    root = tmp_path / "extension"
    root.mkdir()
    monkeypatch.chdir(root)
    settings = DataSettings(data_root="data")

    days = [date(2024, 1, 30), date(2024, 1, 31)]
    content = b'{"v": 4}'
    artifact = persist_ingest(
        _fx_frame(days, [1300.0, 1301.0], _RETRIEVED_EARLY),
        Dataset.FX,
        _payload_with(_RETRIEVED_EARLY, content),
        settings,
    )
    document = json.loads(artifact.manifest_path.read_text(encoding="utf-8"))
    raw_section = document["raw_artifact"]
    assert isinstance(raw_section, dict)
    evil_relative = PurePosixPath(str(raw_section["relative_path"])).with_name("payload.ZIP")
    raw_section["relative_path"] = evil_relative.as_posix()
    evil_bytes = json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")
    evil_path = artifact.manifest_path.parent / f"{hashlib.sha256(evil_bytes).hexdigest()}.json"
    evil_path.write_bytes(evil_bytes)
    artifact.manifest_path.unlink()

    def fake_fetcher(dataset: Dataset, manifest: DatasetManifest) -> bytes | None:
        return content

    plan = plan_data_maintenance(settings)
    with pytest.raises(UntrustedDatasetError):
        apply_data_maintenance(plan, settings, fetcher=fake_fetcher)


def test_inspect_skips_empty_and_unknown_partitions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty or foreign manifest directories contribute no dataset findings."""
    root = tmp_path / "skipped"
    root.mkdir()
    monkeypatch.chdir(root)
    settings = DataSettings(data_root="data")

    assert inspect_data(settings).datasets == ()

    (root / "data" / "manifests" / "fx").mkdir(parents=True)
    (root / "data" / "manifests" / "foreign").mkdir(parents=True)
    (root / "data" / "manifests" / "stray.json").write_text("{}", encoding="utf-8")

    assert inspect_data(settings).datasets == ()
