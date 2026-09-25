"""Catalog frame cache behavior."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from src.data.calendar import load_calendar
from src.data.catalog import clear_catalog_frame_cache, latest_artifact, load_visible
from src.data.pipeline import persist_ingest
from src.data.schema import Dataset, spec_for
from src.data.settings import DataSettings
from src.data.storage import DataStore, RawPayload, UntrustedDatasetError

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


def test_catalog_cache_second_load_visible_avoids_reread(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clear_catalog_frame_cache()
    root = tmp_path / "catalog_cache"
    root.mkdir()
    monkeypatch.chdir(root)
    settings = DataSettings(data_root="data")

    days = [date(2024, 1, 30), date(2024, 1, 31)]
    persist_ingest(_fx_frame(days, [1300.0, 1301.0], _RETRIEVED_EARLY), Dataset.FX, _payload(_RETRIEVED_EARLY), settings)

    calls = {"count": 0}
    original = DataStore.read_normalized

    def counting_read(self, artifact, spec):
        calls["count"] += 1
        return original(self, artifact, spec)

    monkeypatch.setattr(DataStore, "read_normalized", counting_read)

    calendar = load_calendar("XNYS")
    cutoff = calendar.close_ts(date(2024, 1, 31))
    first = load_visible(settings, Dataset.FX, cutoff)
    second = load_visible(settings, Dataset.FX, cutoff)

    assert calls["count"] == 1
    assert first.height == second.height
    assert first.item(0, "usdkrw") == second.item(0, "usdkrw")


def test_catalog_cache_cleared_on_persist_ingest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clear_catalog_frame_cache()
    root = tmp_path / "catalog_cache_clear"
    root.mkdir()
    monkeypatch.chdir(root)
    settings = DataSettings(data_root="data")

    days = [date(2024, 1, 30), date(2024, 1, 31)]
    persist_ingest(_fx_frame(days, [1300.0, 1301.0], _RETRIEVED_EARLY), Dataset.FX, _payload(_RETRIEVED_EARLY), settings)

    calls = {"count": 0}
    original = DataStore.read_normalized

    def counting_read(self, artifact, spec):
        calls["count"] += 1
        return original(self, artifact, spec)

    monkeypatch.setattr(DataStore, "read_normalized", counting_read)

    calendar = load_calendar("XNYS")
    cutoff = calendar.close_ts(date(2024, 1, 31))
    load_visible(settings, Dataset.FX, cutoff)

    persist_ingest(
        _fx_frame(days, [1300.5, 1302.0], _RETRIEVED_LATE),
        Dataset.FX,
        _payload(_RETRIEVED_LATE),
        settings,
    )

    visible = load_visible(settings, Dataset.FX, cutoff)
    jan31 = visible.filter(pl.col("date") == date(2024, 1, 31))
    assert jan31.height == 1
    assert jan31.item(0, "usdkrw") == 1302.0
    assert calls["count"] >= 2


def test_load_visible_rejects_deleted_cached_parquet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A removed Parquet fails closed instead of serving cached rows."""
    clear_catalog_frame_cache()
    root = tmp_path / "deleted_parquet"
    root.mkdir()
    monkeypatch.chdir(root)
    settings = DataSettings(data_root="data")

    days = [date(2024, 1, 30), date(2024, 1, 31)]
    persist_ingest(_fx_frame(days, [1300.0, 1301.0], _RETRIEVED_EARLY), Dataset.FX, _payload(_RETRIEVED_EARLY), settings)
    calendar = load_calendar("XNYS")
    cutoff = calendar.close_ts(date(2024, 1, 31))
    assert load_visible(settings, Dataset.FX, cutoff).height == 2

    artifact = latest_artifact(settings, Dataset.FX)
    Path(artifact.normalized_path).unlink()
    with pytest.raises(UntrustedDatasetError):
        load_visible(settings, Dataset.FX, cutoff)


def test_load_visible_rejects_changed_cached_raw(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A modified raw payload triggers re-verification that rejects the artifact."""
    clear_catalog_frame_cache()
    root = tmp_path / "changed_raw"
    root.mkdir()
    monkeypatch.chdir(root)
    settings = DataSettings(data_root="data")

    days = [date(2024, 1, 30), date(2024, 1, 31)]
    persist_ingest(_fx_frame(days, [1300.0, 1301.0], _RETRIEVED_EARLY), Dataset.FX, _payload(_RETRIEVED_EARLY), settings)
    calendar = load_calendar("XNYS")
    cutoff = calendar.close_ts(date(2024, 1, 31))
    assert load_visible(settings, Dataset.FX, cutoff).height == 2

    artifact = latest_artifact(settings, Dataset.FX)
    raw_path = root / "data" / Path(*artifact.manifest.raw_artifact.relative_path.parts)
    raw_path.write_bytes(b'{"rows": ["tampered"]}')
    with pytest.raises(UntrustedDatasetError):
        load_visible(settings, Dataset.FX, cutoff)


def test_load_visible_isolates_data_roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Two roots sharing a frame hash verify and cache independently."""
    clear_catalog_frame_cache()
    first_root = tmp_path / "root_a"
    second_root = tmp_path / "root_b"
    first_root.mkdir()
    second_root.mkdir()
    first_settings = DataSettings(data_root=first_root / "data")
    second_settings = DataSettings(data_root=second_root / "data")

    days = [date(2024, 1, 30), date(2024, 1, 31)]
    frame = _fx_frame(days, [1300.0, 1301.0], _RETRIEVED_EARLY)
    persist_ingest(frame, Dataset.FX, _payload(_RETRIEVED_EARLY), first_settings)
    persist_ingest(frame, Dataset.FX, _payload(_RETRIEVED_EARLY), second_settings)

    calls = {"count": 0}
    original = DataStore.read_normalized

    def counting_read(self, artifact, spec):
        calls["count"] += 1
        return original(self, artifact, spec)

    monkeypatch.setattr(DataStore, "read_normalized", counting_read)

    calendar = load_calendar("XNYS")
    cutoff = calendar.close_ts(date(2024, 1, 31))
    first = load_visible(first_settings, Dataset.FX, cutoff)
    second = load_visible(second_settings, Dataset.FX, cutoff)
    assert first.equals(second)
    assert calls["count"] == 2
    assert load_visible(first_settings, Dataset.FX, cutoff).equals(first)
    assert load_visible(second_settings, Dataset.FX, cutoff).equals(second)
    assert calls["count"] == 2


def test_latest_artifact_selects_newer_provenance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The same frame with later lineage is selected and stays readable."""
    clear_catalog_frame_cache()
    root = tmp_path / "new_provenance"
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

    selected = latest_artifact(settings, Dataset.FX)
    assert selected.manifest.retrieved_at == _RETRIEVED_LATE
    assert selected.manifest_path == second.manifest_path

    calendar = load_calendar("XNYS")
    visible = load_visible(settings, Dataset.FX, calendar.close_ts(date(2024, 1, 31)))
    assert visible.height == 2


def test_latest_artifact_rejects_tampered_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A manifest with rewritten lineage is rejected instead of selected."""
    clear_catalog_frame_cache()
    root = tmp_path / "tampered_manifest"
    root.mkdir()
    monkeypatch.chdir(root)
    settings = DataSettings(data_root="data")

    days = [date(2024, 1, 30), date(2024, 1, 31)]
    persist_ingest(_fx_frame(days, [1300.0, 1301.0], _RETRIEVED_EARLY), Dataset.FX, _payload(_RETRIEVED_EARLY), settings)

    selected = latest_artifact(settings, Dataset.FX)
    document = json.loads(selected.manifest_path.read_text(encoding="utf-8"))
    document["retrieved_at"] = (_RETRIEVED_LATE + timedelta(days=1)).isoformat()
    selected.manifest_path.write_bytes(
        json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    with pytest.raises(UntrustedDatasetError):
        latest_artifact(settings, Dataset.FX)
