"""Unit tests for the trusted-partition catalog."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from src.data.calendar import load_calendar
from src.data.catalog import (
    CatalogSnapshot,
    latest_artifact,
    load_snapshot_visible,
    load_visible,
    resolve_snapshot,
)
from src.data.pipeline import persist_ingest
from src.data.schema import Dataset, spec_for
from src.data.settings import DataSettings
from src.data.storage import RawPayload, UntrustedDatasetError

_RETRIEVED_EARLY = datetime(2024, 2, 1, 5, 0, tzinfo=UTC)
_RETRIEVED_LATE = datetime(2024, 2, 2, 5, 0, tzinfo=UTC)


def _fx_frame(dates: list[date], rates: list[float], retrieved_at: datetime) -> pl.DataFrame:
    spec = spec_for(Dataset.FX)
    n = len(dates)
    return pl.DataFrame(
        {
            "date": list(dates),
            "usdkrw": list(rates),
            "source": ["synthetic"] * n,
            "retrieved_at": [retrieved_at] * n,
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


def test_cat_d01_latest_and_asof(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """CAT-D01-latest-and-asof"""
    root = tmp_path / "catalog"
    root.mkdir()
    monkeypatch.chdir(root)
    settings = DataSettings(data_root="data")

    days = [date(2024, 1, 30), date(2024, 1, 31)]
    persist_ingest(_fx_frame(days, [1300.0, 1301.0], _RETRIEVED_EARLY), Dataset.FX, _payload(_RETRIEVED_EARLY), settings)
    persist_ingest(_fx_frame(days, [1300.5, 1302.0], _RETRIEVED_LATE), Dataset.FX, _payload(_RETRIEVED_LATE), settings)

    artifact = latest_artifact(settings, Dataset.FX)
    assert artifact.manifest.retrieved_at == _RETRIEVED_LATE

    calendar = load_calendar("XNYS")
    visible = load_visible(settings, Dataset.FX, calendar.close_ts(date(2024, 1, 30)))
    assert visible.height == 1
    assert visible.item(0, "usdkrw") == 1300.5

    before_any = load_visible(settings, Dataset.FX, calendar.close_ts(date(2024, 1, 30)) - timedelta(microseconds=1))
    assert before_any.height == 0


def test_cat_d01_missing_manifests_raise(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """CAT-D01-latest-and-asof"""
    root = tmp_path / "empty"
    root.mkdir()
    monkeypatch.chdir(root)

    with pytest.raises(UntrustedDatasetError):
        latest_artifact(DataSettings(data_root="data"), Dataset.FX)


def _macro_frame() -> pl.DataFrame:
    spec = spec_for(Dataset.MACRO)
    return pl.DataFrame(
        {
            "series_id": ["CPI", "CPI"],
            "observation_date": [date(2024, 1, 1), date(2024, 1, 1)],
            "release_date": [
                datetime(2024, 2, 10, tzinfo=UTC),
                datetime(2024, 3, 12, tzinfo=UTC),
            ],
            "value": [1.0, 1.2],
        },
        schema=dict(spec.columns),
    )


def _macro_payload() -> RawPayload:
    return RawPayload(
        provider="synthetic",
        endpoint="macro/vintages",
        request_params={"series": "CPI"},
        retrieved_at=datetime(2024, 3, 13, 5, 0, tzinfo=UTC),
        extension="json",
        content=b"{}",
    )


def test_snapshot_run_remains_pinned_after_ingest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Pinned snapshot keeps original manifest and rows after a later ingest."""
    root = tmp_path / "pinned"
    root.mkdir()
    monkeypatch.chdir(root)
    settings = DataSettings(data_root="data")
    days = [date(2024, 1, 30), date(2024, 1, 31)]
    persist_ingest(_fx_frame(days, [1300.0, 1301.0], _RETRIEVED_EARLY), Dataset.FX, _payload(_RETRIEVED_EARLY), settings)
    snapshot = resolve_snapshot(settings, (Dataset.FX,))
    pinned_stem = Path(snapshot.artifacts[Dataset.FX].manifest_path).stem

    persist_ingest(_fx_frame(days, [1400.0, 1401.0], _RETRIEVED_LATE), Dataset.FX, _payload(_RETRIEVED_LATE), settings)

    calendar = load_calendar("XNYS")
    visible = load_snapshot_visible(snapshot, Dataset.FX, calendar.close_ts(date(2024, 1, 31)))
    assert Path(snapshot.artifacts[Dataset.FX].manifest_path).stem == pinned_stem
    assert visible.item(0, "usdkrw") == 1300.0
    assert load_visible(settings, Dataset.FX, calendar.close_ts(date(2024, 1, 31))).item(0, "usdkrw") == 1400.0


def test_snapshot_provenance_identities_differ_for_shared_parquet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two provenance records sharing one frame hash keep distinct stem identities."""
    root = tmp_path / "shared"
    root.mkdir()
    monkeypatch.chdir(root)
    settings = DataSettings(data_root="data")
    days = [date(2024, 1, 30), date(2024, 1, 31)]
    shared = _fx_frame(days, [1300.0, 1301.0], _RETRIEVED_EARLY)
    persist_ingest(shared, Dataset.FX, _payload(_RETRIEVED_EARLY), settings)
    first = resolve_snapshot(settings, (Dataset.FX,))
    persist_ingest(shared, Dataset.FX, _payload(_RETRIEVED_LATE), settings)
    second = resolve_snapshot(settings, (Dataset.FX,))

    first_stem = Path(first.artifacts[Dataset.FX].manifest_path).stem
    second_stem = Path(second.artifacts[Dataset.FX].manifest_path).stem
    assert first_stem != second_stem
    assert first.artifacts[Dataset.FX].manifest.normalized_sha256 == second.artifacts[Dataset.FX].manifest.normalized_sha256


def test_snapshot_missing_pinned_file_stops_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Removing the pinned Parquet fails closed instead of selecting a newer partition."""
    root = tmp_path / "missing"
    root.mkdir()
    monkeypatch.chdir(root)
    settings = DataSettings(data_root="data")
    days = [date(2024, 1, 30), date(2024, 1, 31)]
    persist_ingest(_fx_frame(days, [1300.0, 1301.0], _RETRIEVED_EARLY), Dataset.FX, _payload(_RETRIEVED_EARLY), settings)
    snapshot = resolve_snapshot(settings, (Dataset.FX,))
    Path(snapshot.artifacts[Dataset.FX].normalized_path).unlink()

    calendar = load_calendar("XNYS")
    with pytest.raises(UntrustedDatasetError):
        load_snapshot_visible(snapshot, Dataset.FX, calendar.close_ts(date(2024, 1, 31)))


def test_snapshot_decision_time_remains_causal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A decision before a later revision sees only the earlier visible vintage."""
    root = tmp_path / "causal"
    root.mkdir()
    monkeypatch.chdir(root)
    settings = DataSettings(data_root="data")
    persist_ingest(_macro_frame(), Dataset.MACRO, _macro_payload(), settings)
    snapshot = resolve_snapshot(settings, (Dataset.MACRO,))

    visible = load_snapshot_visible(snapshot, Dataset.MACRO, datetime(2024, 3, 1, tzinfo=UTC))
    assert visible.height == 1
    assert visible.item(0, "value") == 1.0


def test_snapshot_unrequested_dataset_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Loading a dataset absent from the snapshot fails rather than selecting latest."""
    root = tmp_path / "unrequested"
    root.mkdir()
    monkeypatch.chdir(root)
    settings = DataSettings(data_root="data")
    days = [date(2024, 1, 30), date(2024, 1, 31)]
    persist_ingest(_fx_frame(days, [1300.0, 1301.0], _RETRIEVED_EARLY), Dataset.FX, _payload(_RETRIEVED_EARLY), settings)
    snapshot = resolve_snapshot(settings, (Dataset.FX,))

    calendar = load_calendar("XNYS")
    with pytest.raises(ValueError, match="was not pinned"):
        load_snapshot_visible(snapshot, Dataset.FACTORS, calendar.close_ts(date(2024, 1, 31)))
    with pytest.raises(ValueError, match="timezone-aware"):
        load_snapshot_visible(snapshot, Dataset.FX, calendar.close_ts(date(2024, 1, 31)).replace(tzinfo=None))


def test_snapshot_cross_root_and_binding_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Foreign roots and misbound artifacts fail closed with dataset context."""
    root = tmp_path / "binding"
    root.mkdir()
    monkeypatch.chdir(root)
    settings = DataSettings(data_root="data")
    days = [date(2024, 1, 30), date(2024, 1, 31)]
    persist_ingest(_fx_frame(days, [1300.0, 1301.0], _RETRIEVED_EARLY), Dataset.FX, _payload(_RETRIEVED_EARLY), settings)
    persist_ingest(_macro_frame(), Dataset.MACRO, _macro_payload(), settings)
    snapshot = resolve_snapshot(settings, (Dataset.FX, Dataset.MACRO))
    calendar = load_calendar("XNYS")
    decision_ts = calendar.close_ts(date(2024, 1, 31))

    misbound = CatalogSnapshot(
        artifacts={Dataset.FX: snapshot.artifacts[Dataset.MACRO]},
        data_root=snapshot.data_root,
    )
    with pytest.raises(UntrustedDatasetError, match="mismatch"):
        load_snapshot_visible(misbound, Dataset.FX, decision_ts)

    foreign = CatalogSnapshot(artifacts=dict(snapshot.artifacts), data_root=tmp_path / "elsewhere")
    with pytest.raises(UntrustedDatasetError, match="outside snapshot root"):
        load_snapshot_visible(foreign, Dataset.FX, decision_ts)

    unusable = CatalogSnapshot(artifacts=dict(snapshot.artifacts), data_root=Path("/"))
    with pytest.raises(UntrustedDatasetError, match="unusable"):
        load_snapshot_visible(unusable, Dataset.FX, decision_ts)
