"""Catalog frame cache behavior."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl
import pytest

from src.data.calendar import load_calendar
from src.data.catalog import clear_catalog_frame_cache, load_visible
from src.data.pipeline import persist_ingest
from src.data.schema import Dataset, spec_for
from src.data.settings import DataSettings
from src.data.storage import DataStore, RawPayload

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
