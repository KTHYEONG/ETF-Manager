"""Unit tests for the trusted-partition catalog."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from src.data.calendar import load_calendar
from src.data.catalog import latest_artifact, load_visible
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
