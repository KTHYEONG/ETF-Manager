"""Partition shrink guard tests."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl
import pytest

from src.data.calendar import load_calendar
from src.data.ingest_guard import PartitionShrinkError, assert_safe_partition_replacement
from src.data.pipeline import persist_ingest
from src.data.schema import Dataset, spec_for
from src.data.settings import DataSettings
from src.data.storage import RawPayload

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


def test_ingest_guard_blocks_shrinking_fx_partition(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "guard"
    root.mkdir()
    monkeypatch.chdir(root)
    settings = DataSettings(data_root="data")
    cal = load_calendar("XNYS")
    days = list(cal.sessions(date(2023, 8, 1), date(2024, 2, 29)))[:120]
    rates = [1300.0 + i for i in range(len(days))]
    persist_ingest(_fx_frame(days, rates, _RETRIEVED_EARLY), Dataset.FX, _payload(_RETRIEVED_EARLY), settings)

    tiny_days = days[:2]
    tiny = _fx_frame(tiny_days, rates[:2], _RETRIEVED_LATE)
    with pytest.raises(PartitionShrinkError):
        assert_safe_partition_replacement(Dataset.FX, tiny.height, settings)

    persist_ingest(tiny, Dataset.FX, _payload(_RETRIEVED_LATE), settings, allow_shrink=True)
