"""Integration tests for the ingest write-path seam and composed persistence."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl
import pytest

from src.etf_manager.data.pipeline import ingest, persist_ingest
from src.etf_manager.data.pit import AVAILABLE_AT
from src.etf_manager.data.quality import DataQualityError, FindingSeverity
from src.etf_manager.data.schema import Dataset, spec_for
from src.etf_manager.data.settings import DataSettings
from src.etf_manager.data.storage import RawPayload

TS_DTYPE = pl.Datetime("us", "UTC")
_RETRIEVED_AT = datetime(2024, 2, 1, 5, 0, tzinfo=UTC)


def _prices_frame(dates: list[date], closes: list[float], ticker: str = "AAA") -> pl.DataFrame:
    spec = spec_for(Dataset.PRICES)
    n = len(dates)
    return pl.DataFrame(
        {
            "ticker": [ticker] * n,
            "date": list(dates),
            "open": [value * 0.98 for value in closes],
            "high": [value * 1.02 for value in closes],
            "low": [value * 0.97 for value in closes],
            "close": list(closes),
            "volume": [10_000] * n,
            "adjusted_close": list(closes),
            "dividend": [0.0] * n,
            "split_factor": [1.0] * n,
            "source": ["synthetic"] * n,
            "retrieved_at": [_RETRIEVED_AT] * n,
        },
        schema=dict(spec.columns),
    )


def test_pipe_a11_ingest_stamps_availability() -> None:
    """PIPE-A11-ingest-stamps-availability"""
    raw = _prices_frame([date(2024, 1, 31), date(2024, 2, 1)], [100.0, 101.0])
    raw_snapshot = raw.clone()
    stamped = ingest(raw, Dataset.PRICES)

    assert stamped.height == raw.height
    assert set(stamped.columns) == {*raw.columns, AVAILABLE_AT}
    assert stamped.schema[AVAILABLE_AT] == TS_DTYPE
    assert raw.equals(raw_snapshot)

    broken = raw.drop("date")
    try:
        ingest(broken, Dataset.PRICES)
        raised = False
    except ValueError:
        raised = True
    assert raised is True


@pytest.mark.parametrize("scenario_id", ["PL-B10-composed-persist-ingest"])
def test_composed_persist_ingest(
    scenario_id: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PL-B10-composed-persist-ingest"""
    payload = RawPayload(
        provider="synthetic",
        endpoint="daily/prices",
        request_params={"format": "json"},
        retrieved_at=_RETRIEVED_AT,
        extension="json",
        content=b'{"rows": []}',
    )
    valid_raw = _prices_frame([date(2024, 1, 30), date(2024, 1, 31)], [100.0, 101.0])

    accepted_root = tmp_path / "accepted"
    accepted_root.mkdir()
    monkeypatch.chdir(accepted_root)
    accepted_settings = DataSettings(data_root="data")
    artifact = persist_ingest(valid_raw, Dataset.PRICES, payload, accepted_settings)

    raw_files = sorted((accepted_root / "data" / "raw").rglob("payload.*"))
    assert len(raw_files) == 1
    assert artifact.normalized_path.is_file()
    assert artifact.manifest_path.is_file()
    assert all(
        finding.severity is not FindingSeverity.ERROR
        for finding in artifact.manifest.quality_findings
    )

    rejected_root = tmp_path / "rejected"
    rejected_root.mkdir()
    monkeypatch.chdir(rejected_root)
    malformed_raw = valid_raw.with_columns((pl.col("low") * 10.0).alias("low"))
    with pytest.raises(DataQualityError):
        persist_ingest(malformed_raw, Dataset.PRICES, payload, DataSettings(data_root="data"))

    assert not (rejected_root / "data" / "normalized").exists()
    assert not (rejected_root / "data" / "manifests").exists()
    retained = sorted((rejected_root / "data" / "raw").rglob("payload.*"))
    assert len(retained) == 1
