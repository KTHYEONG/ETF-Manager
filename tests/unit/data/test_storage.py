"""Unit tests for immutable raw storage, canonical hashing, and manifest-bound reads."""

from __future__ import annotations

import re
from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl
import pytest

from src.etf_manager.data.calendar import TradingCalendar, load_calendar
from src.etf_manager.data.pit import stamp_availability
from src.etf_manager.data.quality import validate_frame
from src.etf_manager.data.schema import Dataset, DatasetSpec, spec_for
from src.etf_manager.data.settings import DataSettings
from src.etf_manager.data.storage import (
    DataStore,
    RawPayload,
    UntrustedDatasetError,
    canonical_frame_sha256,
)

_RETRIEVED_AT = datetime(2024, 2, 1, 5, 0, tzinfo=UTC)


def _payload(content: bytes) -> RawPayload:
    return RawPayload(
        provider="synthetic",
        endpoint="daily/prices",
        request_params={"format": "json"},
        retrieved_at=_RETRIEVED_AT,
        extension="json",
        content=content,
    )


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


def _stamped_prices(spec: DatasetSpec, calendar: TradingCalendar) -> pl.DataFrame:
    raw = _prices_frame([date(2024, 1, 30), date(2024, 1, 31)], [100.0, 101.0])
    return stamp_availability(raw, spec, calendar)


@pytest.mark.parametrize("scenario_id", ["ST-B07-raw-immutability"])
def test_raw_immutability(scenario_id: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """ST-B07-raw-immutability"""
    monkeypatch.chdir(tmp_path)
    store = DataStore(DataSettings())

    payload = _payload(b"raw-bytes-1")
    first_artifact = store.store_raw(Dataset.FX, payload)
    second_artifact = store.store_raw(Dataset.FX, _payload(b"raw-bytes-1"))

    assert first_artifact.relative_path == second_artifact.relative_path
    stored_payloads = sorted((tmp_path / "data").rglob("payload.*"))
    assert len(stored_payloads) == 1
    assert stored_payloads[0].read_bytes() == b"raw-bytes-1"

    changed_artifact = store.store_raw(Dataset.FX, _payload(b"raw-bytes-2"))
    assert changed_artifact.sha256 != first_artifact.sha256
    assert changed_artifact.relative_path != first_artifact.relative_path

    parts = first_artifact.relative_path.parts
    assert parts[0] == "raw"
    assert ".." not in parts


@pytest.mark.parametrize("scenario_id", ["ST-B08-manifest-bound-round-trip"])
def test_manifest_bound_round_trip(scenario_id: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """ST-B08-manifest-bound-round-trip"""
    monkeypatch.chdir(tmp_path)
    store = DataStore(DataSettings())
    spec = spec_for(Dataset.PRICES)
    calendar = load_calendar("XNYS")

    frame = _stamped_prices(spec, calendar)
    payload = _payload(b"prices-payload")
    raw_artifact = store.store_raw(Dataset.PRICES, payload)
    report = validate_frame(frame, spec, calendar)
    assert report.has_errors is False

    artifact = store.write_normalized(frame, spec, raw_artifact, payload, report)
    assert artifact.normalized_path.is_file()
    assert artifact.manifest_path.is_file()

    restored = store.read_normalized(artifact, spec)
    assert restored.equals(frame)
    assert artifact.manifest.row_count == frame.height
    assert re.fullmatch(r"[0-9a-f]{64}", artifact.manifest.normalized_sha256) is not None
    assert re.fullmatch(r"[0-9a-f]{64}", artifact.manifest.raw_artifact.sha256) is not None

    tampered = pl.read_parquet(artifact.normalized_path).with_columns(pl.col("close") + 1.0)
    tampered.write_parquet(artifact.normalized_path)
    with pytest.raises(UntrustedDatasetError):
        store.read_normalized(artifact, spec)

    artifact.manifest_path.unlink()
    with pytest.raises(UntrustedDatasetError):
        store.read_normalized(artifact, spec)


@pytest.mark.parametrize("scenario_id", ["ST-B09-canonical-order-hash"])
def test_canonical_order_hash(scenario_id: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """ST-B09-canonical-order-hash"""
    monkeypatch.chdir(tmp_path)
    spec = spec_for(Dataset.PRICES)
    calendar = load_calendar("XNYS")
    frame = _stamped_prices(spec, calendar)
    snapshot = frame.clone()

    canonical_hash = canonical_frame_sha256(frame, spec)
    reversed_hash = canonical_frame_sha256(frame.reverse(), spec)
    assert canonical_hash == reversed_hash
    assert len(canonical_hash) == 64
    assert canonical_hash == canonical_hash.lower()

    shifted = frame.with_columns((pl.col("close") + 0.01).alias("close"))
    assert canonical_frame_sha256(shifted, spec) != canonical_hash

    assert frame.equals(snapshot)
