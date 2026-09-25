"""Unit tests for immutable raw storage, canonical hashing, and manifest-bound reads."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path, PurePosixPath

import polars as pl
import pytest

from src.data.calendar import TradingCalendar, load_calendar
from src.data.pit import stamp_availability
from src.data.quality import validate_frame
from src.data.schema import Dataset, DatasetSpec, spec_for
from src.data.settings import DataSettings
from src.data.storage import (
    DatasetArtifact,
    DataStore,
    RawArtifact,
    RawPayload,
    UntrustedDatasetError,
    canonical_frame_sha256,
    canonical_manifest_sha256,
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


def _publish_prices(
    store: DataStore,
    spec: DatasetSpec,
    calendar: TradingCalendar,
    payload: RawPayload,
) -> DatasetArtifact:
    frame = _stamped_prices(spec, calendar)
    raw_artifact = store.store_raw(Dataset.PRICES, payload)
    report = validate_frame(frame, spec, calendar)
    assert report.has_errors is False
    return store.write_normalized(frame, spec, raw_artifact, payload, report)


def _payload_at(content: bytes, retrieved_at: datetime) -> RawPayload:
    return RawPayload(
        provider="synthetic",
        endpoint="daily/prices",
        request_params={"format": "json"},
        retrieved_at=retrieved_at,
        extension="json",
        content=content,
    )


def test_write_normalized_shares_parquet_across_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Identical rows with distinct lineage share Parquet but publish two manifests."""
    monkeypatch.chdir(tmp_path)
    store = DataStore(DataSettings())
    spec = spec_for(Dataset.PRICES)
    calendar = load_calendar("XNYS")

    first = _publish_prices(store, spec, calendar, _payload_at(b"prices-v1", _RETRIEVED_AT))
    second = _publish_prices(
        store, spec, calendar, _payload_at(b"prices-v2", _RETRIEVED_AT + timedelta(hours=1))
    )
    assert first.normalized_path == second.normalized_path
    assert first.manifest_path != second.manifest_path
    assert first.manifest_path.is_file()
    assert second.manifest_path.is_file()
    assert store.read_normalized(first, spec).equals(store.read_normalized(second, spec))


def test_write_normalized_retry_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Repeating identical frame and lineage keeps paths and manifest bytes identical."""
    monkeypatch.chdir(tmp_path)
    store = DataStore(DataSettings())
    spec = spec_for(Dataset.PRICES)
    calendar = load_calendar("XNYS")

    payload = _payload(b"prices-payload")
    first = _publish_prices(store, spec, calendar, payload)
    second = _publish_prices(store, spec, calendar, payload)
    assert first.normalized_path == second.normalized_path
    assert first.manifest_path == second.manifest_path
    assert first.manifest_path.read_bytes() == second.manifest_path.read_bytes()
    assert first.manifest_path.name == f"{canonical_manifest_sha256(first.manifest)}.json"
    assert store.read_normalized(second, spec).equals(store.read_normalized(first, spec))


def test_write_normalized_rejects_corrupt_parquet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A conflicting Parquet at the content address fails publication without repair."""
    monkeypatch.chdir(tmp_path)
    store = DataStore(DataSettings())
    spec = spec_for(Dataset.PRICES)
    calendar = load_calendar("XNYS")

    payload = _payload(b"prices-payload")
    artifact = _publish_prices(store, spec, calendar, payload)
    tampered = pl.read_parquet(artifact.normalized_path).with_columns(pl.col("close") + 1.0)
    tampered.write_parquet(artifact.normalized_path)

    frame = _stamped_prices(spec, calendar)
    raw_artifact = store.store_raw(Dataset.PRICES, payload)
    report = validate_frame(frame, spec, calendar)
    with pytest.raises(UntrustedDatasetError):
        store.write_normalized(frame, spec, raw_artifact, payload, report)


def test_write_normalized_rejects_conflicting_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A conflicting document at the manifest address fails publication."""
    monkeypatch.chdir(tmp_path)
    store = DataStore(DataSettings())
    spec = spec_for(Dataset.PRICES)
    calendar = load_calendar("XNYS")

    payload = _payload(b"prices-payload")
    artifact = _publish_prices(store, spec, calendar, payload)
    artifact.manifest_path.write_text('{"dataset": "tampered"}', encoding="utf-8")

    frame = _stamped_prices(spec, calendar)
    raw_artifact = store.store_raw(Dataset.PRICES, payload)
    report = validate_frame(frame, spec, calendar)
    with pytest.raises(UntrustedDatasetError):
        store.write_normalized(frame, spec, raw_artifact, payload, report)


def test_read_normalized_rejects_corrupt_raw_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Modified or missing archived raw bytes fail the read."""
    monkeypatch.chdir(tmp_path)
    store = DataStore(DataSettings())
    spec = spec_for(Dataset.PRICES)
    calendar = load_calendar("XNYS")

    artifact = _publish_prices(store, spec, calendar, _payload(b"prices-payload"))
    raw_path = tmp_path / "data" / Path(*artifact.manifest.raw_artifact.relative_path.parts)
    assert raw_path.is_file()

    raw_path.write_bytes(b"tampered-bytes")
    with pytest.raises(UntrustedDatasetError):
        store.read_normalized(artifact, spec)

    raw_path.unlink()
    with pytest.raises(UntrustedDatasetError):
        store.read_normalized(artifact, spec)


def test_read_normalized_rejects_raw_path_escape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A recorded raw path outside the data root fails the read."""
    monkeypatch.chdir(tmp_path)
    store = DataStore(DataSettings())
    spec = spec_for(Dataset.PRICES)
    calendar = load_calendar("XNYS")

    artifact = _publish_prices(store, spec, calendar, _payload(b"prices-payload"))
    evil_manifest = replace(
        artifact.manifest,
        raw_artifact=RawArtifact(
            relative_path=PurePosixPath("../evil"),
            sha256=artifact.manifest.raw_artifact.sha256,
            retrieved_at=artifact.manifest.raw_artifact.retrieved_at,
        ),
    )
    document = json.loads(artifact.manifest_path.read_text(encoding="utf-8"))
    document["raw_artifact"]["relative_path"] = "../evil"
    evil_bytes = json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")
    evil_path = artifact.manifest_path.parent / f"{hashlib.sha256(evil_bytes).hexdigest()}.json"
    evil_path.write_bytes(evil_bytes)
    evil_artifact = DatasetArtifact(
        normalized_path=artifact.normalized_path,
        manifest_path=evil_path,
        manifest=evil_manifest,
    )
    with pytest.raises(UntrustedDatasetError):
        store.read_normalized(evil_artifact, spec)


def test_read_normalized_rejects_renamed_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A new-style manifest whose content no longer matches its filename fails closed."""
    monkeypatch.chdir(tmp_path)
    store = DataStore(DataSettings())
    spec = spec_for(Dataset.PRICES)
    calendar = load_calendar("XNYS")

    artifact = _publish_prices(store, spec, calendar, _payload(b"prices-payload"))
    doctored_manifest = replace(
        artifact.manifest, retrieved_at=_RETRIEVED_AT + timedelta(days=1)
    )
    document = json.loads(artifact.manifest_path.read_text(encoding="utf-8"))
    document["retrieved_at"] = (_RETRIEVED_AT + timedelta(days=1)).isoformat()
    artifact.manifest_path.write_bytes(
        json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    doctored_artifact = DatasetArtifact(
        normalized_path=artifact.normalized_path,
        manifest_path=artifact.manifest_path,
        manifest=doctored_manifest,
    )
    with pytest.raises(UntrustedDatasetError, match=r"filename hash"):
        store.read_normalized(doctored_artifact, spec)


def test_read_normalized_rejects_lineage_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An artifact whose lineage differs from the stored document fails closed."""
    monkeypatch.chdir(tmp_path)
    store = DataStore(DataSettings())
    spec = spec_for(Dataset.PRICES)
    calendar = load_calendar("XNYS")

    artifact = _publish_prices(store, spec, calendar, _payload(b"prices-payload"))
    impostor = DatasetArtifact(
        normalized_path=artifact.normalized_path,
        manifest_path=artifact.manifest_path,
        manifest=replace(artifact.manifest, provider="impostor"),
    )
    with pytest.raises(UntrustedDatasetError):
        store.read_normalized(impostor, spec)


def test_read_normalized_keeps_legacy_manifest_readable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A previously published frame-addressed manifest still verifies."""
    monkeypatch.chdir(tmp_path)
    store = DataStore(DataSettings())
    spec = spec_for(Dataset.PRICES)
    calendar = load_calendar("XNYS")

    artifact = _publish_prices(store, spec, calendar, _payload(b"prices-payload"))
    legacy_path = artifact.manifest_path.parent / f"{artifact.manifest.normalized_sha256}.json"
    legacy_path.write_bytes(artifact.manifest_path.read_bytes())
    legacy_artifact = DatasetArtifact(
        normalized_path=artifact.normalized_path,
        manifest_path=legacy_path,
        manifest=artifact.manifest,
    )
    assert store.read_normalized(legacy_artifact, spec).equals(
        store.read_normalized(artifact, spec)
    )


def test_write_normalized_records_prior_manifest_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A write with prior_manifest_sha256 round-trips and changes manifest identity."""
    import src.data.catalog as catalog_module

    monkeypatch.chdir(tmp_path)
    settings = DataSettings()
    store = DataStore(settings)
    spec = spec_for(Dataset.PRICES)
    calendar = load_calendar("XNYS")

    frame = _stamped_prices(spec, calendar)
    payload = _payload(b"prices-payload")
    raw_artifact = store.store_raw(Dataset.PRICES, payload)
    report = validate_frame(frame, spec, calendar)
    plain = store.write_normalized(frame, spec, raw_artifact, payload, report)
    assert plain.manifest.prior_manifest_sha256 is None

    prior_sha = "ab" * 32
    linked = store.write_normalized(frame, spec, raw_artifact, payload, report, "1", prior_sha)
    assert linked.manifest.prior_manifest_sha256 == prior_sha
    assert linked.manifest_path.name != plain.manifest_path.name
    assert linked.manifest_path.is_file()

    reloaded = catalog_module.latest_artifact(settings, Dataset.PRICES)
    assert reloaded.manifest.prior_manifest_sha256 in (None, prior_sha)


def test_write_normalized_legacy_manifest_hash_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A manifest written without the field keeps canonical hash equal to filename."""
    monkeypatch.chdir(tmp_path)
    store = DataStore(DataSettings())
    spec = spec_for(Dataset.PRICES)
    calendar = load_calendar("XNYS")

    artifact = _publish_prices(store, spec, calendar, _payload(b"prices-payload"))
    assert artifact.manifest.prior_manifest_sha256 is None
    assert artifact.manifest_path.name == f"{canonical_manifest_sha256(artifact.manifest)}.json"
    document = json.loads(artifact.manifest_path.read_text(encoding="utf-8"))
    assert "prior_manifest_sha256" not in document


def test_write_normalized_rejects_malformed_prior_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-64-hex prior hash raises ValueError without creating files."""
    monkeypatch.chdir(tmp_path)
    settings = DataSettings()
    store = DataStore(settings)
    spec = spec_for(Dataset.PRICES)
    calendar = load_calendar("XNYS")

    frame = _stamped_prices(spec, calendar)
    payload = _payload(b"prices-payload")
    raw_artifact = store.store_raw(Dataset.PRICES, payload)
    report = validate_frame(frame, spec, calendar)
    manifests_dir = tmp_path / "data" / "manifests" / "prices"
    before = sorted(manifests_dir.glob("*.json")) if manifests_dir.is_dir() else []
    with pytest.raises(ValueError, match="prior_manifest_sha256"):
        store.write_normalized(frame, spec, raw_artifact, payload, report, "1", "not-hex")
    assert (sorted(manifests_dir.glob("*.json")) if manifests_dir.is_dir() else []) == before


def test_catalog_rejects_malformed_prior_hash_in_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A present but malformed prior_manifest_sha256 fails manifest reconstruction."""
    import src.data.catalog as catalog_module

    monkeypatch.chdir(tmp_path)
    settings = DataSettings()
    store = DataStore(settings)
    spec = spec_for(Dataset.PRICES)
    calendar = load_calendar("XNYS")

    artifact = _publish_prices(store, spec, calendar, _payload(b"prices-payload"))
    document = json.loads(artifact.manifest_path.read_text(encoding="utf-8"))
    document["prior_manifest_sha256"] = "bad-value"
    artifact.manifest_path.write_bytes(
        json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    catalog_module.clear_catalog_frame_cache()
    with pytest.raises(UntrustedDatasetError, match="manifest malformed"):
        catalog_module.latest_artifact(settings, Dataset.PRICES)
