"""Invariant guards for the shared incremental merge layer."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl
import pytest

from unittest.mock import MagicMock

from src.data.merge import (
    PartitionKeyLossError,
    PriorPartition,
    PriorPartitionUntrustedError,
    RecoveryPlan,
    apply_history_recovery,
    assert_key_coverage,
    load_prior_partition,
    merge_incremental,
    plan_history_recovery,
)
from src.data.pipeline import persist_ingest
from src.data.schema import Dataset, spec_for
from src.data.settings import DataSettings
from src.data.storage import RawPayload, UntrustedDatasetError, canonical_frame_sha256

_RETRIEVED = datetime(2024, 2, 1, 5, 0, tzinfo=UTC)


def _settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> DataSettings:
    monkeypatch.chdir(tmp_path)
    return DataSettings(data_root="data")


def _rates_frame(rows: list[tuple[str, date, float]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "series_id": [r[0] for r in rows],
            "observation_date": [r[1] for r in rows],
            "value": [r[2] for r in rows],
            "source": ["fred"] * len(rows),
            "retrieved_at": [_RETRIEVED] * len(rows),
        },
        schema=dict(spec_for(Dataset.RATES).columns),
    )


def _payload(content: bytes = b"merge-payload") -> RawPayload:
    return RawPayload(
        provider="synthetic",
        endpoint="merge/test",
        request_params={"format": "json"},
        retrieved_at=_RETRIEVED,
        extension="json",
        content=content,
    )


def _persist_rates(settings: DataSettings, frame: pl.DataFrame) -> None:
    persist_ingest(frame, Dataset.RATES, _payload(), settings)


def _prior_of(frame: pl.DataFrame) -> PriorPartition:
    return PriorPartition(artifact=MagicMock(), frame=frame)


def test_first_ingest_returns_sorted_incoming(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    settings = _settings(monkeypatch, tmp_path)
    assert load_prior_partition(settings, Dataset.RATES) is None
    incoming = _rates_frame([("B", date(2024, 1, 3), 2.0), ("A", date(2024, 1, 2), 1.0)])
    result = merge_incremental(None, incoming, Dataset.RATES)
    assert result.get_column("series_id").to_list() == ["A", "B"]
    assert list(result.columns) == list(spec_for(Dataset.RATES).columns.keys())


def test_unreadable_prior_aborts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    settings = _settings(monkeypatch, tmp_path)
    _persist_rates(settings, _rates_frame([("X", date(2024, 1, 2), 5.0)]))
    assert issubclass(PriorPartitionUntrustedError, UntrustedDatasetError)
    from src.data.catalog import latest_artifact

    artifact = latest_artifact(settings, Dataset.RATES)
    artifact.normalized_path.unlink()
    with pytest.raises(PriorPartitionUntrustedError):
        load_prior_partition(settings, Dataset.RATES)


def test_incoming_wins_on_key_collision() -> None:
    spec = spec_for(Dataset.RATES)
    prior = _prior_of(_rates_frame([("X", date(2024, 1, 2), 5.0)]))
    incoming = _rates_frame([("X", date(2024, 1, 2), 9.5)])
    result = merge_incremental(prior, incoming, Dataset.RATES)
    assert result.height == 1
    assert result.get_column("value").to_list() == [9.5]
    assert list(result.columns) == list(spec.columns.keys())


def test_prior_keys_preserved_without_refreshed() -> None:
    prior = _prior_of(_rates_frame([("X", date(2024, 1, 2), 5.0), ("Y", date(2024, 1, 2), 6.0)]))
    incoming = _rates_frame([("X", date(2024, 1, 3), 5.1)])
    result = merge_incremental(prior, incoming, Dataset.RATES)
    keys = set(zip(result.get_column("series_id").to_list(), result.get_column("observation_date").to_list(), strict=True))
    assert ("Y", date(2024, 1, 2)) in keys
    assert ("X", date(2024, 1, 2)) in keys
    assert ("X", date(2024, 1, 3)) in keys


def test_refreshed_keys_replaced() -> None:
    prior = _prior_of(_rates_frame([("X", date(2024, 1, 2), 5.0), ("Y", date(2024, 1, 2), 6.0)]))
    refreshed = pl.DataFrame({"series_id": ["X"]}, schema={"series_id": pl.String})
    incoming = _rates_frame([("X", date(2024, 1, 3), 7.0)])
    result = merge_incremental(prior, incoming, Dataset.RATES, refreshed=refreshed)
    keys = set(zip(result.get_column("series_id").to_list(), result.get_column("observation_date").to_list(), strict=True))
    assert ("X", date(2024, 1, 2)) not in keys
    assert ("X", date(2024, 1, 3)) in keys
    assert ("Y", date(2024, 1, 2)) in keys


def test_deterministic_output_for_shuffled_inputs() -> None:
    prior = _prior_of(_rates_frame([("X", date(2024, 1, 2), 5.0), ("Y", date(2024, 1, 2), 6.0)]))
    incoming = _rates_frame([("Z", date(2024, 1, 4), 1.0), ("X", date(2024, 1, 3), 5.1)])
    first = merge_incremental(prior, incoming, Dataset.RATES)
    second = merge_incremental(prior, incoming.reverse(), Dataset.RATES)
    assert canonical_frame_sha256(first, spec_for(Dataset.RATES)) == canonical_frame_sha256(
        second, spec_for(Dataset.RATES)
    )
    assert first.equals(second)


def test_key_loss_detected_and_nothing_written(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    settings = _settings(monkeypatch, tmp_path)
    _persist_rates(settings, _rates_frame([("X", date(2024, 1, 2), 5.0), ("Y", date(2024, 1, 2), 6.0)]))
    prior = load_prior_partition(settings, Dataset.RATES)
    assert prior is not None
    manifests_dir = tmp_path / "data" / "manifests" / "rates"
    before = sorted(manifests_dir.glob("*.json"))
    dropped = _rates_frame([("X", date(2024, 1, 2), 5.0)])
    with pytest.raises(PartitionKeyLossError):
        assert_key_coverage(prior, dropped, Dataset.RATES)
    with pytest.raises(PartitionKeyLossError):
        persist_ingest(dropped, Dataset.RATES, _payload(b"bad-merge"), settings, prior=prior)
    assert sorted(manifests_dir.glob("*.json")) == before


def test_fully_refreshed_prior_needs_no_coverage() -> None:
    prior = _prior_of(_rates_frame([("X", date(2024, 1, 2), 5.0)]))
    refreshed = pl.DataFrame({"series_id": ["X"]}, schema={"series_id": pl.String})
    incoming = _rates_frame([("Y", date(2024, 1, 3), 1.0)])
    assert_key_coverage(prior, incoming, Dataset.RATES, refreshed=refreshed)
    result = merge_incremental(prior, incoming, Dataset.RATES, refreshed=refreshed)
    assert set(result.get_column("series_id").to_list()) == {"Y"}


def test_refreshed_without_shared_key_column_rejected() -> None:
    prior = _prior_of(_rates_frame([("X", date(2024, 1, 2), 5.0)]))
    incoming = _rates_frame([("X", date(2024, 1, 3), 5.1)])
    bad = pl.DataFrame({"unknown_col": ["X"]}, schema={"unknown_col": pl.String})
    with pytest.raises(ValueError, match="at least one key column"):
        merge_incremental(prior, incoming, Dataset.RATES, refreshed=bad)
    with pytest.raises(ValueError, match="at least one key column"):
        assert_key_coverage(prior, incoming, Dataset.RATES, refreshed=bad)


def test_refreshed_non_key_column_rejected() -> None:
    prior = _prior_of(_rates_frame([("X", date(2024, 1, 2), 5.0)]))
    incoming = _rates_frame([("X", date(2024, 1, 3), 5.1)])
    bad = pl.DataFrame(
        {"series_id": ["X"], "value": [5.0]},
        schema={"series_id": pl.String, "value": pl.Float64},
    )
    with pytest.raises(ValueError, match="not part of dataset key"):
        merge_incremental(prior, incoming, Dataset.RATES, refreshed=bad)


def _payload_at(content: bytes, retrieved_at: datetime) -> RawPayload:
    return RawPayload(
        provider="synthetic",
        endpoint="merge/test",
        request_params={"format": "json"},
        retrieved_at=retrieved_at,
        extension="json",
        content=content,
    )


def _seed_rates(settings: DataSettings, rows: list[tuple[str, date, float]], retrieved_at: datetime, content: bytes) -> None:
    persist_ingest(_rates_frame(rows), Dataset.RATES, _payload_at(content, retrieved_at), settings)


def _ten_days(first: int = 1, value: float = 1.0) -> list[tuple[str, date, float]]:
    return [("X", date(2024, 1, day), value + day) for day in range(first, 11)]


def test_history_recovery_restores_truncated_latest(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from src.data.catalog import latest_artifact

    settings = _settings(monkeypatch, tmp_path)
    _seed_rates(settings, _ten_days(), datetime(2024, 3, 1, tzinfo=UTC), b"history-full")
    _seed_rates(
        settings,
        [("X", date(2024, 1, day), 100.0 + day) for day in range(6, 9)],
        datetime(2024, 4, 1, tzinfo=UTC),
        b"history-truncated",
    )
    plan = plan_history_recovery(settings, Dataset.RATES)
    assert isinstance(plan, RecoveryPlan)
    assert plan.latest_rows == 3
    assert plan.recovered_rows == 10
    assert plan.latest_manifest_sha256 == latest_artifact(settings, Dataset.RATES).manifest_path.stem
    assert len(plan.source_manifest_sha256s) == 2

    before_manifests = set((tmp_path / "data" / "manifests" / "rates").glob("*.json"))
    before_parquets = set((tmp_path / "data" / "normalized" / "rates").glob("*.parquet"))
    artifact = apply_history_recovery(plan, settings)
    assert artifact.manifest.prior_manifest_sha256 == plan.latest_manifest_sha256
    assert artifact.manifest.row_count == 10
    after_manifests = set((tmp_path / "data" / "manifests" / "rates").glob("*.json"))
    assert before_manifests < after_manifests
    assert len(after_manifests) == len(before_manifests) + 1
    for path in before_manifests:
        assert path.is_file()
    for path in before_parquets:
        assert path.is_file()


def test_history_recovery_newest_value_wins(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    settings = _settings(monkeypatch, tmp_path)
    _seed_rates(settings, [("X", date(2024, 1, 6), 5.0)], datetime(2024, 3, 1, tzinfo=UTC), b"old-value")
    _seed_rates(settings, [("X", date(2024, 1, 6), 9.0)], datetime(2024, 4, 1, tzinfo=UTC), b"new-value")
    _seed_rates(
        settings,
        [("X", date(2024, 1, day), 1.0) for day in range(1, 6)],
        datetime(2024, 2, 1, tzinfo=UTC),
        b"older-wider",
    )
    plan = plan_history_recovery(settings, Dataset.RATES)
    assert plan is not None
    row = plan.recovered.filter(
        (pl.col("series_id") == "X") & (pl.col("observation_date") == date(2024, 1, 6))
    )
    assert row.height == 1
    assert row.get_column("value").to_list() == [9.0]


def test_history_recovery_nothing_to_recover(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    settings = _settings(monkeypatch, tmp_path)
    _seed_rates(
        settings,
        [("X", date(2024, 1, day), float(day)) for day in range(1, 4)],
        datetime(2024, 3, 1, tzinfo=UTC),
        b"subset",
    )
    _seed_rates(settings, _ten_days(), datetime(2024, 4, 1, tzinfo=UTC), b"superset")
    assert plan_history_recovery(settings, Dataset.RATES) is None


def test_history_recovery_ignores_untrusted_partition(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    import json as _json

    settings = _settings(monkeypatch, tmp_path)
    _seed_rates(settings, _ten_days(), datetime(2024, 3, 1, tzinfo=UTC), b"history-full")
    _seed_rates(
        settings,
        [("X", date(2024, 1, day), 100.0 + day) for day in range(6, 9)],
        datetime(2024, 4, 1, tzinfo=UTC),
        b"history-truncated",
    )
    data_root = tmp_path / "data"
    for manifest_path in sorted((data_root / "manifests" / "rates").glob("*.json")):
        document = _json.loads(manifest_path.read_text(encoding="utf-8"))
        if document.get("row_count") == 10:
            (data_root / str(document["normalized_relative_path"])).unlink()
            break
    with caplog.at_level("WARNING"):
        assert plan_history_recovery(settings, Dataset.RATES) is None
    assert any(
        "recover_skip_untrusted" in record.message and ".json" in record.message for record in caplog.records
    )


def test_history_recovery_rejects_concurrent_change(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    settings = _settings(monkeypatch, tmp_path)
    _seed_rates(settings, _ten_days(), datetime(2024, 3, 1, tzinfo=UTC), b"history-full")
    _seed_rates(
        settings,
        [("X", date(2024, 1, day), 100.0 + day) for day in range(6, 9)],
        datetime(2024, 4, 1, tzinfo=UTC),
        b"history-truncated",
    )
    plan = plan_history_recovery(settings, Dataset.RATES)
    assert plan is not None
    _seed_rates(
        settings,
        [("X", date(2024, 1, day), 200.0 + day) for day in range(6, 9)],
        datetime(2024, 5, 1, tzinfo=UTC),
        b"history-concurrent",
    )
    manifests_dir = tmp_path / "data" / "manifests" / "rates"
    before = sorted(manifests_dir.glob("*.json"))
    with pytest.raises(UntrustedDatasetError, match="changed since planning"):
        apply_history_recovery(plan, settings)
    assert sorted(manifests_dir.glob("*.json")) == before


def test_history_recovery_becomes_catalog_view(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from src.data.catalog import latest_artifact

    settings = _settings(monkeypatch, tmp_path)
    _seed_rates(settings, _ten_days(), datetime(2024, 3, 1, tzinfo=UTC), b"history-full")
    _seed_rates(
        settings,
        [("X", date(2024, 1, day), 100.0 + day) for day in range(6, 9)],
        datetime(2024, 4, 1, tzinfo=UTC),
        b"history-truncated",
    )
    plan = plan_history_recovery(settings, Dataset.RATES)
    assert plan is not None
    apply_history_recovery(plan, settings)
    selected = latest_artifact(settings, Dataset.RATES)
    assert selected.manifest.row_count == plan.recovered_rows


def test_history_recovery_empty_dataset_returns_none(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    settings = _settings(monkeypatch, tmp_path)
    assert plan_history_recovery(settings, Dataset.RATES) is None


def test_history_recovery_all_untrusted_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import json as _json

    settings = _settings(monkeypatch, tmp_path)
    _seed_rates(settings, _ten_days(), datetime(2024, 3, 1, tzinfo=UTC), b"history-full")
    data_root = tmp_path / "data"
    for manifest_path in sorted((data_root / "manifests" / "rates").glob("*.json")):
        document = _json.loads(manifest_path.read_text(encoding="utf-8"))
        (data_root / str(document["normalized_relative_path"])).unlink()
    with pytest.raises(PriorPartitionUntrustedError):
        plan_history_recovery(settings, Dataset.RATES)


def _prices_frame(rows: list[tuple[str, date, float]]) -> pl.DataFrame:
    spec = spec_for(Dataset.PRICES)
    return pl.DataFrame(
        {
            "ticker": [row[0] for row in rows],
            "date": [row[1] for row in rows],
            "open": [row[2] for row in rows],
            "high": [row[2] for row in rows],
            "low": [row[2] for row in rows],
            "close": [row[2] for row in rows],
            "volume": [10_000] * len(rows),
            "adjusted_close": [row[2] for row in rows],
            "dividend": [0.0] * len(rows),
            "split_factor": [1.0] * len(rows),
            "source": ["synthetic"] * len(rows),
            "retrieved_at": [_RETRIEVED] * len(rows),
        },
        schema=dict(spec.columns),
    )


def test_history_recovery_session_close_dataset(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from src.data.catalog import latest_artifact

    settings = _settings(monkeypatch, tmp_path)
    persist_ingest(
        _prices_frame([("A", date(2024, 1, 30), 100.0), ("A", date(2024, 1, 31), 101.0)]),
        Dataset.PRICES,
        _payload_at(b"prices-full", datetime(2024, 3, 1, tzinfo=UTC)),
        settings,
    )
    persist_ingest(
        _prices_frame([("A", date(2024, 1, 31), 105.0)]),
        Dataset.PRICES,
        _payload_at(b"prices-truncated", datetime(2024, 4, 1, tzinfo=UTC)),
        settings,
    )
    plan = plan_history_recovery(settings, Dataset.PRICES)
    assert plan is not None
    assert plan.latest_rows == 1
    assert plan.recovered_rows == 2
    artifact = apply_history_recovery(plan, settings)
    assert artifact.manifest.row_count == 2
    assert latest_artifact(settings, Dataset.PRICES).manifest.row_count == 2
