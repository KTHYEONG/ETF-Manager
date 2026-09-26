"""Invariant guards for the per-experiment result store."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from src.data.result_store import (
    ResultKind,
    normalize_result_slug,
    read_run_ledger,
    record_run,
    result_ref,
    write_result,
)
from src.data.settings import DataSettings


def _settings(tmp_path: Path) -> DataSettings:
    return DataSettings(data_root=tmp_path / "data")


def test_write_result_layout_per_experiment(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    ref = write_result(
        settings,
        experiment="wf_qqq_adaptive_v5",
        kind=ResultKind.WALK_FORWARD,
        run_id="893bec335e979f14",
        payload={"a": 1},
    )
    expected = tmp_path / "data" / "runs" / "wf_qqq_adaptive_v5" / "walk_forward_893bec335e979f14.json"
    assert ref.json_path == expected
    assert expected.read_text(encoding="utf-8") == json.dumps({"a": 1}, indent=2)
    assert not ref.markdown_path.exists()


def test_write_result_markdown_sidecar_shares_stem(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    ref = write_result(
        settings,
        experiment="wf_qqq_adaptive_v5",
        kind=ResultKind.WALK_FORWARD,
        run_id="893bec335e979f14",
        payload={"a": 1},
        markdown="# x\n",
    )
    assert ref.markdown_path == ref.json_path.with_suffix(".md")
    assert ref.markdown_path.read_text(encoding="utf-8") == "# x\n"


def test_write_result_case_folding_merges_campaign(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    ref = write_result(
        settings,
        experiment="FINAL_HISTORICAL_CAMPAIGN_V1",
        kind=ResultKind.FINAL_HISTORICAL,
        run_id="abc123",
        payload={},
    )
    assert ref.json_path.parent.name == "final_historical_campaign_v1"


def test_result_ref_sanitizes_timestamp_run_id(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    ref = result_ref(
        settings,
        experiment="thesis_wave",
        kind=ResultKind.THESIS_WAVE,
        run_id="2026-08-28T00:00:00+00:00",
    )
    assert ref.json_path.name == "thesis_wave_2026-08-28t00-00-00-00-00.json"
    slug = normalize_result_slug("2026-08-28T00:00:00+00:00")
    assert normalize_result_slug(slug) == slug


def test_write_result_rejects_reserved_slug(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    for bad in ("", "_legacy", "../x"):
        with pytest.raises(ValueError, match="invalid result slug"):
            write_result(
                settings,
                experiment=bad,
                kind=ResultKind.WALK_FORWARD,
                run_id="abc123",
                payload={},
            )
    assert not (tmp_path / "data" / "runs").exists() or list(
        (tmp_path / "data" / "runs").rglob("*.json")
    ) == []


def test_write_result_idempotent_overwrite_with_growing_ledger(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    t1 = datetime(2026, 1, 1, tzinfo=UTC)
    t2 = datetime(2026, 1, 2, tzinfo=UTC)
    kwargs = {
        "experiment": "wf_qqq_adaptive_v5",
        "kind": ResultKind.WALK_FORWARD,
        "run_id": "893bec335e979f14",
        "payload": {"a": 1},
    }
    first = write_result(settings, written_at=t1, **kwargs)  # type: ignore[arg-type]
    second = write_result(settings, written_at=t2, **kwargs)  # type: ignore[arg-type]
    assert first.json_path == second.json_path
    assert len(list(first.json_path.parent.glob("*.json"))) == 1
    entries = read_run_ledger(settings, "wf_qqq_adaptive_v5")
    assert len(entries) == 2
    assert entries[0].written_at == t1
    assert entries[1].written_at == t2


def test_write_result_rejects_naive_timestamp(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    with pytest.raises(ValueError, match="timezone-aware"):
        write_result(
            settings,
            experiment="wf_qqq_adaptive_v5",
            kind=ResultKind.WALK_FORWARD,
            run_id="abc123",
            payload={},
            written_at=datetime(2026, 1, 1),
        )
    assert not (tmp_path / "data" / "runs" / "wf_qqq_adaptive_v5" / "runs.jsonl").exists()


def test_read_run_ledger_fails_closed_on_corruption(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    ledger = tmp_path / "data" / "runs" / "wf_qqq_adaptive_v5" / "runs.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text("not json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="corrupted run ledger"):
        read_run_ledger(settings, "wf_qqq_adaptive_v5")


def test_record_run_rejects_naive_timestamp(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    ref = result_ref(
        settings,
        experiment="wf_qqq_adaptive_v5",
        kind=ResultKind.WALK_FORWARD,
        run_id="abc123",
    )
    with pytest.raises(ValueError, match="timezone-aware"):
        record_run(settings, ref, written_at=datetime(2026, 1, 1))


def test_read_run_ledger_rejects_non_object_line(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    ledger = tmp_path / "data" / "runs" / "wf_qqq_adaptive_v5" / "runs.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text("[1, 2]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="corrupted run ledger"):
        read_run_ledger(settings, "wf_qqq_adaptive_v5")


def test_read_run_ledger_rejects_unknown_kind(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    ledger = tmp_path / "data" / "runs" / "wf_qqq_adaptive_v5" / "runs.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text(
        json.dumps({"experiment": "wf_qqq_adaptive_v5", "kind": "nope", "run_id": "a", "written_at": "2026-01-01T00:00:00+00:00"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="corrupted run ledger"):
        read_run_ledger(settings, "wf_qqq_adaptive_v5")


def test_read_run_ledger_rejects_bad_timestamp(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    ledger = tmp_path / "data" / "runs" / "wf_qqq_adaptive_v5" / "runs.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text(
        json.dumps({"experiment": "wf_qqq_adaptive_v5", "kind": "walk_forward", "run_id": "a", "written_at": "not-a-time"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="corrupted run ledger"):
        read_run_ledger(settings, "wf_qqq_adaptive_v5")


def test_read_run_ledger_rejects_missing_keys(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    ledger = tmp_path / "data" / "runs" / "wf_qqq_adaptive_v5" / "runs.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text(
        "\n" + json.dumps({"experiment": "wf_qqq_adaptive_v5", "kind": "walk_forward", "written_at": "2026-01-01T00:00:00+00:00"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="corrupted run ledger"):
        read_run_ledger(settings, "wf_qqq_adaptive_v5")


def test_read_run_ledger_missing_is_empty(tmp_path: Path) -> None:
    assert read_run_ledger(_settings(tmp_path), "wf_qqq_adaptive_v5") == ()


def test_write_result_leaves_stale_markdown_untouched(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    first = write_result(
        settings,
        experiment="after_tax_smoke",
        kind=ResultKind.AFTER_TAX,
        run_id="abc123",
        payload={"a": 1},
        markdown="# original\n",
    )
    second = write_result(
        settings,
        experiment="after_tax_smoke",
        kind=ResultKind.AFTER_TAX,
        run_id="abc123",
        payload={"a": 2},
    )
    assert second.markdown_path == first.markdown_path
    assert first.markdown_path.read_text(encoding="utf-8") == "# original\n"
