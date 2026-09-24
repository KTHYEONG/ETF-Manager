"""Invariant guards for result inspection, promotion, pruning, and migration."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pytest

from src.data.result_maintenance import (
    apply_result_migration,
    apply_result_prune,
    list_results,
    plan_result_migration,
    plan_result_prune,
    promote_result,
)
from src.data.result_store import ResultKind, read_run_ledger, write_result
from src.data.settings import DataSettings

T1 = datetime(2026, 1, 1, tzinfo=UTC)
T2 = datetime(2026, 1, 2, tzinfo=UTC)
T3 = datetime(2026, 1, 3, tzinfo=UTC)


def _settings(tmp_path: Path) -> DataSettings:
    return DataSettings(data_root=tmp_path / "data")


def _write(settings: DataSettings, exp: str, kind: ResultKind, run: str, at: datetime, md: str | None = None):  # type: ignore[no-untyped-def]
    return write_result(settings, experiment=exp, kind=kind, run_id=run, payload={"run": run}, markdown=md, written_at=at)


def test_list_results_reports_latest_and_run_count(tmp_path: Path) -> None:
    """List reports latest and run count, newest run first."""
    settings = _settings(tmp_path)
    _write(settings, "exp", ResultKind.WALK_FORWARD, "a", T1)
    _write(settings, "exp", ResultKind.WALK_FORWARD, "b", T2)
    _write(settings, "exp", ResultKind.WALK_FORWARD, "a", T3)

    rows = list_results(settings)

    assert [(r.run_id, r.run_count, r.last_written_at) for r in rows] == [("a", 2, T3), ("b", 1, T2)]
    assert rows[0].experiment == "exp"
    assert rows[0].kind is ResultKind.WALK_FORWARD
    assert rows[0].json_path.is_file()
    assert rows[0].has_markdown is False


def test_list_results_skips_removed_files(tmp_path: Path) -> None:
    """List skips runs whose file was removed."""
    settings = _settings(tmp_path)
    ref = _write(settings, "exp", ResultKind.WALK_FORWARD, "a", T1)
    ref.json_path.unlink()

    assert list_results(settings) == ()


def test_list_results_empty_root(tmp_path: Path) -> None:
    """List on a fresh root is empty."""
    assert list_results(_settings(tmp_path)) == ()


def test_promote_result_copies_json_and_markdown(tmp_path: Path) -> None:
    """Promote copies json and markdown with identical bytes; source untouched."""
    settings = _settings(tmp_path)
    ref = _write(settings, "exp", ResultKind.WALK_FORWARD, "a", T1, md="# note\n")
    dest_root = tmp_path / "docs" / "results"

    written = promote_result(settings, experiment="exp", kind=ResultKind.WALK_FORWARD, run_id="a", dest_root=dest_root)

    assert written == (
        dest_root / "exp" / "walk_forward_a.json",
        dest_root / "exp" / "walk_forward_a.md",
    )
    assert written[0].read_bytes() == ref.json_path.read_bytes()
    assert written[1].read_text(encoding="utf-8") == "# note\n"
    assert ref.json_path.is_file()


def test_promote_result_without_sidecar(tmp_path: Path) -> None:
    """Promote without a markdown sidecar returns only the JSON path."""
    settings = _settings(tmp_path)
    _write(settings, "exp", ResultKind.WALK_FORWARD, "a", T1)
    dest_root = tmp_path / "docs" / "results"

    assert promote_result(settings, experiment="exp", kind=ResultKind.WALK_FORWARD, run_id="a", dest_root=dest_root) == (
        dest_root / "exp" / "walk_forward_a.json",
    )


def test_promote_result_is_idempotent(tmp_path: Path) -> None:
    """Promote is idempotent."""
    settings = _settings(tmp_path)
    _write(settings, "exp", ResultKind.WALK_FORWARD, "a", T1, md="# note\n")
    dest_root = tmp_path / "docs" / "results"
    first = promote_result(settings, experiment="exp", kind=ResultKind.WALK_FORWARD, run_id="a", dest_root=dest_root)

    assert promote_result(settings, experiment="exp", kind=ResultKind.WALK_FORWARD, run_id="a", dest_root=dest_root) == first


def test_promote_result_refuses_divergent_overwrite(tmp_path: Path) -> None:
    """Promote refuses divergent overwrite."""
    settings = _settings(tmp_path)
    _write(settings, "exp", ResultKind.WALK_FORWARD, "a", T1)
    dest = tmp_path / "docs" / "results" / "exp" / "walk_forward_a.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("{}", encoding="utf-8")

    with pytest.raises(FileExistsError, match="refusing overwrite"):
        promote_result(settings, experiment="exp", kind=ResultKind.WALK_FORWARD, run_id="a", dest_root=tmp_path / "docs" / "results")
    assert dest.read_text(encoding="utf-8") == "{}"


def test_promote_result_missing_source(tmp_path: Path) -> None:
    """Promote missing source raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="no run artifact"):
        promote_result(_settings(tmp_path), experiment="exp", kind=ResultKind.WALK_FORWARD, run_id="nope", dest_root=tmp_path / "docs")


def _prune_fixture(settings: DataSettings) -> None:
    _write(settings, "exp", ResultKind.WALK_FORWARD, "r1", T1, md="# r1\n")
    _write(settings, "exp", ResultKind.WALK_FORWARD, "r2", T2)
    _write(settings, "exp", ResultKind.WALK_FORWARD, "r3", T3)
    _write(settings, "exp", ResultKind.COSTS, "c1", T1)
    _write(settings, "other", ResultKind.WALK_FORWARD, "solo", T1)


def test_plan_result_prune_keeps_recent_per_kind(tmp_path: Path) -> None:
    """Prune keeps N most recent per kind."""
    settings = _settings(tmp_path)
    _prune_fixture(settings)

    plan = plan_result_prune(settings, keep=2)

    assert [(r.experiment, r.kind, r.run_id) for r in plan.to_delete] == [("exp", ResultKind.WALK_FORWARD, "r1")]
    assert plan.keep == 2


def test_apply_result_prune_dry_run_is_inert(tmp_path: Path) -> None:
    """Prune dry run is inert."""
    settings = _settings(tmp_path)
    _prune_fixture(settings)
    root = tmp_path / "data" / "results"
    before_files = sorted(p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file())
    before_ledger = (root / "exp" / "runs.jsonl").read_bytes()

    plan = plan_result_prune(settings, keep=2)
    deleted = apply_result_prune(settings, plan, dry_run=True)

    assert deleted == (root / "exp" / "walk_forward_r1.json", root / "exp" / "walk_forward_r1.md")
    assert sorted(p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()) == before_files
    assert (root / "exp" / "runs.jsonl").read_bytes() == before_ledger


def test_apply_result_prune_rewrites_ledger(tmp_path: Path) -> None:
    """Prune apply rewrites ledger."""
    settings = _settings(tmp_path)
    _prune_fixture(settings)
    root = tmp_path / "data" / "results"

    plan = plan_result_prune(settings, keep=2)
    deleted = apply_result_prune(settings, plan, dry_run=False)

    assert deleted == (root / "exp" / "walk_forward_r1.json", root / "exp" / "walk_forward_r1.md")
    assert not (root / "exp" / "walk_forward_r1.json").exists()
    assert not (root / "exp" / "walk_forward_r1.md").exists()
    remaining = read_run_ledger(settings, "exp")
    assert {e.run_id for e in remaining} == {"r2", "r3", "c1"}
    assert [e.run_id for e in remaining] == ["r2", "r3", "c1"]
    assert [e.run_id for e in read_run_ledger(settings, "other")] == ["solo"]


def test_apply_result_prune_tolerates_missing_files(tmp_path: Path) -> None:
    """Prune apply tolerates planned artifacts that are already gone."""
    from src.data.result_maintenance import ResultPrunePlan
    from src.data.result_store import ResultRef

    settings = _settings(tmp_path)
    _write(settings, "exp", ResultKind.WALK_FORWARD, "a", T1)
    root = tmp_path / "data" / "results"
    ghost = ResultRef(
        experiment="exp",
        kind=ResultKind.WALK_FORWARD,
        run_id="ghost",
        json_path=root / "exp" / "walk_forward_ghost.json",
        markdown_path=root / "exp" / "walk_forward_ghost.md",
    )
    deleted = apply_result_prune(settings, ResultPrunePlan(keep=1, to_delete=(ghost,)), dry_run=False)

    assert deleted == (root / "exp" / "walk_forward_ghost.json",)
    assert [e.run_id for e in read_run_ledger(settings, "exp")] == ["a"]


def test_apply_result_prune_preserves_blank_lines(tmp_path: Path) -> None:
    """Prune apply preserves blank ledger lines."""
    settings = _settings(tmp_path)
    _write(settings, "exp", ResultKind.WALK_FORWARD, "r1", T1)
    _write(settings, "exp", ResultKind.WALK_FORWARD, "r2", T2)
    ledger = tmp_path / "data" / "results" / "exp" / "runs.jsonl"
    ledger.write_text(ledger.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    plan = plan_result_prune(settings, keep=1)
    apply_result_prune(settings, plan, dry_run=False)

    assert [e.run_id for e in read_run_ledger(settings, "exp")] == ["r2"]
    assert ledger.read_text(encoding="utf-8").endswith("\n\n")


def test_apply_result_prune_fails_closed_on_corrupt_ledger(tmp_path: Path) -> None:
    """Prune apply fails closed on a corrupted ledger."""
    from src.data.result_maintenance import ResultPrunePlan
    from src.data.result_store import ResultRef

    settings = _settings(tmp_path)
    root = tmp_path / "data" / "results"
    bad_lines = ["not json", "[1, 2]", json.dumps({"run_id": "r1"})]
    for i, bad in enumerate(bad_lines):
        exp = f"corrupt{i}"
        exp_dir = root / exp
        exp_dir.mkdir(parents=True, exist_ok=True)
        (exp_dir / "walk_forward_r1.json").write_text("{}", encoding="utf-8")
        (exp_dir / "runs.jsonl").write_text(bad + "\n", encoding="utf-8")
        ref = ResultRef(
            experiment=exp,
            kind=ResultKind.WALK_FORWARD,
            run_id="r1",
            json_path=exp_dir / "walk_forward_r1.json",
            markdown_path=exp_dir / "walk_forward_r1.md",
        )
        with pytest.raises(ValueError, match="corrupted run ledger"):
            apply_result_prune(settings, ResultPrunePlan(keep=1, to_delete=(ref,)), dry_run=False)


def test_plan_result_prune_skips_legacy(tmp_path: Path) -> None:
    """Prune never touches legacy."""
    settings = _settings(tmp_path)
    root = tmp_path / "data" / "results"
    legacy = root / "_legacy"
    legacy.mkdir(parents=True)
    (legacy / "x.json").write_text("{}", encoding="utf-8")
    for i in range(5):
        _write(settings, "old", ResultKind.LEGACY, f"run{i}", T1)

    assert plan_result_prune(settings, keep=1).to_delete == ()


def test_plan_result_prune_rejects_keep_below_one(tmp_path: Path) -> None:
    """Prune rejects keep below one."""
    with pytest.raises(ValueError, match="keep must be"):
        plan_result_prune(_settings(tmp_path), keep=0)


def _flat(settings: DataSettings, flat: str, name: str, payload: object, mtime: float | None = None) -> Path:
    root = Path(settings.resolved_data_root()) / "results" / flat
    root.mkdir(parents=True, exist_ok=True)
    path = root / name
    path.write_text(json.dumps(payload) if not isinstance(payload, str) else payload, encoding="utf-8")
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


def test_plan_result_migration_groups_by_payload_name(tmp_path: Path) -> None:
    """Migration groups by payload name."""
    settings = _settings(tmp_path)
    root = tmp_path / "data" / "results"
    mtime = datetime(2026, 2, 1, tzinfo=UTC).timestamp()
    src = _flat(settings, "experiments", "wf_qqq_adaptive_v5_893bec.json", {"name": "wf_qqq_adaptive_v5"}, mtime)
    (src.with_suffix(".md")).write_text("# old\n", encoding="utf-8")

    plan = plan_result_migration(settings)
    assert [(s.as_posix(), d.as_posix()) for s, d in plan.moves] == [
        (src.as_posix(), (root / "wf_qqq_adaptive_v5" / "legacy_893bec.json").as_posix()),
        ((src.with_suffix(".md")).as_posix(), (root / "wf_qqq_adaptive_v5" / "legacy_893bec.md").as_posix()),
    ]
    assert apply_result_migration(settings, plan, dry_run=False) == 2
    assert (root / "wf_qqq_adaptive_v5" / "legacy_893bec.json").is_file()
    assert (root / "wf_qqq_adaptive_v5" / "legacy_893bec.md").is_file()
    assert not src.exists()
    entries = read_run_ledger(settings, "wf_qqq_adaptive_v5")
    assert len(entries) == 1
    assert entries[0].kind is ResultKind.LEGACY
    assert entries[0].run_id == "893bec"
    assert entries[0].written_at == datetime(2026, 2, 1, tzinfo=UTC)
    assert not (root / "experiments").exists()


def test_plan_result_migration_uses_campaign_and_thesis_ids(tmp_path: Path) -> None:
    """Migration uses campaign id and thesis id."""
    settings = _settings(tmp_path)
    root = tmp_path / "data" / "results"
    _flat(settings, "audits", "FINAL_HISTORICAL_CAMPAIGN_V1_8201d9e.json", {"campaign_id": "FINAL_HISTORICAL_CAMPAIGN_V1"})
    _flat(settings, "thesis", "wave_ai_compute.json", {"thesis_id": "ai_compute"})

    plan = plan_result_migration(settings)
    apply_result_migration(settings, plan, dry_run=False)

    assert (root / "final_historical_campaign_v1" / "legacy_8201d9e.json").is_file()
    assert (root / "thesis_ai_compute" / "legacy_wave_ai_compute.json").is_file()


def test_plan_result_migration_quarantines_uninferrable(tmp_path: Path) -> None:
    """Uninferrable files are quarantined."""
    settings = _settings(tmp_path)
    root = tmp_path / "data" / "results"
    _flat(settings, "experiments", "mystery.json", {"x": 1})
    (root / "experiments" / "mystery.md").write_text("# m\n", encoding="utf-8")
    _flat(settings, "experiments", "array.json", [1, 2])
    _flat(settings, "experiments", "broken.json", "{not json")
    _flat(settings, "experiments", "badslug.json", {"name": "_nope"})
    (root / "experiments" / "odd.json").mkdir()

    plan = plan_result_migration(settings)
    assert plan.ledger_entries == ()
    assert sorted(d.name for _, d in plan.moves) == ["array.json", "badslug.json", "broken.json", "mystery.json", "mystery.md"]
    assert all(d.parent == root / "_legacy" / "experiments" for _, d in plan.moves)

    apply_result_migration(settings, plan, dry_run=False)
    assert (root / "_legacy" / "experiments" / "mystery.json").is_file()
    assert (root / "experiments").is_dir()


def test_plan_result_migration_rejects_collision(tmp_path: Path) -> None:
    """Migration collision rejects whole plan."""
    settings = _settings(tmp_path)
    _flat(settings, "experiments", "dup.json", {"name": "e"})
    _flat(settings, "experiments", "e_dup.json", {"name": "e"})

    with pytest.raises(FileExistsError, match="collision"):
        plan_result_migration(settings)
    assert (tmp_path / "data" / "results" / "experiments" / "dup.json").is_file()
    assert (tmp_path / "data" / "results" / "experiments" / "e_dup.json").is_file()
    assert not (tmp_path / "data" / "results" / "e").exists()


def test_apply_result_migration_dry_run_is_inert(tmp_path: Path) -> None:
    """Migration dry run is inert."""
    settings = _settings(tmp_path)
    src = _flat(settings, "experiments", "wf_qqq_adaptive_v5_893bec.json", {"name": "wf_qqq_adaptive_v5"})

    plan = plan_result_migration(settings)
    assert apply_result_migration(settings, plan, dry_run=True) == len(plan.moves) == 1
    assert src.is_file()
    assert not (tmp_path / "data" / "results" / "wf_qqq_adaptive_v5").exists()
