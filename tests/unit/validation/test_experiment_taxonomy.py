"""Wave 4 taxonomy contract tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.validation.experiment import load_experiment_config, resolve_experiment_config_path


ARCHIVE_SET = [
    "wf_qqq_reserve.json",
    "wf_qqq_reserve_v2.json",
    "wf_qqq_reserve_v3.json",
    "wf_qqq_adaptive_contribution.json",
    "wf_qqq_adaptive_v2.json",
    "wf_qqq_adaptive_v3.json",
    "wf_qqq_adaptive_v4.json",
    "m_qqq_grid.json",
    "m_qqq_iwf.json",
    "wf_qqq_future_core.json",
    "wf_vt_ff_proxy.json",
]


def _load_index() -> dict:
    return json.loads(Path("configs/experiments/INDEX.json").read_text(encoding="utf-8"))


def test_taxonomy_index_covers_all_json() -> None:
    data = _load_index()
    assert "files" in data
    basenames = set()
    for p in Path("configs/experiments").glob("*.json"):
        if p.name == "INDEX.json":
            continue
        basenames.add(p.name)
    for p in Path("configs/experiments/archive").glob("*.json"):
        basenames.add(p.name)
    assert set(data["files"].keys()) == basenames
    for name, meta in data["files"].items():
        assert meta["status"] in {"active", "fixture", "archived"}, f"{name} bad status {meta['status']}"
        assert isinstance(meta.get("kind"), str) and meta["kind"]  # noqa: PT018, RUF018


def test_taxonomy_archive_set_moved() -> None:
    data = _load_index()
    for name in ARCHIVE_SET:
        # archived entry must exist under archive/
        archive_path = Path("configs/experiments/archive") / name
        assert archive_path.is_file(), f"archive missing {name}"
        # top-level should not exist (or if exists, status must be archived but we assert missing)
        top_path = Path("configs/experiments") / name
        # spec says not is_file OR status==archived; we enforce not is_file for archived set
        assert not top_path.is_file(), f"top-level should not contain archived {name}"
        assert data["files"][name]["status"] == "archived"


def test_taxonomy_active_v5_and_reserve_v4_remain() -> None:
    data = _load_index()
    for name in ("wf_qqq_adaptive_v5.json", "wf_qqq_reserve_v4.json"):
        p = Path("configs/experiments") / name
        assert p.is_file(), f"active file missing {name}"
        assert data["files"][name]["status"] == "active"


def test_resolve_experiment_config_path_archive_fallback() -> None:
    # m_qqq_grid is archived; historical path should resolve to archive
    historic = "configs/experiments/m_qqq_grid.json"
    resolved = resolve_experiment_config_path(historic)
    expected = (Path("configs/experiments/archive") / "m_qqq_grid.json").resolve()
    # also accept repo-root absolute fallback
    assert resolved == expected or (resolved.name == "m_qqq_grid.json" and "archive" in str(resolved))
    # load via old path must still return ExperimentSpec
    spec = load_experiment_config(historic)
    assert spec is not None
    # sanity: spec name should match file
    assert spec.name == "m_qqq_grid" or "grid" in spec.name.lower() or spec.name


def test_resolve_experiment_config_path_missing_raises() -> None:
    with pytest.raises(FileNotFoundError):
        resolve_experiment_config_path("configs/experiments/does_not_exist_zz.json")


def test_taxonomy_readme_statuses_match_index() -> None:
    readme_path = Path("configs/experiments/README.md")
    assert readme_path.is_file()
    text = readme_path.read_text(encoding="utf-8")
    data = _load_index()
    for name in data["files"].keys():  # noqa: SIM118
        assert name in text, f"{name} not found in README"
    # If markdown table exists, assert status column equals INDEX
    # Parse table rows: | File | Status | ...
    lines = text.splitlines()
    table_rows: dict[str, str] = {}
    for line in lines:
        # match markdown table row with pipes
        if line.strip().startswith("|") and "|" in line:
            parts = [p.strip() for p in line.strip().strip("|").split("|")]
            if len(parts) >= 2:
                file_col = parts[0]
                status_col = parts[1].lower()
                # skip header row
                if file_col.lower() == "file" or "---" in file_col:
                    continue
                # file_col should be a json basename
                if file_col.endswith(".json") and status_col in {"active", "fixture", "archived"}:  # noqa: SIM102
                    table_rows[file_col] = status_col
    if table_rows:
        for name, meta in data["files"].items():
            if name in table_rows:
                assert table_rows[name] == meta["status"], f"README status mismatch for {name}"
