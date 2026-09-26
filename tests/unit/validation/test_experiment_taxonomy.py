"""Wave 4 taxonomy contract tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.data.paths import EXPERIMENTS_DIR, EXPERIMENT_ARCHIVE_DIR, EXPERIMENT_INDEX_PATH
from src.validation.experiment import resolve_experiment_config_path


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
    return json.loads(Path(EXPERIMENT_INDEX_PATH).read_text(encoding="utf-8"))


def test_taxonomy_index_covers_all_json() -> None:
    data = _load_index()
    assert "files" in data
    basenames = set()
    for p in Path(EXPERIMENTS_DIR).glob("*.json"):
        if p.name == "INDEX.json":
            continue
        basenames.add(p.name)
    for p in Path(EXPERIMENT_ARCHIVE_DIR).glob("*.json"):
        basenames.add(p.name)
    retired = {name for name, meta in data["files"].items() if meta.get("retired") is True}
    assert retired.isdisjoint(basenames), "retired entries must have no config file"
    assert set(data["files"].keys()) - retired == basenames
    for name, meta in data["files"].items():
        assert meta["status"] in {"active", "fixture", "archived"}, f"{name} bad status {meta['status']}"
        assert isinstance(meta.get("kind"), str) and meta["kind"]  # noqa: PT018, RUF018


def test_resolve_experiment_config_path_missing_raises() -> None:
    with pytest.raises(FileNotFoundError):
        resolve_experiment_config_path("configs/experiments/does_not_exist_zz.json")


def test_taxonomy_readme_describes_catalog_without_mirror_table() -> None:
    readme_path = Path(EXPERIMENTS_DIR) / "README.md"
    assert readme_path.is_file()
    text = readme_path.read_text(encoding="utf-8")
    assert "INDEX.json` is the only catalog" in text
    assert "tests/unit/validation/test_experiment_taxonomy.py" in text
    assert "trial-lineage census counts them" in text


def test_configs_holds_no_experiment_definitions() -> None:
    assert not Path("configs/experiments").exists()


def test_experiment_map_points_at_existing_configs() -> None:
    from src.analytics.thesis.wave import load_thesis_experiment_map

    mapping = load_thesis_experiment_map()
    assert mapping
    raw = json.loads(Path("configs/theses/experiment_map.json").read_text(encoding="utf-8"))
    for thesis_id, path in mapping.items():
        assert Path(path).is_file(), f"{thesis_id}: missing {path}"
        assert Path(path).name == Path(raw[thesis_id.value]).name, f"{thesis_id}: {path}"
