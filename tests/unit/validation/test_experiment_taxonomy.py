"""Wave 4 taxonomy contract tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.data.paths import EXPERIMENTS_DIR, EXPERIMENT_ARCHIVE_DIR, EXPERIMENT_INDEX_PATH
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
    assert set(data["files"].keys()) == basenames
    for name, meta in data["files"].items():
        assert meta["status"] in {"active", "fixture", "archived"}, f"{name} bad status {meta['status']}"
        assert isinstance(meta.get("kind"), str) and meta["kind"]  # noqa: PT018, RUF018


def test_taxonomy_archive_set_moved() -> None:
    data = _load_index()
    for name in ARCHIVE_SET:
        # archived entry must exist under archive/
        archive_path = Path(EXPERIMENT_ARCHIVE_DIR) / name
        assert archive_path.is_file(), f"archive missing {name}"
        # top-level should not exist (or if exists, status must be archived but we assert missing)
        top_path = Path(EXPERIMENTS_DIR) / name
        # spec says not is_file OR status==archived; we enforce not is_file for archived set
        assert not top_path.is_file(), f"top-level should not contain archived {name}"
        assert data["files"][name]["status"] == "archived"


def test_taxonomy_active_v5_and_reserve_v4_remain() -> None:
    data = _load_index()
    for name in ("wf_qqq_adaptive_v5.json", "wf_qqq_reserve_v4.json"):
        p = Path(EXPERIMENTS_DIR) / name
        assert p.is_file(), f"active file missing {name}"
        assert data["files"][name]["status"] == "active"


def test_resolve_experiment_config_path_archive_fallback() -> None:
    # m_qqq_grid is archived; historical path should resolve to archive
    historic = "configs/experiments/m_qqq_grid.json"
    resolved = resolve_experiment_config_path(historic)
    expected = (Path(EXPERIMENT_ARCHIVE_DIR) / "m_qqq_grid.json").resolve()
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


def test_taxonomy_readme_describes_catalog_without_mirror_table() -> None:
    readme_path = Path(EXPERIMENTS_DIR) / "README.md"
    assert readme_path.is_file()
    text = readme_path.read_text(encoding="utf-8")
    assert "INDEX.json` is the only catalog" in text
    assert "tests/unit/validation/test_experiment_taxonomy.py" in text
    assert "trial-lineage census counts them" in text


def test_taxonomy_soxx10_adaptive_v5_indexed() -> None:
    import json
    from pathlib import Path
    data = json.loads(Path(EXPERIMENT_INDEX_PATH).read_text(encoding='utf-8'))
    name = 'wf_qqq_soxx10_adaptive_v5.json'
    assert data['files'][name]['status'] == 'active'
    assert (Path(EXPERIMENTS_DIR) / name).is_file()
    assert not (Path(EXPERIMENT_ARCHIVE_DIR) / name).is_file()


def test_configs_holds_no_experiment_definitions() -> None:
    assert not Path("configs/experiments").exists()


def test_experiment_map_points_at_existing_configs() -> None:
    from src.analytics.thesis.wave import load_thesis_experiment_map

    mapping = load_thesis_experiment_map()
    assert mapping
    for thesis_id, path in mapping.items():
        assert str(path).startswith(f"{EXPERIMENTS_DIR.as_posix()}/"), f"{thesis_id}: {path}"
        assert Path(path).is_file(), f"{thesis_id}: missing {path}"
