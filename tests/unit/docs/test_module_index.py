"""Documentation layout and code_map coverage guards."""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
MODULE_INDEX = REPO / "docs/architecture/04_module_index.md"
OVERVIEW = REPO / "docs/architecture/00_system_overview.md"
CODE_MAP = REPO / "docs/code_map.json"

REQUIRED_ANCHORS = (
    "## ingest",
    "## policy-run",
    "## validate-campaign",
    "## thesis-research",
    "## diagnose-qqq",
    "## experiment-config",
)

PATH_TOKEN = re.compile(r"`((?:src|configs|docs)/[^`]+)`|^- ((?:src|configs|docs)/\S+)")


def test_module_index_has_required_anchors() -> None:
    text = MODULE_INDEX.read_text(encoding="utf-8")
    for anchor in REQUIRED_ANCHORS:
        assert text.count(anchor) == 1


def test_module_index_under_120_lines() -> None:
    assert len(MODULE_INDEX.read_text(encoding="utf-8").splitlines()) <= 120


def test_module_index_paths_exist() -> None:
    text = MODULE_INDEX.read_text(encoding="utf-8")
    sections = re.split(r"(?=^## )", text, flags=re.MULTILINE)
    for section in sections:
        if not section.startswith("## "):
            continue
        paths: list[str] = []
        for match in PATH_TOKEN.finditer(section):
            token = match.group(1) or match.group(2)
            if token.endswith("/"):
                token = token.rstrip("/")
            paths.append(token)
        assert 1 <= len(paths) <= 8, section.splitlines()[0]
        for rel in paths:
            assert (REPO / rel).exists(), rel


def test_code_map_covers_all_src_py() -> None:
    code_map = json.loads(CODE_MAP.read_text(encoding="utf-8"))
    src_files = sorted(
        p.relative_to(REPO).as_posix()
        for p in (REPO / "src").rglob("*.py")
        if "__pycache__" not in p.parts
    )
    for rel in src_files:
        assert rel in code_map, rel
        entry = code_map[rel]
        assert isinstance(entry, dict)
        family = entry.get("family")
        assert isinstance(family, str)
        assert bool(family)


def test_overview_links_module_index() -> None:
    assert "04_module_index.md" in OVERVIEW.read_text(encoding="utf-8")
