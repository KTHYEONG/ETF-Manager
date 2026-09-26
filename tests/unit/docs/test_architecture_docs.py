"""Architecture documentation layout and code_map coverage guards."""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
ARCHITECTURE_DIR = REPO / "docs/architecture"
CANONICAL_DOCS = ("system-design.md", "engineering-decisions.md")
README = REPO / "README.md"
CODE_MAP = REPO / "docs/code_map.json"
DOC_LINE_BUDGET = 300
README_LINE_BUDGET = 200
CHECKED_ROOTS = ("src/", "configs/", "tests/", "experiments/", "docs/")


def _documented_repo_paths(text: str) -> list[str]:
    """Backticked repository paths, without symbol suffixes or trailing slashes."""
    paths: list[str] = []
    for token in re.findall(r"`([^`\n]+)`", text):
        if not token.startswith(CHECKED_ROOTS):
            continue
        if any(marker in token for marker in ("<", "*", " ", ",", "(")):
            continue
        paths.append(token.split("::")[0].rstrip("/"))
    return paths


def test_architecture_directory_holds_exactly_the_two_canonical_docs() -> None:
    names = sorted(path.name for path in ARCHITECTURE_DIR.iterdir() if path.is_file())
    assert names == sorted(CANONICAL_DOCS)


def test_architecture_docs_respect_line_budget() -> None:
    for name in CANONICAL_DOCS:
        lines = (ARCHITECTURE_DIR / name).read_text(encoding="utf-8").splitlines()
        assert len(lines) <= DOC_LINE_BUDGET, name
    assert len(README.read_text(encoding="utf-8").splitlines()) <= README_LINE_BUDGET


def test_documented_paths_exist() -> None:
    for doc in (README, *(ARCHITECTURE_DIR / name for name in CANONICAL_DOCS)):
        paths = _documented_repo_paths(doc.read_text(encoding="utf-8"))
        assert paths, doc.name
        missing = [rel for rel in paths if not (REPO / rel).exists()]
        assert not missing, f"{doc.name} references missing paths: {missing}"


def test_docs_contain_no_local_absolute_paths() -> None:
    for doc in (README, *(ARCHITECTURE_DIR / name for name in CANONICAL_DOCS)):
        text = doc.read_text(encoding="utf-8")
        assert "file:///" not in text, doc.name
        assert "/home/" not in text, doc.name


def test_readme_links_both_canonical_docs() -> None:
    text = README.read_text(encoding="utf-8")
    for name in CANONICAL_DOCS:
        assert f"docs/architecture/{name}" in text


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


def test_code_map_architecture_links_resolve() -> None:
    code_map = json.loads(CODE_MAP.read_text(encoding="utf-8"))
    for rel, entry in code_map.items():
        target = entry.get("architecture") if isinstance(entry, dict) else None
        if isinstance(target, str) and target.startswith("docs/architecture/"):
            assert (REPO / target).is_file(), f"{rel} -> {target}"


def test_cli_command_modules_exist() -> None:
    from src.cli_commands import CLI_COMMAND_MODULES

    expected = ("parser", "resolvers", "ingest", "thesis", "diagnose", "sim_run", "campaign")
    assert expected == CLI_COMMAND_MODULES
    for stem in expected:
        assert Path(f"src/cli_commands/{stem}.py").exists()
