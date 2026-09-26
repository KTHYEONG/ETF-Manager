"""Architecture documentation layout and code_map coverage guards."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
ARCHITECTURE_DIR = REPO / "docs/architecture"
CANONICAL_DOCS = ("system-design.md", "engineering-decisions.md")
README = REPO / "README.md"
CODE_MAP = REPO / "docs/code_map.json"
DOC_LINE_BUDGET = 300
README_LINE_BUDGET = 200
CHECKED_ROOTS = ("src/", "configs/", "tests/", "docs/")


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


RESULTS_DIR = REPO / "docs/results"
RESULT_PAGES = ("general.md", "pension.md", "isa.md")
RESULT_PAGE_BUDGET = 80
RESULTS_README_BUDGET = 40
_HEX64_PATTERN = re.compile(r"[0-9a-f]{64}")
_PAGE_EVIDENCE = {
    "general.md": ("data/runs/final_historical_campaign_v1/legacy_1c7a8194f7ce5a59.json",),
    "pension.md": ("data/frozen/pension/pension_decision_v3__2026-08-31__1175d4ef48b20a9a.json",),
    "isa.md": ("data/frozen/isa/isa_household_v1__94cad1c1f1d4b913.json",),
}


def test_result_summaries_exist_and_are_short() -> None:
    """docs/results/ holds exactly the index plus three account pages within line budgets."""
    names = sorted(path.name for path in RESULTS_DIR.iterdir() if path.is_file())
    assert names == ["README.md", *sorted(RESULT_PAGES)]
    assert len((RESULTS_DIR / "README.md").read_text(encoding="utf-8").splitlines()) <= RESULTS_README_BUDGET
    for page in RESULT_PAGES:
        lines = (RESULTS_DIR / page).read_text(encoding="utf-8").splitlines()
        assert len(lines) <= RESULT_PAGE_BUDGET, page


def test_result_summaries_cite_frozen_evidence() -> None:
    """Each account page carries a 64-hex sha256 and its decision config command with an existing config."""
    for page in RESULT_PAGES:
        text = (RESULTS_DIR / page).read_text(encoding="utf-8")
        assert _HEX64_PATTERN.search(text), f"{page} lacks a sha256 value"
        configs = re.findall(r"configs/decision/\S+\.json", text)
        assert configs, f"{page} lacks a decision config command"
        for rel in configs:
            assert (REPO / rel).is_file(), f"{page} references missing {rel}"


def test_result_summary_hashes_match_local_evidence() -> None:
    """Recomputed evidence hashes equal the page values; skipped when data/ is absent."""
    for page, candidates in _PAGE_EVIDENCE.items():
        text = (RESULTS_DIR / page).read_text(encoding="utf-8")
        present = [rel for rel in candidates if (REPO / rel).is_file()]
        if not present:
            pytest.skip(f"{page}: local evidence absent (fresh clone without data/)")
        for rel in present:
            digest = hashlib.sha256((REPO / rel).read_bytes()).hexdigest()
            assert digest in text, f"{page} hash mismatch for {rel}"


def test_readme_and_architecture_docs_contain_no_stale_roots() -> None:
    """README and canonical docs never cite the removed experiments/, records/, or data/results roots."""
    assert "experiments/" not in CHECKED_ROOTS
    for doc in (README, *(ARCHITECTURE_DIR / name for name in CANONICAL_DOCS)):
        text = doc.read_text(encoding="utf-8")
        assert "experiments/" not in text, doc.name
        assert "records/" not in text, doc.name
        assert "data/results" not in text, doc.name


def test_removed_roots_stay_removed() -> None:
    """experiments/ and records/ are gone and docs/results/ holds only the four allowed pages."""
    assert not (REPO / "experiments").exists()
    assert not (REPO / "records").exists()
    git = shutil.which("git")
    assert git is not None
    tracked = subprocess.run(  # noqa: S603 - fixed argv (git ls-files) in the repo root
        [git, "ls-files", "docs/results"], capture_output=True, text=True, cwd=REPO
    ).stdout.split()
    assert sorted(tracked) == [
        "docs/results/README.md",
        "docs/results/general.md",
        "docs/results/isa.md",
        "docs/results/pension.md",
    ]
