#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import pathlib
import sys

if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())


def _repository_test_files() -> list[str]:
    """Collect every repository test file by walking the tests tree."""
    found: list[str] = []
    for root, dirs, files in os.walk("tests"):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        found.extend(
            os.path.join(root, filename)
            for filename in sorted(files)
            if filename.startswith("test_") and filename.endswith(".py")
        )
    return sorted(found)


def _test_references_source(test_path: str, source_file: str) -> bool:
    """Check whether a test file references a source module path or stem."""
    try:
        with open(test_path, encoding="utf-8", errors="ignore") as handle:
            text = handle.read()
    except OSError:
        return False
    dotted = source_file[:-3].replace("/", ".")
    stem = source_file.rsplit("/", 1)[-1][:-3]
    return dotted in text or stem in text


def _matching_tests(source_file: str, test_files: list[str]) -> list[str]:
    """Return every repository test that covers ``source_file``.

    Exact mirrored ``tests/<category>/<dir>/test_<module>.py`` paths are the fast
    path; otherwise the lean-check AST semantic reference matcher is reused so
    feature-named CLI/workflow tests remain linked.
    """
    parts = source_file.split("/")
    module_name = parts[-1]
    test_name = f"test_{module_name}"
    exact = {
        f"tests/{category}/{'/'.join(parts[1:-1])}/{test_name}" if parts[1:-1]
        else f"tests/{category}/{test_name}"
        for category in ("unit", "integration", "e2e")
    }
    matched = [tp for tp in test_files if tp in exact]
    if matched:
        return matched
    return [tp for tp in test_files if _test_references_source(tp, source_file)]


def _default_family(source_file: str) -> str:
    """Derive a family for sources that have no curated entry yet."""
    parts = source_file.split("/")
    if parts[1] in ("cli.py", "cli_commands"):
        return "cli"
    if len(parts) == 2:
        return "core"
    if parts[1] == "analytics" and parts[2] == "thesis":
        return "thesis"
    return parts[1]


def main() -> None:
    py_files: list[str] = []
    for root, dirs, files in os.walk("src"):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        py_files.extend(
            os.path.join(root, filename)
            for filename in sorted(files)
            if filename.endswith(".py")
        )
    py_files = sorted(py_files)
    test_files = _repository_test_files()

    docs_path = pathlib.Path("docs/code_map.json")
    previous: dict[str, dict[str, object]] = {}
    if docs_path.exists():
        previous = json.loads(docs_path.read_text(encoding="utf-8"))

    code_map: dict[str, object] = {}
    for source_file in py_files:
        # family/architecture는 수작업 큐레이션 값이므로 보존하고, testing만 재계산한다.
        entry: dict[str, object] = {k: v for k, v in previous.get(source_file, {}).items() if k != "testing"}
        entry.setdefault("family", _default_family(source_file))
        matched = _matching_tests(source_file, test_files)
        if matched:
            entry["testing"] = matched[0] if len(matched) == 1 else matched
        code_map[source_file] = entry

    # Tolerate absent active code_map.json; do not recreate archived records under docs/
    # Only active src files are mapped; legacy sources remain in legacy/docs/code_map.json
    import contextlib
    # If active docs/code_map.json is absent, still generate active-only map without archived entries
    with contextlib.suppress(FileNotFoundError):
        if not docs_path.parent.exists():
            docs_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(docs_path, "w", encoding="utf-8") as handle:
            json.dump(code_map, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print(f"regenerated docs/code_map.json with {len(code_map)} canonical sources")
    except FileNotFoundError:
        print("active docs/code_map.json absent, skipped regeneration (archived map remains in legacy/docs)")


if __name__ == "__main__":
    main()
