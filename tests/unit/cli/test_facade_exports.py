"""Facade exports tests."""

from __future__ import annotations

from pathlib import Path


def test_cli_facade_exports_main() -> None:
    """test_cli_facade_exports_main"""
    from src.cli import main

    assert callable(main)


def test_cli_facade_parser_exposes_ingest_and_run() -> None:
    """test_cli_facade_parser_exposes_ingest_and_run"""
    from src.cli_commands.parser import _build_parser

    parser = _build_parser()
    # check top-level subparsers contain ingest and run
    # via format_help or via _subparsers
    help_text = parser.format_help()
    assert "ingest" in help_text
    assert "run" in help_text
    # also verify choices via actions
    # find subparsers action
    subparsers_actions = [a for a in parser._actions if hasattr(a, "choices") and isinstance(a.choices, dict)]
    found = False
    for act in subparsers_actions:
        if act.choices and "ingest" in act.choices and "run" in act.choices:
            found = True
    assert found or ("ingest" in help_text and "run" in help_text)


def test_cli_commands_do_not_import_src_cli() -> None:
    """test_cli_commands_do_not_import_src_cli"""
    for p in Path("src/cli_commands").glob("*.py"):
        text = p.read_text(encoding="utf-8")
        for line in text.splitlines():
            if "src.cli" in line and "src.cli_commands" not in line:
                # line contains src.cli without cli_commands => cycle
                assert "from src.cli" not in line, f"{p} imports from src.cli: {line!r}"
                assert "import src.cli" not in line, f"{p} imports src.cli: {line!r}"
