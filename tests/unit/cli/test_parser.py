"""Parser wiring test."""
from __future__ import annotations
import src.cli_commands.parser  # noqa: F401

def test_parser_wiring() -> None:
    assert src.cli_commands.parser._build_parser is not None
