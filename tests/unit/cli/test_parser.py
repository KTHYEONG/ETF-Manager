"""Parser wiring test."""
from __future__ import annotations
import src.cli_commands.parser  # noqa: F401

def test_parser_wiring() -> None:
    assert src.cli_commands.parser._build_parser is not None


def test_maintain_data_defaults_to_dry_run() -> None:
    """`maintain data` parses without apply."""
    from src.cli_commands.parser import _build_parser

    args = _build_parser().parse_args(["maintain", "data"])
    assert args.target == "data"
    assert args.apply is False


def test_maintain_data_apply_flag_and_existing_targets_still_parse() -> None:
    """`maintain data --apply` opts in while existing maintain targets keep parsing."""
    from src.cli_commands.parser import _build_parser

    assert _build_parser().parse_args(["maintain", "data", "--apply"]).apply is True
    assert _build_parser().parse_args(["maintain", "prune"]).target == "prune"
    assert _build_parser().parse_args(["maintain", "recover", "rates"]).target == "recover"
    assert _build_parser().parse_args(["maintain", "results", "list"]).target == "results"
