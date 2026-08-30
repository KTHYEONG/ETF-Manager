"""Layout bounds tests."""

from __future__ import annotations

from pathlib import Path


def test_cli_layout_facade_under_450_lines() -> None:
    """test_cli_layout_facade_under_450_lines"""
    p = Path("src/cli.py")
    assert p.exists()
    lines = p.read_text(encoding="utf-8").splitlines()
    assert len(lines) <= 450, f"facade {len(lines)} lines exceeds 450"


def test_cli_layout_command_modules_under_500_lines() -> None:
    """test_cli_layout_command_modules_under_500_lines"""
    modules = ["parser.py", "resolvers.py", "ingest.py", "thesis.py", "diagnose.py", "sim_run.py", "campaign.py"]
    for name in modules:
        p = Path(f"src/cli_commands/{name}")
        assert p.exists(), f"missing {p}"
        lines = p.read_text(encoding="utf-8").splitlines()
        assert len(lines) <= 500, f"{name} {len(lines)} exceeds 500"


def test_cli_layout_required_modules_exist() -> None:
    """test_cli_layout_required_modules_exist"""
    init = Path("src/cli_commands/__init__.py")
    assert init.exists()
    text = init.read_text(encoding="utf-8")
    assert "CLI_COMMAND_MODULES" in text
    # ensure tuple listing seven stems
    from src.cli_commands import CLI_COMMAND_MODULES

    expected = ("parser", "resolvers", "ingest", "thesis", "diagnose", "sim_run", "campaign")
    assert expected == CLI_COMMAND_MODULES
    for stem in expected:
        assert Path(f"src/cli_commands/{stem}.py").exists()


def test_cli_monolith_test_file_removed_or_shim() -> None:
    """test_cli_monolith_test_file_removed_or_shim"""
    p = Path("tests/unit/test_cli.py")
    if p.exists():
        lines = p.read_text(encoding="utf-8").splitlines()
        assert len(lines) <= 80, f"shim {len(lines)} lines exceeds 80"
    else:
        assert not p.exists()
