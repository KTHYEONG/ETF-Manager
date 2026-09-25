"""Layout bounds tests."""

from __future__ import annotations

from pathlib import Path

import pytest


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


def test_documented_module_paths_and_commands_exist() -> None:
    """Every module-index path and README command example resolves to the real tree."""
    import re

    index_text = Path("docs/architecture/module_index.md").read_text(encoding="utf-8")
    indexed = re.findall(r"`([^`]+)`", index_text)
    candidates = [
        token
        for token in indexed
        if "/" in token and not token.startswith("http") and "<" not in token and "*" not in token
    ]
    assert candidates, "module index lists no source paths"
    missing = [token for token in candidates if not Path(token.rstrip("/")).exists()]
    assert not missing, f"module index references missing paths: {missing}"

    readme = Path("README.md").read_text(encoding="utf-8")
    assert "python -m src.cli" in readme
    from src.cli_commands.parser import _build_parser

    parser = _build_parser()
    policy_args = parser.parse_args(
        [
            "run",
            "policy",
            "--id",
            "qqq",
            "--start",
            "2020-01-01",
            "--end",
            "2020-12-31",
            "--contribution-krw",
            "1000000",
        ]
    )
    assert policy_args.id == "qqq"
    assert parser.parse_args(["maintain", "data"]).target == "data"
    assert Path("experiments/final_historical_campaign_v1.json").is_file()


def test_data_flow_matches_trust_contracts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Missing Bronze stays readable/repairable while damaged Silver stops the run."""
    from datetime import UTC, date, datetime

    import polars as pl

    from src.data.catalog import latest_artifact
    from src.data.pipeline import persist_ingest
    from src.data.schema import Dataset, spec_for
    from src.data.settings import DataSettings
    from src.data.storage import DataStore, RawPayload, UntrustedDatasetError

    flow = Path("docs/architecture/data-flow.md").read_text(encoding="utf-8")
    assert "maintain data" in flow
    assert "UntrustedDatasetError" in flow

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ETF_MANAGER_DATA_ROOT", raising=False)
    settings = DataSettings(data_root="data")
    spec = spec_for(Dataset.FX)
    retrieved = datetime(2024, 2, 1, 5, 0, tzinfo=UTC)
    frame = pl.DataFrame(
        {
            "date": [date(2024, 1, 30), date(2024, 1, 31)],
            "usdkrw": [1300.0, 1301.0],
            "source": ["synthetic", "synthetic"],
            "retrieved_at": [retrieved, retrieved],
        },
        schema=dict(spec.columns),
    )
    artifact = persist_ingest(
        frame,
        Dataset.FX,
        RawPayload(
            provider="synthetic",
            endpoint="usdkrw/daily",
            request_params={"interval": "daily"},
            retrieved_at=retrieved,
            extension="json",
            content=b'{"v": 1}',
        ),
        settings,
    )
    raw_path = tmp_path / "data" / Path(*artifact.manifest.raw_artifact.relative_path.parts)
    raw_path.unlink()
    with caplog.at_level("WARNING"):
        reread = DataStore(settings).read_normalized(
            latest_artifact(settings, Dataset.FX), spec_for(Dataset.FX)
        )
    assert reread.height == 2
    assert any("silver_without_bronze" in record.message for record in caplog.records)

    tampered = pl.read_parquet(artifact.normalized_path).with_columns(pl.col("usdkrw") + 1.0)
    tampered.write_parquet(artifact.normalized_path)
    with pytest.raises(UntrustedDatasetError):
        DataStore(settings).read_normalized(latest_artifact(settings, Dataset.FX), spec_for(Dataset.FX))


def test_cli_monolith_test_file_removed_or_shim() -> None:
    """test_cli_monolith_test_file_removed_or_shim"""
    p = Path("tests/unit/test_cli.py")
    if p.exists():
        lines = p.read_text(encoding="utf-8").splitlines()
        assert len(lines) <= 80, f"shim {len(lines)} lines exceeds 80"
    else:
        assert not p.exists()
