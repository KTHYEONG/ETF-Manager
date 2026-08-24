"""Tests for bootstrap ETF metadata ingest."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.etf_manager.data.catalog import latest_artifact
from src.etf_manager.data.etf_metadata_bootstrap import persist_bootstrap_etf_metadata
from src.etf_manager.data.schema import Dataset
from src.etf_manager.data.settings import DataSettings


@pytest.mark.parametrize("scenario_id", ["DATA-J-bootstrap-metadata"])
def test_data_j_bootstrap_metadata(scenario_id: str, tmp_path: Path) -> None:
    """DATA-J-bootstrap-metadata"""
    settings = DataSettings(data_root=tmp_path / "data")
    artifact = persist_bootstrap_etf_metadata(settings)
    assert artifact.manifest.row_count >= 10
    latest = latest_artifact(settings, Dataset.ETF_METADATA)
    assert latest.manifest.row_count == artifact.manifest.row_count
