"""NPORT batch tests (Wave 7)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.data.schema import Dataset
from src.data.settings import DataSettings
from src.data.storage import DatasetArtifact, DatasetManifest, RawArtifact
from pathlib import PurePosixPath
from datetime import UTC, datetime


@pytest.mark.parametrize("scenario_id", ["NPORT-BATCH-sequential"])
def test_nport_batch_sequential(scenario_id: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """NPORT-BATCH-sequential"""
    from src.data import nport_ingest

    settings = DataSettings(data_root=tmp_path / "data")

    calls: list[str] = []

    def fake_quarter(*, filing_quarter: str, series_map_path: Path = Path("configs/etf_metadata/nport_series_map.json"), settings: DataSettings, client=None):
        calls.append(filing_quarter)
        # Return dummy artifact
        manifest = DatasetManifest(
            dataset=Dataset.ETF_HOLDINGS,
            provider="sec",
            endpoint=f"https://example.com/{filing_quarter}",
            request_params={"filing_quarter": filing_quarter},
            retrieved_at=datetime.now(UTC),
            raw_artifact=RawArtifact(relative_path=PurePosixPath(f"raw/sec/etf_holdings/{filing_quarter}/payload.zip"), sha256="a"*64, retrieved_at=datetime.now(UTC)),
            normalized_relative_path=PurePosixPath(f"normalized/etf_holdings/schema_version=1/{filing_quarter}.parquet"),
            normalized_sha256="b"*64,
            row_count=1,
            schema_version="1",
            normalization_version="1",
            quality_findings=(),
        )
        return DatasetArtifact(normalized_path=tmp_path / f"{filing_quarter}.parquet", manifest_path=tmp_path / f"{filing_quarter}.json", manifest=manifest)

    monkeypatch.setattr(nport_ingest, "fetch_and_persist_nport_quarter", fake_quarter)

    result = nport_ingest.fetch_and_persist_nport_quarters(filing_quarters=["2019q4", "2020q1"], series_map_path=Path("configs/etf_metadata/nport_series_map.json"), settings=settings)
    assert isinstance(result, tuple)
    assert len(result) == 1
    assert calls == ["2019q4", "2020q1"]
