"""Persist static bootstrap ETF_METADATA for mapping feasibility and research."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from src.etf_manager.data.pipeline import persist_ingest
from src.etf_manager.data.schema import Dataset, spec_for
from src.etf_manager.data.settings import DataSettings
from src.etf_manager.data.storage import DatasetArtifact, RawPayload

__all__ = ["persist_bootstrap_etf_metadata"]

_DEFAULT_PATH = Path("configs/etf_metadata/bootstrap.json")


def persist_bootstrap_etf_metadata(
    settings: DataSettings,
    *,
    path: Path = _DEFAULT_PATH,
) -> DatasetArtifact:
    """Load ``configs/etf_metadata/bootstrap.json`` and persist one ETF_METADATA partition.

    Bootstrap rows are static filing snapshots for implementation scoring only;
    they are not a live SEC ingest path.

    Raises:
        OSError: When the bootstrap file cannot be read.
        ValueError: When the payload is invalid or fails schema validation.
    """
    if not path.is_file():
        raise OSError(f"bootstrap ETF metadata file not found: {path}")
    spec = spec_for(Dataset.ETF_METADATA)
    frame = (
        pl.read_json(path)
        .with_columns(
            pl.col("effective_date").str.to_date(),
            pl.col("filing_date").str.to_datetime(time_zone="UTC"),
            pl.col("inception_date").str.to_date(),
            pl.col("retrieved_at").str.to_datetime(time_zone="UTC"),
        )
        .cast(pl.Schema(dict(spec.columns)))
    )
    if frame.is_empty():
        raise ValueError(f"bootstrap ETF metadata must be a non-empty JSON array in {path}")
    retrieved_at = datetime.now(tz=UTC)
    raw = RawPayload(
        provider="bootstrap",
        endpoint=str(path),
        request_params={},
        retrieved_at=retrieved_at,
        extension="json",
        content=path.read_bytes(),
    )
    return persist_ingest(frame, Dataset.ETF_METADATA, raw, settings)
