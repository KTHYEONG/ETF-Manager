"""Tests for sec_nport provider."""

from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from src.data.providers.sec_nport import SecNportClient, normalize_nport_holdings


def test_sec_nport_parse_smoke() -> None:
    fixture = Path("tests/fixtures/nport/minimal_2019q4.zip")
    content = fixture.read_bytes()
    df = SecNportClient.parse_quarter_zip(content, filing_quarter="2019q4")
    assert df.height >= 1


def test_normalize_preserves_amendment_filing_dates() -> None:
    holding = pl.DataFrame(
        {
            "holding_id": ["H1", "H1"],
            "series_id": ["S1", "S1"],
            "report_date": ["2019-12-31", "2019-12-31"],
            "weight_pct": [5.0, 6.0],
            "value_usd": [1_000_000.0, 1_200_000.0],
            "accession_number": ["0001", "0002"],
        }
    )
    info = pl.DataFrame(
        {
            "series_id": ["S1", "S1"],
            "filing_date": ["2020-03-15T12:00:00Z", "2020-04-20T12:00:00Z"],
            "report_date": ["2019-12-31", "2019-12-31"],
            "accession_number": ["0001", "0002"],
        }
    )
    raw = {"FUND_REPORTED_HOLDING": holding, "FUND_REPORTED_INFO": info}
    out = normalize_nport_holdings(
        raw,
        series_map={"S1": "SOXX"},
        retrieved_at=datetime.now(UTC),
    )
    assert out.height == 2
    by_filing = {row["filing_date"].date(): row["weight_pct"] for row in out.sort("filing_date").to_dicts()}
    assert by_filing[datetime(2020, 3, 15, tzinfo=UTC).date()] == 5.0
    assert by_filing[datetime(2020, 4, 20, tzinfo=UTC).date()] == 6.0
