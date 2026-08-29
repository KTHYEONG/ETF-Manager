"""Overlap analytics spec tests."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl
import pytest

from src.analytics.overlap import pairwise_overlap
from src.data.schema import Dataset, spec_for


@pytest.mark.parametrize("scenario_id", ["OVL-A-pairwise-min"])
def test_ovl_a_pairwise_min(scenario_id: str) -> None:
    spec = spec_for(Dataset.ETF_HOLDINGS)
    retrieved = datetime(2020, 1, 1, tzinfo=UTC)
    filing = datetime(2019, 12, 15, tzinfo=UTC)
    report = date(2019, 12, 31)
    rows = [
        {
            "etf_ticker": "A",
            "report_date": report,
            "filing_date": filing,
            "holding_id": "X",
            "issuer_name": "X Inc",
            "cusip": "X-cusip",
            "isin": None,
            "lei": None,
            "weight_pct": 60.0,
            "value_usd": 60,
            "source": "sec_nport",
            "retrieved_at": retrieved,
        },
        {
            "etf_ticker": "A",
            "report_date": report,
            "filing_date": filing,
            "holding_id": "Y",
            "issuer_name": "Y Inc",
            "cusip": "Y-cusip",
            "isin": None,
            "lei": None,
            "weight_pct": 40.0,
            "value_usd": 40,
            "source": "sec_nport",
            "retrieved_at": retrieved,
        },
        {
            "etf_ticker": "B",
            "report_date": report,
            "filing_date": filing,
            "holding_id": "X",
            "issuer_name": "X Inc",
            "cusip": "X-cusip",
            "isin": None,
            "lei": None,
            "weight_pct": 50.0,
            "value_usd": 50,
            "source": "sec_nport",
            "retrieved_at": retrieved,
        },
        {
            "etf_ticker": "B",
            "report_date": report,
            "filing_date": filing,
            "holding_id": "Z",
            "issuer_name": "Z Inc",
            "cusip": "Z-cusip",
            "isin": None,
            "lei": None,
            "weight_pct": 50.0,
            "value_usd": 50,
            "source": "sec_nport",
            "retrieved_at": retrieved,
        },
    ]
    df = pl.DataFrame(rows).cast(pl.Schema(dict(spec.columns)))
    from src.data.pit import stamp_availability

    stamped = stamp_availability(df, spec)
    as_of = datetime(2020, 1, 15, tzinfo=UTC)
    report_out = pairwise_overlap(stamped, vehicle_a="A", vehicle_b="B", as_of=as_of)
    assert report_out.overlap_pct == 50.0
    assert report_out.shared_holdings_count == 1


@pytest.mark.parametrize("scenario_id", ["OVL-B-no-adoption-import"])
def test_ovl_b_no_adoption_import(scenario_id: str) -> None:
    text = Path("src/analytics/overlap.py").read_text(encoding="utf-8")
    assert "adoption_passes" not in text
