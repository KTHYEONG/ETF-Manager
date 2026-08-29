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


@pytest.mark.parametrize("scenario_id", ["OVL-TS-skip-missing"])
def test_ovl_ts_skip_missing(scenario_id: str) -> None:
    """OVL-TS-skip-missing"""
    from src.analytics.overlap import overlap_time_series

    spec = spec_for(Dataset.ETF_HOLDINGS)
    retrieved = datetime(2020, 1, 1, tzinfo=UTC)
    filing = datetime(2019, 12, 15, tzinfo=UTC)
    report = date(2019, 12, 31)
    rows = [
        {"etf_ticker": "A", "report_date": report, "filing_date": filing, "holding_id": "X", "issuer_name": "X Inc", "cusip": "X-cusip", "isin": None, "lei": None, "weight_pct": 60.0, "value_usd": 60, "source": "sec_nport", "retrieved_at": retrieved},
        {"etf_ticker": "A", "report_date": report, "filing_date": filing, "holding_id": "Y", "issuer_name": "Y Inc", "cusip": "Y-cusip", "isin": None, "lei": None, "weight_pct": 40.0, "value_usd": 40, "source": "sec_nport", "retrieved_at": retrieved},
        {"etf_ticker": "B", "report_date": report, "filing_date": filing, "holding_id": "X", "issuer_name": "X Inc", "cusip": "X-cusip", "isin": None, "lei": None, "weight_pct": 50.0, "value_usd": 50, "source": "sec_nport", "retrieved_at": retrieved},
        {"etf_ticker": "B", "report_date": report, "filing_date": filing, "holding_id": "Z", "issuer_name": "Z Inc", "cusip": "Z-cusip", "isin": None, "lei": None, "weight_pct": 50.0, "value_usd": 50, "source": "sec_nport", "retrieved_at": retrieved},
    ]
    df = pl.DataFrame(rows).cast(pl.Schema(dict(spec.columns)))
    from src.data.pit import stamp_availability

    stamped = stamp_availability(df, spec)
    valid_as_of = datetime(2020, 1, 15, tzinfo=UTC)
    empty_as_of = datetime(2019, 1, 15, tzinfo=UTC)
    result = overlap_time_series(stamped, vehicle_a="A", vehicle_b="B", as_ofs=[valid_as_of, empty_as_of])
    assert isinstance(result, tuple)
    assert len(result) == 1


@pytest.mark.parametrize("scenario_id", ["test_ovl_c_single_snapshot_no_double_count"])
def test_ovl_c_single_snapshot_no_double_count(scenario_id: str) -> None:
    spec = spec_for(Dataset.ETF_HOLDINGS)
    retrieved = datetime(2020, 1, 1, tzinfo=UTC)
    filing_old = datetime(2019, 12, 15, tzinfo=UTC)
    filing_new = datetime(2020, 3, 15, tzinfo=UTC)
    report_old = date(2019, 12, 31)
    report_new = date(2020, 3, 31)
    rows = [
        {"etf_ticker": "A", "report_date": report_old, "filing_date": filing_old, "holding_id": "X", "issuer_name": "X Inc", "cusip": "X-cusip", "isin": None, "lei": None, "weight_pct": 60.0, "value_usd": 60, "source": "sec_nport", "retrieved_at": retrieved},
        {"etf_ticker": "A", "report_date": report_old, "filing_date": filing_old, "holding_id": "Y", "issuer_name": "Y Inc", "cusip": "Y-cusip", "isin": None, "lei": None, "weight_pct": 40.0, "value_usd": 40, "source": "sec_nport", "retrieved_at": retrieved},
        {"etf_ticker": "A", "report_date": report_new, "filing_date": filing_new, "holding_id": "X", "issuer_name": "X Inc", "cusip": "X-cusip", "isin": None, "lei": None, "weight_pct": 60.0, "value_usd": 60, "source": "sec_nport", "retrieved_at": retrieved},
        {"etf_ticker": "A", "report_date": report_new, "filing_date": filing_new, "holding_id": "Y", "issuer_name": "Y Inc", "cusip": "Y-cusip", "isin": None, "lei": None, "weight_pct": 40.0, "value_usd": 40, "source": "sec_nport", "retrieved_at": retrieved},
        {"etf_ticker": "B", "report_date": report_new, "filing_date": filing_new, "holding_id": "X", "issuer_name": "X Inc", "cusip": "X-cusip", "isin": None, "lei": None, "weight_pct": 50.0, "value_usd": 50, "source": "sec_nport", "retrieved_at": retrieved},
        {"etf_ticker": "B", "report_date": report_new, "filing_date": filing_new, "holding_id": "Z", "issuer_name": "Z Inc", "cusip": "Z-cusip", "isin": None, "lei": None, "weight_pct": 50.0, "value_usd": 50, "source": "sec_nport", "retrieved_at": retrieved},
    ]
    df = pl.DataFrame(rows).cast(pl.Schema(dict(spec.columns)))
    from src.data.pit import stamp_availability

    stamped = stamp_availability(df, spec)
    as_of = datetime(2020, 4, 1, tzinfo=UTC)
    report_out = pairwise_overlap(stamped, vehicle_a="A", vehicle_b="B", as_of=as_of)
    assert report_out.a_only_weight_pct <= 100.0
    assert report_out.b_only_weight_pct <= 100.0
    assert report_out.a_only_weight_pct >= 0
    assert report_out.b_only_weight_pct >= 0
    # If double counting, overlap would be ~100 (2x50) and a_only ~150. Check single snapshot.
    assert report_out.overlap_pct == 50.0
    total_implied = report_out.overlap_pct + report_out.a_only_weight_pct
    assert total_implied <= 105
    total_implied_b = report_out.overlap_pct + report_out.b_only_weight_pct
    assert total_implied_b <= 105


@pytest.mark.parametrize("scenario_id", ["test_ovl_d_weight_sum_fail_closed"])
def test_ovl_d_weight_sum_fail_closed(scenario_id: str) -> None:
    spec = spec_for(Dataset.ETF_HOLDINGS)
    retrieved = datetime(2020, 1, 1, tzinfo=UTC)
    filing = datetime(2019, 12, 15, tzinfo=UTC)
    report = date(2019, 12, 31)
    rows = [
        {"etf_ticker": "A", "report_date": report, "filing_date": filing, "holding_id": "X", "issuer_name": "X Inc", "cusip": "X-cusip", "isin": None, "lei": None, "weight_pct": 100.0, "value_usd": 100, "source": "sec_nport", "retrieved_at": retrieved},
        {"etf_ticker": "A", "report_date": report, "filing_date": filing, "holding_id": "Y", "issuer_name": "Y Inc", "cusip": "Y-cusip", "isin": None, "lei": None, "weight_pct": 100.0, "value_usd": 100, "source": "sec_nport", "retrieved_at": retrieved},
        {"etf_ticker": "B", "report_date": report, "filing_date": filing, "holding_id": "X", "issuer_name": "X Inc", "cusip": "X-cusip", "isin": None, "lei": None, "weight_pct": 50.0, "value_usd": 50, "source": "sec_nport", "retrieved_at": retrieved},
        {"etf_ticker": "B", "report_date": report, "filing_date": filing, "holding_id": "Z", "issuer_name": "Z Inc", "cusip": "Z-cusip", "isin": None, "lei": None, "weight_pct": 50.0, "value_usd": 50, "source": "sec_nport", "retrieved_at": retrieved},
    ]
    df = pl.DataFrame(rows).cast(pl.Schema(dict(spec.columns)))
    from src.data.pit import stamp_availability

    stamped = stamp_availability(df, spec)
    as_of = datetime(2020, 1, 15, tzinfo=UTC)
    with pytest.raises(ValueError, match=r"weight.*sum|band"):
        pairwise_overlap(stamped, vehicle_a="A", vehicle_b="B", as_of=as_of)
