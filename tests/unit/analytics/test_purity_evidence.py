"""Tests for purity evidence."""
from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl
import pytest

from src.data.schema import Dataset, spec_for
from src.data.settings import DataSettings
from src.data.thesis_fundamentals import ExposureNote, PuritySpec
from src.policy.thesis import Horizon, ThesisId, ThesisSpec


def test_thesis_aligned_weight_matches_isin() -> None:
    from src.analytics.purity_evidence import thesis_aligned_weight_pct

    snapshot = pl.DataFrame(
        {
            "weight_pct": [60.0, 40.0],
            "isin": ["X_ISIN", "Y_ISIN"],
            "cusip": [None, None],
            "holding_id": ["H1", "H2"],
        }
    )
    notes = (ExposureNote(isin="X_ISIN", cusip=None, role="grid", note="test"),)
    result = thesis_aligned_weight_pct(snapshot=snapshot, notes=notes)
    assert result["thesis_aligned_weight_pct"] == pytest.approx(60.0)
    assert result["non_aligned_weight_pct"] == pytest.approx(40.0)
    assert result["matched_notes_count"] == 1


def test_thesis_aligned_weight_cusip_fallback() -> None:
    from src.analytics.purity_evidence import thesis_aligned_weight_pct

    snapshot = pl.DataFrame(
        {
            "weight_pct": [25.0, 30.0],
            "isin": [None, None],
            "cusip": ["C1", "000000000"],
            "holding_id": ["H1", "H2"],
        }
    )
    notes = (ExposureNote(isin=None, cusip="C1", role="grid", note="test cusip"),)
    result = thesis_aligned_weight_pct(snapshot=snapshot, notes=notes)
    assert result["thesis_aligned_weight_pct"] == pytest.approx(25.0)
    assert result["matched_notes_count"] == 1
    # placeholder should not be matched; ensure not counted
    assert result["non_aligned_weight_pct"] == pytest.approx(30.0)


def test_compute_purity_slot_labels_and_incremental(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from src.analytics.purity_evidence import compute_purity_slot
    from src.data.pit import stamp_availability

    thesis = ThesisSpec(
        id=ThesisId.AI_POWER_BOTTLENECK,
        version=1,
        title="test",
        status="research",
        horizon=Horizon(min_years=5, target_years=10),
        causal_chain=["a"],
        falsifiers=["f1"],
        candidate_sleeves=["ai_power_equipment"],
        historical_proxies=["GRID"],
    )
    settings = DataSettings(data_root=tmp_path / "data")
    as_of = datetime(2020, 1, 15, tzinfo=UTC)

    # Build purity spec mock
    aligned_isin = "ALIGNED_ISIN_80"
    notes = (ExposureNote(isin=aligned_isin, cusip=None, role="grid", note="aligned"),)
    purity_spec = PuritySpec(
        vehicle_ticker="GRID",
        incumbent_ticker="QQQ",
        pure_min_pct=70.0,
        impure_max_pct=40.0,
        exposure_notes=notes,
    )

    # Holdings fixture where GRID has aligned 80 + shared 10 + non-aligned 10, QQQ has shared 10
    spec = spec_for(Dataset.ETF_HOLDINGS)
    retrieved = datetime(2020, 1, 1, tzinfo=UTC)
    filing = datetime(2019, 12, 15, tzinfo=UTC)
    report_date = date(2019, 12, 31)
    rows = [
        {"etf_ticker": "GRID", "report_date": report_date, "filing_date": filing, "holding_id": "A1", "issuer_name": "Aligned Co", "cusip": "CUSIP_A1", "isin": aligned_isin, "lei": None, "weight_pct": 80.0, "value_usd": 80, "source": "sec_nport", "retrieved_at": retrieved},
        {"etf_ticker": "GRID", "report_date": report_date, "filing_date": filing, "holding_id": "SHARED", "issuer_name": "Shared Co", "cusip": "SHARED_CUSIP", "isin": None, "lei": None, "weight_pct": 10.0, "value_usd": 10, "source": "sec_nport", "retrieved_at": retrieved},
        {"etf_ticker": "GRID", "report_date": report_date, "filing_date": filing, "holding_id": "NON", "issuer_name": "NonAligned", "cusip": "CUSIP_NON", "isin": "NON_ISIN", "lei": None, "weight_pct": 10.0, "value_usd": 10, "source": "sec_nport", "retrieved_at": retrieved},
        {"etf_ticker": "QQQ", "report_date": report_date, "filing_date": filing, "holding_id": "SHARED", "issuer_name": "Shared Co", "cusip": "SHARED_CUSIP", "isin": None, "lei": None, "weight_pct": 10.0, "value_usd": 10, "source": "sec_nport", "retrieved_at": retrieved},
        {"etf_ticker": "QQQ", "report_date": report_date, "filing_date": filing, "holding_id": "Q_OTHER", "issuer_name": "Other", "cusip": "CUSIP_Q", "isin": "Q_ISIN", "lei": None, "weight_pct": 90.0, "value_usd": 90, "source": "sec_nport", "retrieved_at": retrieved},
    ]
    df = pl.DataFrame(rows).cast(pl.Schema(dict(spec.columns)))
    stamped = stamp_availability(df, spec)

    monkeypatch.setattr("src.data.thesis_fundamentals.load_purity_spec", lambda **kwargs: purity_spec)
    # also patch the imported reference inside purity_evidence
    monkeypatch.setattr("src.analytics.purity_evidence.load_purity_spec", lambda **kwargs: purity_spec)

    def fake_load_visible(settings, dataset, decision_ts):
        if dataset == Dataset.ETF_HOLDINGS:
            return stamped
        raise ValueError("unexpected dataset")

    monkeypatch.setattr("src.data.catalog.load_visible", fake_load_visible)

    slot = compute_purity_slot(thesis=thesis, settings=settings, as_of=as_of)
    assert slot is not None
    assert slot.status == "computed"
    assert slot.summary.startswith("purity:")
    assert slot.metrics["purity_label"] == "pure"
    assert slot.metrics["thesis_aligned_weight_pct"] == pytest.approx(80.0)
    assert slot.metrics["incremental_weight_pct"] == pytest.approx(90.0, abs=0.01)
    assert slot.metrics["overlap_pct"] == pytest.approx(10.0, abs=0.01)

    # second case: aligned forced to 30 -> impure
    # Adjust GRID weight to 30 aligned, keep total 100: change aligned row weight to 30, add extra non-aligned 50
    rows2 = [
        {"etf_ticker": "GRID", "report_date": report_date, "filing_date": filing, "holding_id": "A1", "issuer_name": "Aligned Co", "cusip": "CUSIP_A1", "isin": aligned_isin, "lei": None, "weight_pct": 30.0, "value_usd": 30, "source": "sec_nport", "retrieved_at": retrieved},
        {"etf_ticker": "GRID", "report_date": report_date, "filing_date": filing, "holding_id": "SHARED", "issuer_name": "Shared Co", "cusip": "SHARED_CUSIP", "isin": None, "lei": None, "weight_pct": 10.0, "value_usd": 10, "source": "sec_nport", "retrieved_at": retrieved},
        {"etf_ticker": "GRID", "report_date": report_date, "filing_date": filing, "holding_id": "NON", "issuer_name": "NonAligned", "cusip": "CUSIP_NON", "isin": "NON_ISIN", "lei": None, "weight_pct": 60.0, "value_usd": 60, "source": "sec_nport", "retrieved_at": retrieved},
        {"etf_ticker": "QQQ", "report_date": report_date, "filing_date": filing, "holding_id": "SHARED", "issuer_name": "Shared Co", "cusip": "SHARED_CUSIP", "isin": None, "lei": None, "weight_pct": 10.0, "value_usd": 10, "source": "sec_nport", "retrieved_at": retrieved},
        {"etf_ticker": "QQQ", "report_date": report_date, "filing_date": filing, "holding_id": "Q_OTHER", "issuer_name": "Other", "cusip": "CUSIP_Q", "isin": "Q_ISIN", "lei": None, "weight_pct": 90.0, "value_usd": 90, "source": "sec_nport", "retrieved_at": retrieved},
    ]
    df2 = pl.DataFrame(rows2).cast(pl.Schema(dict(spec.columns)))
    stamped2 = stamp_availability(df2, spec)

    def fake_load_visible2(settings, dataset, decision_ts):
        if dataset == Dataset.ETF_HOLDINGS:
            return stamped2
        raise ValueError("unexpected dataset")

    monkeypatch.setattr("src.data.catalog.load_visible", fake_load_visible2)
    slot2 = compute_purity_slot(thesis=thesis, settings=settings, as_of=as_of)
    assert slot2 is not None
    assert slot2.metrics["purity_label"] == "impure"


def test_thesis_aligned_weight_pave_identifier_regression() -> None:
    """Exposure notes must use PAVE N-PORT ISINs (not approximate tickers)."""
    from src.analytics.purity_evidence import thesis_aligned_weight_pct
    from src.data.thesis_fundamentals import load_purity_spec

    spec = load_purity_spec(thesis_id=ThesisId.AI_POWER_BOTTLENECK)
    assert spec is not None
    snapshot = pl.DataFrame(
        {
            "weight_pct": [3.37, 1.50, 95.13],
            "isin": ["US74762E1029", "US4435106079", "US67066G1040"],
            "cusip": [None, None, "67066G104"],
            "holding_id": ["PWR", "HUB", "NVDA"],
        }
    )
    result = thesis_aligned_weight_pct(snapshot=snapshot, notes=spec.exposure_notes)
    assert result["thesis_aligned_weight_pct"] == pytest.approx(4.87, abs=0.01)
    assert result["matched_notes_count"] == 2


def test_compute_purity_slot_none_when_unconfigured(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from src.analytics.purity_evidence import compute_purity_slot

    thesis = ThesisSpec(
        id=ThesisId.AI_COMPUTE,
        version=1,
        title="test",
        status="research",
        horizon=Horizon(min_years=5, target_years=10),
        causal_chain=["a"],
        falsifiers=["f1"],
        candidate_sleeves=["ai_semiconductor"],
        historical_proxies=["SOXX"],
    )
    settings = DataSettings(data_root=tmp_path / "data")
    as_of = datetime(2020, 1, 15, tzinfo=UTC)
    monkeypatch.setattr("src.analytics.purity_evidence.load_purity_spec", lambda **kwargs: None)
    monkeypatch.setattr("src.data.thesis_fundamentals.load_purity_spec", lambda **kwargs: None)
    slot = compute_purity_slot(thesis=thesis, settings=settings, as_of=as_of)
    assert slot is None
