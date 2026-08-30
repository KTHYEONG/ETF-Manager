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
    assert "industrial_weight_pct" not in slot.metrics
    assert "humanoid_weight_pct" not in slot.metrics

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


def test_role_aligned_weight_pct_by_role() -> None:
    from src.analytics.purity_evidence import role_aligned_weight_pct, thesis_aligned_weight_pct

    snapshot = pl.DataFrame(
        {
            "weight_pct": [40.0, 10.0, 50.0],
            "isin": ["ISIN_I", "ISIN_H", "ISIN_X"],
            "cusip": [None, None, None],
            "holding_id": ["H1", "H2", "H3"],
        }
    )
    notes = (
        ExposureNote(isin="ISIN_I", cusip=None, role="industrial_automation", note="i"),
        ExposureNote(isin="ISIN_H", cusip=None, role="humanoid_optionality", note="h"),
    )
    role_weights = role_aligned_weight_pct(snapshot=snapshot, notes=notes)
    assert role_weights["industrial_automation"] == pytest.approx(40.0)
    assert role_weights["humanoid_optionality"] == pytest.approx(10.0)
    aligned = thesis_aligned_weight_pct(snapshot=snapshot, notes=notes)
    assert aligned["thesis_aligned_weight_pct"] == pytest.approx(50.0)
    assert aligned["matched_notes_count"] == 2


def test_compute_purity_slot_emits_split_metrics_when_roles_present(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from src.analytics.purity_evidence import compute_purity_slot
    from src.data.pit import stamp_availability

    thesis = ThesisSpec(
        id=ThesisId.PHYSICAL_AUTOMATION,
        version=1,
        title="test physical",
        status="research",
        horizon=Horizon(min_years=5, target_years=10),
        causal_chain=["a"],
        falsifiers=["commercialization_lag"],
        candidate_sleeves=["physical_automation"],
        historical_proxies=["BOTZ"],
    )
    settings = DataSettings(data_root=tmp_path / "data")
    as_of = datetime(2020, 1, 15, tzinfo=UTC)
    industrial_isin = "IND_ISIN"
    humanoid_isin = "HUM_ISIN"
    notes = (
        ExposureNote(isin=industrial_isin, cusip=None, role="industrial_automation", note="ind"),
        ExposureNote(isin=humanoid_isin, cusip=None, role="humanoid_optionality", note="hum"),
    )
    purity_spec = PuritySpec(
        vehicle_ticker="BOTZ",
        incumbent_ticker="QQQ",
        pure_min_pct=70.0,
        impure_max_pct=40.0,
        exposure_notes=notes,
    )
    spec = spec_for(Dataset.ETF_HOLDINGS)
    retrieved = datetime(2020, 1, 1, tzinfo=UTC)
    filing = datetime(2019, 12, 15, tzinfo=UTC)
    report_date = date(2019, 12, 31)
    rows = [
        {
            "etf_ticker": "BOTZ",
            "report_date": report_date,
            "filing_date": filing,
            "holding_id": "I1",
            "issuer_name": "Industrial Co",
            "cusip": "CUSIP_I",
            "isin": industrial_isin,
            "lei": None,
            "weight_pct": 45.0,
            "value_usd": 45,
            "source": "sec_nport",
            "retrieved_at": retrieved,
        },
        {
            "etf_ticker": "BOTZ",
            "report_date": report_date,
            "filing_date": filing,
            "holding_id": "H1",
            "issuer_name": "Humanoid Co",
            "cusip": "CUSIP_H",
            "isin": humanoid_isin,
            "lei": None,
            "weight_pct": 10.0,
            "value_usd": 10,
            "source": "sec_nport",
            "retrieved_at": retrieved,
        },
        {
            "etf_ticker": "BOTZ",
            "report_date": report_date,
            "filing_date": filing,
            "holding_id": "N1",
            "issuer_name": "Other Co",
            "cusip": "CUSIP_N",
            "isin": "NON_ISIN",
            "lei": None,
            "weight_pct": 45.0,
            "value_usd": 45,
            "source": "sec_nport",
            "retrieved_at": retrieved,
        },
        {
            "etf_ticker": "QQQ",
            "report_date": report_date,
            "filing_date": filing,
            "holding_id": "Q1",
            "issuer_name": "QQQ Only",
            "cusip": "CUSIP_Q",
            "isin": "Q_ISIN",
            "lei": None,
            "weight_pct": 100.0,
            "value_usd": 100,
            "source": "sec_nport",
            "retrieved_at": retrieved,
        },
    ]
    df = pl.DataFrame(rows).cast(pl.Schema(dict(spec.columns)))
    stamped = stamp_availability(df, spec)

    monkeypatch.setattr("src.data.thesis_fundamentals.load_purity_spec", lambda **kwargs: purity_spec)
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
    assert slot.metrics["purity_label"] == "mixed"
    assert slot.metrics["thesis_aligned_weight_pct"] == pytest.approx(55.0)
    assert slot.metrics["industrial_weight_pct"] == pytest.approx(45.0)
    assert slot.metrics["humanoid_weight_pct"] == pytest.approx(10.0)
    assert slot.metrics["vehicle_ticker"] == "BOTZ"
    assert slot.metrics["incremental_weight_pct"] == pytest.approx(100.0, abs=0.01)
    assert slot.metrics["industrial_weight_pct"] > slot.metrics["humanoid_weight_pct"]


def test_thesis_aligned_weight_botz_identifier_regression() -> None:
    from src.analytics.purity_evidence import thesis_aligned_weight_pct
    from src.data.thesis_fundamentals import load_purity_spec

    spec = load_purity_spec(thesis_id=ThesisId.PHYSICAL_AUTOMATION)
    assert spec is not None
    snapshot = pl.DataFrame(
        {
            "weight_pct": [10.03, 9.26, 9.51],
            "isin": ["CH0012221716", "JP3802400006", "US67066G1040"],
            "cusip": [None, None, "67066G104"],
            "holding_id": ["ABB", "FANUC", "NVDA"],
        }
    )
    result = thesis_aligned_weight_pct(snapshot=snapshot, notes=spec.exposure_notes)
    assert result["thesis_aligned_weight_pct"] == pytest.approx(19.29, abs=0.01)
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
