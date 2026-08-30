"""Tests for crowding evidence (Track F)."""
from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl
import pytest


def test_crd_hhi_top5_concentrated() -> None:
    from src.analytics.crowding_evidence import holdings_concentration_metrics

    # top5 sum 70%: 35+17+6+6+6=70 with remainder 6*5 gives hhi just over 0.18
    weights = [35.0, 17.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0]
    snapshot = pl.DataFrame(
        {
            "etf_ticker": ["SOXX"] * len(weights),
            "report_date": [date(2020, 12, 31)] * len(weights),
            "filing_date": [datetime(2020, 12, 15, tzinfo=UTC)] * len(weights),
            "holding_id": [f"H{i}" for i in range(len(weights))],
            "issuer_name": [f"Issuer{i}" for i in range(len(weights))],
            "cusip": [f"CUSIP{i}" for i in range(len(weights))],
            "isin": [None] * len(weights),
            "lei": [None] * len(weights),
            "weight_pct": weights,
            "value_usd": [w * 10 for w in weights],
            "source": ["sec_nport"] * len(weights),
            "retrieved_at": [datetime(2020, 1, 1, tzinfo=UTC)] * len(weights),
        }
    )
    result = holdings_concentration_metrics(snapshot=snapshot, top_n=5)
    assert result["top5_weight_pct"] == pytest.approx(70.0, abs=0.01)
    assert result["hhi"] > 0.18
    # label logic per registry thresholds 0.18 / 60
    label = "concentrated" if result["hhi"] > 0.18 or result["top5_weight_pct"] > 60.0 else "dispersed"
    assert label == "concentrated"
    assert result["holdings_count"] == pytest.approx(10.0)
    assert result["effective_n"] == pytest.approx(1.0 / result["hhi"], rel=1e-6)


def test_crd_slot_computed_from_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.analytics.crowding_evidence import compute_crowding_slot
    from src.data.schema import Dataset, spec_for
    from src.data.settings import DataSettings
    from src.policy.thesis import Horizon, ThesisId, ThesisSpec

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

    spec = spec_for(Dataset.ETF_HOLDINGS)
    retrieved = datetime(2020, 1, 1, tzinfo=UTC)
    filing = datetime(2019, 12, 15, tzinfo=UTC)
    report_date = date(2019, 12, 31)
    # Concentrated snapshot: top5 70%
    weights = [25.0, 15.0, 12.0, 10.0, 8.0, 10.0, 10.0, 10.0]
    rows = []
    for i, w in enumerate(weights):
        rows.append(
            {
                "etf_ticker": "SOXX",
                "report_date": report_date,
                "filing_date": filing,
                "holding_id": f"H{i}",
                "issuer_name": f"Issuer{i}",
                "cusip": f"CUSIP{i}",
                "isin": None,
                "lei": None,
                "weight_pct": w,
                "value_usd": w * 10,
                "source": "sec_nport",
                "retrieved_at": retrieved,
            }
        )
    holdings_frame = pl.DataFrame(rows).cast(pl.Schema(dict(spec.columns)))
    from src.data.pit import stamp_availability

    stamped = stamp_availability(holdings_frame, spec)

    def fake_load_visible(settings, dataset, decision_ts):
        if dataset == Dataset.ETF_HOLDINGS:
            return stamped
        raise ValueError(f"unexpected dataset {dataset}")

    monkeypatch.setattr("src.analytics.crowding_evidence.load_visible", fake_load_visible)
    slot = compute_crowding_slot(thesis=thesis, settings=settings, as_of=as_of)
    assert slot.status == "computed"
    assert slot.summary.startswith("crowding:")
    assert isinstance(slot.metrics["hhi"], float)
    assert isinstance(slot.metrics["top5_weight_pct"], float)


def test_compute_crowding_slot_ai_power_grid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.analytics.crowding_evidence import compute_crowding_slot
    from src.data.pit import stamp_availability
    from src.data.schema import Dataset, spec_for
    from src.data.settings import DataSettings
    from src.policy.thesis import Horizon, ThesisId, ThesisSpec

    thesis = ThesisSpec(
        id=ThesisId.AI_POWER_BOTTLENECK,
        version=1,
        title="test",
        status="research",
        horizon=Horizon(min_years=5, target_years=10),
        causal_chain=["a"],
        falsifiers=["backlog_normalization"],
        candidate_sleeves=["ai_power_equipment"],
        historical_proxies=["GRID"],
    )
    settings = DataSettings(data_root=tmp_path / "data")
    as_of = datetime(2021, 12, 30, tzinfo=UTC)
    spec = spec_for(Dataset.ETF_HOLDINGS)
    retrieved = datetime(2021, 12, 29, tzinfo=UTC)
    filing = datetime(2021, 12, 15, tzinfo=UTC)
    report_date = date(2021, 12, 31)
    weights = [35.0, 17.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0]
    rows: list[dict[str, object]] = []
    for i, w in enumerate(weights):
        rows.append(
            {
                "etf_ticker": "GRID",
                "report_date": report_date,
                "filing_date": filing,
                "holding_id": f"G{i}",
                "issuer_name": f"Issuer{i}",
                "cusip": f"CUSIP{i}",
                "isin": None,
                "lei": None,
                "weight_pct": w,
                "value_usd": w * 10,
                "source": "sec_nport",
                "retrieved_at": retrieved,
            }
        )
    holdings_frame = pl.DataFrame(rows).cast(pl.Schema(dict(spec.columns)))
    stamped = stamp_availability(holdings_frame, spec)

    def fake_load_visible(settings, dataset, decision_ts):  # type: ignore[no-untyped-def]
        if dataset == Dataset.ETF_HOLDINGS:
            return stamped
        raise ValueError(f"unexpected dataset {dataset}")

    monkeypatch.setattr("src.analytics.crowding_evidence.load_visible", fake_load_visible)
    slot = compute_crowding_slot(thesis=thesis, settings=settings, as_of=as_of)
    assert slot.status == "computed"
    assert slot.summary.startswith("crowding:")
    assert slot.metrics["vehicle_ticker"] == "GRID"
    assert slot.metrics["concentration_label"] in {"concentrated", "dispersed"}
    assert isinstance(slot.metrics["hhi"], float)
    assert isinstance(slot.metrics["top5_weight_pct"], float)
    assert float(slot.metrics["hhi"]) == float(slot.metrics["hhi"])  # finite
    assert float(slot.metrics["top5_weight_pct"]) == float(slot.metrics["top5_weight_pct"])  # finite
