"""Tests for thesis fundamental registry and ingest wiring."""

from __future__ import annotations

from src.data.thesis_fundamentals import (
    FalsifierCollection,
    ThesisFundamentalsSpec,
    fundamental_series_ids,
    load_thesis_fundamentals,
)
from src.policy.thesis import ThesisId


def test_fund_load_ai_compute_registry() -> None:
  spec = load_thesis_fundamentals(thesis_id=ThesisId.AI_COMPUTE)
  assert spec.primary_series_id == "PNFI"
  assert "capex_structural_slowdown" in spec.falsifiers
  fals = spec.falsifiers["capex_structural_slowdown"]
  assert fals.consecutive_periods == 2
  ids = fundamental_series_ids(spec)
  assert ids == ("PNFI",)


def test_fund_series_ids_dedup_sorted() -> None:
  spec = ThesisFundamentalsSpec(
    thesis_id=ThesisId.AI_COMPUTE,
    primary_series_id="PNFI",
    secondary_series_ids=("IPG3344S", "PNFI"),
    falsifiers=FalsifierCollection(()),
    min_history_periods=8,
    lookback_periods=20,
  )
  ids = fundamental_series_ids(spec)
  assert ids == ("IPG3344S", "PNFI")
  assert spec.primary_series_id in ids


def test_thesis_fundamentals_valuation_spec_load() -> None:
    from src.data.thesis_fundamentals import load_crowding_spec, load_valuation_spec

    vspec = load_valuation_spec(thesis_id=ThesisId.AI_COMPUTE)
    assert vspec is not None
    assert vspec.vehicle_ticker == "SOXX"
    assert vspec.benchmark_ticker == "QQQ"
    assert vspec.trailing_sessions == 1260
    assert vspec.rich_percentile == 80
    assert vspec.cheap_percentile == 20
    cspec = load_crowding_spec(thesis_id=ThesisId.AI_COMPUTE)
    assert cspec is not None
    assert cspec.top_n == 5
    assert cspec.vehicle_ticker == "SOXX"


def test_load_ai_power_fundamentals_registry() -> None:
    spec = load_thesis_fundamentals(thesis_id=ThesisId.AI_POWER_BOTTLENECK)
    assert spec.primary_series_id == "A35SNO"
    assert "backlog_normalization" in spec.falsifiers
    fals = spec.falsifiers["backlog_normalization"]
    assert fals.series_id == "A35SNO"
    assert fals.metric == "yoy_pct"
    assert fals.threshold_pct == 0.0
    assert fals.consecutive_periods == 2
    ids = fundamental_series_ids(spec)
    assert "A35SNO" in ids
    assert "PNFI" in ids
    assert ids == tuple(sorted(ids))
    assert ids == ("A35SNO", "PNFI")


def test_load_ai_power_valuation_crowding_registry() -> None:
    from src.data.thesis_fundamentals import load_crowding_spec, load_valuation_spec

    vspec = load_valuation_spec(thesis_id=ThesisId.AI_POWER_BOTTLENECK)
    assert vspec is not None
    assert vspec.vehicle_ticker == "GRID"
    assert vspec.benchmark_ticker == "QQQ"
    assert vspec.trailing_sessions == 1260
    assert vspec.rich_percentile == 80
    assert vspec.cheap_percentile == 20
    assert vspec.min_sessions == 252
    assert vspec.return_lookback_sessions == 252
    assert vspec.collapse_return_pct == -15.0
    cspec = load_crowding_spec(thesis_id=ThesisId.AI_POWER_BOTTLENECK)
    assert cspec is not None
    assert cspec.vehicle_ticker == "GRID"
    assert cspec.top_n == 5
    assert cspec.concentrated_hhi_threshold == 0.18
    assert cspec.concentrated_top5_pct == 60.0
    spec = load_thesis_fundamentals(thesis_id=ThesisId.AI_POWER_BOTTLENECK)
    assert spec.primary_series_id == "A35SNO"


def test_load_ai_power_purity_spec() -> None:
    from src.data.thesis_fundamentals import load_purity_spec

    spec = load_purity_spec(thesis_id=ThesisId.AI_POWER_BOTTLENECK)
    assert spec is not None
    assert spec.vehicle_ticker == "GRID"
    assert spec.incumbent_ticker == "QQQ"
    assert spec.pure_min_pct == 70.0
    assert spec.impure_max_pct == 40.0
    assert len(spec.exposure_notes) >= 12
    for note in spec.exposure_notes:
        assert note.role.strip()
        assert note.note.strip()
        assert note.isin or note.cusip
    isins = {n.isin for n in spec.exposure_notes if n.isin}
    required = {
        "IE00B8KQN827",
        "CH0012221716",
        "FR0000121972",
        "IT0004176001",
        "US74762E1029",
        "GB00BDR05C01",
        "IT0003242622",
        "BE0003822393",
    }
    for req in required:
        assert req in isins
    # GRID N-PORT identifiers (regression: wrong ISIN silently drops aligned weight)
    grid_holdings_isins = {
        "US4435106079",  # Hubbell Incorporated
        "IE00BDVJJQ56",  # NVent Electric PLC
    }
    for req in grid_holdings_isins:
        assert req in isins
    dilution = {
        "US67066G1040",
        "US88160R1014",
        "US17275R1023",
        "US68389X1054",
        "DE0007164600",
    }
    for d in dilution:
        assert d not in isins
    assert load_purity_spec(thesis_id=ThesisId.AI_COMPUTE) is None
