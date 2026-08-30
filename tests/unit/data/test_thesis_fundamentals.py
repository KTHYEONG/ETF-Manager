"""Tests for thesis fundamental registry and ingest wiring."""

from __future__ import annotations

import pytest

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
    assert vspec.vehicle_ticker == "PAVE"
    assert vspec.benchmark_ticker == "QQQ"
    assert vspec.trailing_sessions == 1260
    assert vspec.rich_percentile == 80
    assert vspec.cheap_percentile == 20
    assert vspec.min_sessions == 252
    assert vspec.return_lookback_sessions == 252
    assert vspec.collapse_return_pct == -15.0
    cspec = load_crowding_spec(thesis_id=ThesisId.AI_POWER_BOTTLENECK)
    assert cspec is not None
    assert cspec.vehicle_ticker == "PAVE"
    assert cspec.top_n == 5
    assert cspec.concentrated_hhi_threshold == 0.18
    assert cspec.concentrated_top5_pct == 60.0
    spec = load_thesis_fundamentals(thesis_id=ThesisId.AI_POWER_BOTTLENECK)
    assert spec.primary_series_id == "A35SNO"


def test_load_ai_power_purity_spec() -> None:
    from src.data.thesis_fundamentals import load_purity_spec

    spec = load_purity_spec(thesis_id=ThesisId.AI_POWER_BOTTLENECK)
    assert spec is not None
    assert spec.vehicle_ticker == "PAVE"
    assert spec.incumbent_ticker == "QQQ"
    assert spec.pure_min_pct == 70.0
    assert spec.impure_max_pct == 40.0
    assert len(spec.exposure_notes) >= 8
    for note in spec.exposure_notes:
        assert note.role.strip()
        assert note.note.strip()
        assert note.isin or note.cusip
    isins = {n.isin for n in spec.exposure_notes if n.isin}
    required = {
        "IE00B8KQN827",
        "IE00BK9ZQ967",
        "IE0001827041",
        "US2441991054",
        "US29084Q1004",
        "US2910111044",
        "US4435106079",
        "US5763231090",
        "US74762E1029",
        "US7587501039",
        "US8168511090",
        "US9291601097",
    }
    for req in required:
        assert req in isins
    # PAVE N-PORT identifiers (regression: wrong ISIN silently drops aligned weight)
    pave_holdings_isins = {
        "US4435106079",  # Hubbell Incorporated
        "US74762E1029",  # Quanta Services
    }
    for req in pave_holdings_isins:
        assert req in isins
    dilution = {
        "US67066G1040",
        "US88160R1014",
        "US68389X1054",
        "DE0007164600",
    }
    for d in dilution:
        assert d not in isins
    assert load_purity_spec(thesis_id=ThesisId.AI_COMPUTE) is None


@pytest.mark.parametrize("scenario_id", ["test_load_ai_power_fundamentals_retarget_pave"])
def test_load_ai_power_fundamentals_retarget_pave(scenario_id: str) -> None:
    """test_load_ai_power_fundamentals_retarget_pave"""
    from src.data.thesis_fundamentals import load_crowding_spec, load_purity_spec, load_valuation_spec

    vspec = load_valuation_spec(thesis_id=ThesisId.AI_POWER_BOTTLENECK)
    cspec = load_crowding_spec(thesis_id=ThesisId.AI_POWER_BOTTLENECK)
    pspec = load_purity_spec(thesis_id=ThesisId.AI_POWER_BOTTLENECK)
    assert vspec is not None
    assert vspec.vehicle_ticker == "PAVE"
    assert cspec is not None
    assert cspec.vehicle_ticker == "PAVE"
    assert pspec is not None
    assert pspec.vehicle_ticker == "PAVE"
    assert pspec.incumbent_ticker == "QQQ"
    assert len(pspec.exposure_notes) >= 8
    allowed_roles = {"grid_equipment", "grid_contractor", "cable_manufacturer", "transmission_operator", "infrastructure_materials", "electrical_equipment"}
    for note in pspec.exposure_notes:
        assert note.role in allowed_roles
    isins = {n.isin for n in pspec.exposure_notes if n.isin}
    assert "US74762E1029" in isins


@pytest.mark.parametrize("scenario_id", ["test_nport_series_map_includes_pave"])
def test_nport_series_map_includes_pave(scenario_id: str) -> None:
    """test_nport_series_map_includes_pave"""
    import json
    from pathlib import Path

    mapping = json.loads(Path("configs/etf_metadata/nport_series_map.json").read_text(encoding="utf-8"))
    assert mapping["S000056509"] == "PAVE"
    assert mapping["S000026919"] == "GRID"
    assert mapping["S000004354"] == "SOXX"
