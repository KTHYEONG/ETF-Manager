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
