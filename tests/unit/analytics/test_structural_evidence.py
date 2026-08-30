"""Tests for structural evidence from PIT fundamentals."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl
import pytest

from src.analytics.structural_evidence import (
    compute_structural_slot,
    detect_yoy_regime_change,
    evaluate_falsifier_slowdown,
    yoy_growth_pct,
)
from src.data.schema import Dataset
from src.data.settings import DataSettings
from src.policy.thesis import Horizon, ThesisId, ThesisSpec


def _quarterly_levels(values: list[float]) -> pl.DataFrame:
  quarter_ends = [
    date(2015, 3, 31),
    date(2015, 6, 30),
    date(2015, 9, 30),
    date(2015, 12, 31),
    date(2016, 3, 31),
    date(2016, 6, 30),
    date(2016, 9, 30),
    date(2016, 12, 31),
    date(2017, 3, 31),
    date(2017, 6, 30),
    date(2017, 9, 30),
    date(2017, 12, 31),
  ]
  dates = quarter_ends[: len(values)]
  return pl.DataFrame({"observation_date": dates, "value": values})


def test_str_yoy_quarterly_four_lag() -> None:
  levels = _quarterly_levels([100.0, 102.0, 104.0, 106.0, 110.0])
  yoy = yoy_growth_pct(levels, periods=4)
  last = float(yoy.tail(1).get_column("yoy_pct").to_list()[0])
  assert last == pytest.approx(10.0)


def test_str_falsifier_two_negative_yoy() -> None:
  yoy = pl.DataFrame(
    {
      "observation_date": [date(2020, 3, 31), date(2020, 6, 30), date(2020, 9, 30)],
      "yoy_pct": [1.0, -1.0, -2.0],
    }
  )
  assert evaluate_falsifier_slowdown(yoy=yoy, threshold_pct=0.0, consecutive_periods=2)
  single_neg = yoy.head(2)
  assert not evaluate_falsifier_slowdown(yoy=single_neg, threshold_pct=0.0, consecutive_periods=2)


def test_str_change_point_cross_below_zero() -> None:
  yoy = pl.DataFrame(
    {
      "observation_date": [date(2020, 3, 31), date(2020, 6, 30), date(2020, 9, 30), date(2020, 12, 31)],
      "yoy_pct": [5.0, 3.0, -1.0, -2.0],
    }
  )
  regime, cp_date = detect_yoy_regime_change(yoy=yoy, lookback_periods=20, min_positive_periods=2)
  assert regime == "slowdown"
  assert cp_date == date(2020, 9, 30)


def test_str_slot_computed_from_fixture(monkeypatch: pytest.MonkeyPatch) -> None:
  release = datetime(2025, 4, 29, 12, 0, tzinfo=UTC)
  obs_dates = [date(2015, 3, 31) + __import__("datetime").timedelta(days=91 * i) for i in range(12)]
  # use explicit quarterly ends
  obs_dates = [
    date(2015, 3, 31),
    date(2015, 6, 30),
    date(2015, 9, 30),
    date(2015, 12, 31),
    date(2016, 3, 31),
    date(2016, 6, 30),
    date(2016, 9, 30),
    date(2016, 12, 31),
    date(2017, 3, 31),
    date(2017, 6, 30),
    date(2017, 9, 30),
    date(2017, 12, 31),
  ]
  values = [3000.0 + i * 50.0 for i in range(12)]
  macro = pl.DataFrame(
    {
      "series_id": ["PNFI"] * 12,
      "observation_date": obs_dates,
      "release_date": [release] * 12,
      "value": values,
    }
  )

  def fake_load_visible(settings: DataSettings, dataset: Dataset, decision_ts: datetime) -> pl.DataFrame:
    if dataset == Dataset.MACRO:
      return macro
    raise ValueError("unexpected dataset")

  monkeypatch.setattr("src.analytics.structural_evidence.load_visible", fake_load_visible)

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
  settings = DataSettings(data_root="data")
  as_of = datetime(2025, 4, 30, tzinfo=UTC)
  slot = compute_structural_slot(thesis=thesis, settings=settings, as_of=as_of)
  assert slot.status == "computed"
  assert slot.summary.startswith("fundamental:")
  assert isinstance(slot.metrics["primary_yoy_pct"], float)
  assert isinstance(slot.metrics["falsifier_capex_structural_slowdown_active"], bool)


def test_str_slot_insufficient_no_macro(tmp_path: Path) -> None:
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
  settings = DataSettings(data_root=tmp_path / "empty_data")
  as_of = datetime(2025, 4, 30, tzinfo=UTC)
  slot = compute_structural_slot(thesis=thesis, settings=settings, as_of=as_of)
  assert slot.status == "insufficient_data"
  assert "error" in slot.metrics


def test_resolve_primary_falsifier_prefers_primary_series() -> None:
  from src.analytics.structural_evidence import resolve_primary_falsifier
  from src.data.thesis_fundamentals import load_thesis_fundamentals

  ai_power_spec = load_thesis_fundamentals(thesis_id=ThesisId.AI_POWER_BOTTLENECK)
  fals = resolve_primary_falsifier(spec=ai_power_spec)
  assert fals is not None
  assert fals.id == "backlog_normalization"
  assert fals.series_id == "A35SNO"

  ai_compute_spec = load_thesis_fundamentals(thesis_id=ThesisId.AI_COMPUTE)
  fals2 = resolve_primary_falsifier(spec=ai_compute_spec)
  assert fals2 is not None
  assert fals2.id == "capex_structural_slowdown"
  assert fals2.series_id == "PNFI"


def test_compute_structural_slot_ai_power_computed(monkeypatch: pytest.MonkeyPatch) -> None:
  import math

  release = datetime(2025, 4, 29, 12, 0, tzinfo=UTC)
  # monthly observations for A35SNO (>=20 to cover YoY lag 12)
  obs_dates = [date(2023, 1, 15 + 0), date(2023, 2, 15), date(2023, 3, 15), date(2023, 4, 15), date(2023, 5, 15), date(2023, 6, 15), date(2023, 7, 15), date(2023, 8, 15), date(2023, 9, 15), date(2023, 10, 15), date(2023, 11, 15), date(2023, 12, 15), date(2024, 1, 15), date(2024, 2, 15), date(2024, 3, 15), date(2024, 4, 15), date(2024, 5, 15), date(2024, 6, 15), date(2024, 7, 15), date(2024, 8, 15)]
  obs_dates = [date(2023, 1, 15), date(2023, 2, 15), date(2023, 3, 15), date(2023, 4, 15), date(2023, 5, 15), date(2023, 6, 15), date(2023, 7, 15), date(2023, 8, 15), date(2023, 9, 15), date(2023, 10, 15), date(2023, 11, 15), date(2023, 12, 15), date(2024, 1, 15), date(2024, 2, 15), date(2024, 3, 15), date(2024, 4, 15), date(2024, 5, 15), date(2024, 6, 15), date(2024, 7, 15), date(2024, 8, 15)]
  values = [100.0 + i * 5.0 for i in range(len(obs_dates))]
  macro = pl.DataFrame(
    {
      "series_id": ["A35SNO"] * len(obs_dates),
      "observation_date": obs_dates,
      "release_date": [release] * len(obs_dates),
      "value": values,
    }
  )

  def fake_load_visible(settings: DataSettings, dataset: Dataset, decision_ts: datetime) -> pl.DataFrame:
    if dataset == Dataset.MACRO:
      return macro
    raise ValueError("unexpected dataset")

  monkeypatch.setattr("src.analytics.structural_evidence.load_visible", fake_load_visible)

  thesis = ThesisSpec(
    id=ThesisId.AI_POWER_BOTTLENECK,
    version=1,
    title="test power",
    status="research",
    horizon=Horizon(min_years=5, target_years=10),
    causal_chain=["a"],
    falsifiers=["backlog_normalization"],
    candidate_sleeves=["ai_power_equipment"],
    historical_proxies=["GRID"],
  )
  settings = DataSettings(data_root="data")
  as_of = datetime(2025, 4, 30, tzinfo=UTC)
  slot = compute_structural_slot(thesis=thesis, settings=settings, as_of=as_of)
  assert slot.status == "computed"
  assert slot.summary.startswith("fundamental:")
  assert slot.metrics["primary_series_id"] == "A35SNO"
  assert math.isfinite(float(slot.metrics["primary_yoy_pct"]))
  assert isinstance(slot.metrics["falsifier_backlog_normalization_active"], bool)
  assert slot.metrics["regime"] in ("expansion", "contraction", "unknown", "slowdown")


def test_compute_structural_slot_ai_compute_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
  release = datetime(2025, 4, 29, 12, 0, tzinfo=UTC)
  obs_dates = [
    date(2015, 3, 31),
    date(2015, 6, 30),
    date(2015, 9, 30),
    date(2015, 12, 31),
    date(2016, 3, 31),
    date(2016, 6, 30),
    date(2016, 9, 30),
    date(2016, 12, 31),
    date(2017, 3, 31),
    date(2017, 6, 30),
    date(2017, 9, 30),
    date(2017, 12, 31),
  ]
  values = [3000.0 + i * 50.0 for i in range(12)]
  macro = pl.DataFrame(
    {
      "series_id": ["PNFI"] * 12,
      "observation_date": obs_dates,
      "release_date": [release] * 12,
      "value": values,
    }
  )

  def fake_load_visible(settings: DataSettings, dataset: Dataset, decision_ts: datetime) -> pl.DataFrame:
    if dataset == Dataset.MACRO:
      return macro
    raise ValueError("unexpected dataset")

  monkeypatch.setattr("src.analytics.structural_evidence.load_visible", fake_load_visible)

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
  settings = DataSettings(data_root="data")
  as_of = datetime(2025, 4, 30, tzinfo=UTC)
  slot = compute_structural_slot(thesis=thesis, settings=settings, as_of=as_of)
  assert slot.status == "computed"
  assert "falsifier_capex_structural_slowdown_active" in slot.metrics
  assert isinstance(slot.metrics["falsifier_capex_structural_slowdown_active"], bool)
