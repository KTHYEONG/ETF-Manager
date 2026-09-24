"""Registry-backed structural slot coverage (path-constant wiring)."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl
import pytest

from src.analytics.thesis.structural import compute_structural_slot
from src.data.schema import Dataset
from src.data.settings import DataSettings
from src.policy.thesis import Horizon, ThesisId, ThesisSpec


def test_structural_slot_reads_registry_constants(monkeypatch: pytest.MonkeyPatch) -> None:
  """Computed slot passes through the THESIS_FUNDAMENTALS_DIR registry reads."""
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

  monkeypatch.setattr("src.analytics.thesis.structural.load_visible", fake_load_visible)

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
  assert (Path("configs/data/thesis_fundamentals") / "ai_compute.json").is_file()
