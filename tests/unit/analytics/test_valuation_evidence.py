"""Tests for valuation evidence (Track F)."""
from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest


def _make_price_frame(ticker: str, start_price: float, growth_per_day: float, n: int, start_date: date) -> pl.DataFrame:
    dates = [start_date + timedelta(days=i) for i in range(n)]
    prices = [start_price * ((1 + growth_per_day) ** i) for i in range(n)]
    return pl.DataFrame(
        {
            "ticker": [ticker] * n,
            "date": dates,
            "close": prices,
            "adjusted_close": prices,
            "volume": [1000] * n,
            "open": prices,
            "high": prices,
            "low": prices,
            "dividend": [0.0] * n,
            "split_factor": [1.0] * n,
            "source": ["test"] * n,
            "retrieved_at": [datetime(2020, 1, 1, tzinfo=UTC)] * n,
        }
    )


def test_val_relative_richness_monotone() -> None:
    from src.analytics.valuation_evidence import relative_richness_percentile

    start = date(2020, 1, 1)
    n = 200
    # Rising relative: vehicle grows faster than benchmark
    vehicle = pl.DataFrame(
        {"date": [start + timedelta(days=i) for i in range(n)], "adjusted_close": [100 * (1.01 ** i) for i in range(n)]}
    )
    benchmark = pl.DataFrame(
        {"date": [start + timedelta(days=i) for i in range(n)], "adjusted_close": [100 * (1.001 ** i) for i in range(n)]}
    )
    ratio, pctile = relative_richness_percentile(vehicle=vehicle, benchmark=benchmark, trailing_sessions=100)
    assert pctile > 80
    label = "rich" if pctile >= 80 else "cheap" if pctile <= 20 else "fair"
    assert label == "rich"

    # Falling relative: vehicle grows slower (or falls) than benchmark
    vehicle2 = pl.DataFrame(
        {"date": [start + timedelta(days=i) for i in range(n)], "adjusted_close": [100 * (0.999 ** i) for i in range(n)]}
    )
    benchmark2 = pl.DataFrame(
        {"date": [start + timedelta(days=i) for i in range(n)], "adjusted_close": [100 * (1.001 ** i) for i in range(n)]}
    )
    ratio2, pctile2 = relative_richness_percentile(vehicle=vehicle2, benchmark=benchmark2, trailing_sessions=100)
    assert pctile2 < 20
    label2 = "rich" if pctile2 >= 80 else "cheap" if pctile2 <= 20 else "fair"
    assert label2 == "cheap"


def test_val_pricing_collapse_falsifier(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.analytics.valuation_evidence import trailing_total_return_pct, compute_valuation_slot
    from src.data.schema import Dataset, spec_for
    from src.data.settings import DataSettings
    from src.policy.thesis import Horizon, ThesisId, ThesisSpec

    # -20% drawdown series
    start = date(2020, 1, 1)
    n = 300
    # create price dropping 20% over 252 sessions: linear decay to 80
    prices = [100 - (20 * i / 252) for i in range(n)]
    # Ensure last ~252 lookback yields approx -20
    series = pl.DataFrame({"date": [start + timedelta(days=i) for i in range(n)], "adjusted_close": prices})
    ret = trailing_total_return_pct(series=series, lookback_sessions=252)
    assert ret == pytest.approx(-20, abs=2.0)

    # Now test falsifier via compute_valuation_slot with mocked PRICES
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
    as_of = datetime(2021, 6, 1, tzinfo=UTC)

    # Build PRICES visible frame with SOXX drawdown and QQQ flat
    spec = spec_for(Dataset.PRICES)
    qqq_prices = [100.0] * n
    soxx_prices = prices
    rows = []
    retrieved = datetime(2021, 5, 31, tzinfo=UTC)
    for i in range(n):
        d = start + timedelta(days=i)
        rows.append(
            {
                "ticker": "SOXX",
                "date": d,
                "open": soxx_prices[i],
                "high": soxx_prices[i],
                "low": soxx_prices[i],
                "close": soxx_prices[i],
                "volume": 1000,
                "adjusted_close": soxx_prices[i],
                "dividend": 0.0,
                "split_factor": 1.0,
                "source": "test",
                "retrieved_at": retrieved,
            }
        )
        rows.append(
            {
                "ticker": "QQQ",
                "date": d,
                "open": 100.0,
                "high": 100.0,
                "low": 100.0,
                "close": 100.0,
                "volume": 1000,
                "adjusted_close": 100.0,
                "dividend": 0.0,
                "split_factor": 1.0,
                "source": "test",
                "retrieved_at": retrieved,
            }
        )
    df = pl.DataFrame(rows).cast(pl.Schema(dict(spec.columns)))

    def fake_load_visible(settings, dataset, decision_ts):
        if dataset == Dataset.PRICES:
            return df
        raise ValueError(f"unexpected dataset {dataset}")

    monkeypatch.setattr("src.analytics.valuation_evidence.load_visible", fake_load_visible)
    slot = compute_valuation_slot(thesis=thesis, settings=settings, as_of=as_of)
    assert slot.status == "computed"
    assert slot.metrics["falsifier_pricing_collapse_active"] is True
    assert "falsifier_semiconductor_pricing_collapse_active" not in slot.metrics


def test_compute_valuation_slot_ai_power_grid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.analytics.valuation_evidence import compute_valuation_slot
    from src.data.calendar import load_calendar
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
        historical_proxies=["PAVE"],
    )
    settings = DataSettings(data_root=tmp_path / "data")
    as_of = datetime(2021, 12, 30, tzinfo=UTC)
    spec_prices = spec_for(Dataset.PRICES)
    cal = load_calendar("XNYS")
    sessions = cal.sessions(date(2020, 1, 1), date(2021, 12, 31))
    n = 300
    assert len(sessions) >= n
    price_dates = list(sessions[:n])
    grid_prices = [100.0 * (1.001 ** i) for i in range(n)]
    qqq_prices = [100.0] * n
    retrieved = datetime(2021, 12, 29, tzinfo=UTC)
    rows: list[dict[str, object]] = []
    for i, d in enumerate(price_dates):
        for ticker, px in [("PAVE", grid_prices[i]), ("QQQ", qqq_prices[i])]:
            rows.append(
                {
                    "ticker": ticker,
                    "date": d,
                    "open": px,
                    "high": px,
                    "low": px,
                    "close": px,
                    "volume": 1000,
                    "adjusted_close": px,
                    "dividend": 0.0,
                    "split_factor": 1.0,
                    "source": "test",
                    "retrieved_at": retrieved,
                }
            )
    df = pl.DataFrame(rows).cast(pl.Schema(dict(spec_prices.columns)))
    stamped = stamp_availability(df, spec_prices, cal)

    def fake_load_visible(settings, dataset, decision_ts):  # type: ignore[no-untyped-def]
        if dataset == Dataset.PRICES:
            return stamped
        raise ValueError(f"unexpected dataset {dataset}")

    monkeypatch.setattr("src.analytics.valuation_evidence.load_visible", fake_load_visible)
    slot = compute_valuation_slot(thesis=thesis, settings=settings, as_of=as_of)
    assert slot.status == "computed"
    assert slot.summary.startswith("valuation:")
    assert slot.metrics["vehicle_ticker"] == "PAVE"
    assert slot.metrics["benchmark_ticker"] == "QQQ"
    assert slot.metrics["richness_label"] in {"rich", "fair", "cheap"}
    assert isinstance(slot.metrics["falsifier_pricing_collapse_active"], bool)
    assert "falsifier_semiconductor_pricing_collapse_active" not in slot.metrics


def test_val_slot_unknown_without_registry(tmp_path: Path) -> None:
    from src.analytics.valuation_evidence import compute_valuation_slot
    from src.data.settings import DataSettings
    from src.policy.thesis import Horizon, ThesisId, ThesisSpec

    thesis = ThesisSpec(
        id=ThesisId.PHYSICAL_AUTOMATION,
        version=1,
        title="test",
        status="research",
        horizon=Horizon(min_years=5, target_years=10),
        causal_chain=["a"],
        falsifiers=["f1"],
        candidate_sleeves=["physical_automation"],
        historical_proxies=["BOTZ"],
    )
    settings = DataSettings(data_root=tmp_path / "data")
    as_of = datetime(2025, 4, 30, tzinfo=UTC)
    slot = compute_valuation_slot(thesis=thesis, settings=settings, as_of=as_of)
    assert slot.status == "unknown"
    assert "not configured" in slot.summary
