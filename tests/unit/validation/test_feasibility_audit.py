"""Unit tests for wave2 feasibility audit."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl
import pytest

from src.data.calendar import TradingCalendar, load_calendar
from src.data.pipeline import persist_ingest
from src.data.schema import Dataset, spec_for
from src.data.settings import DataSettings
from src.data.storage import RawPayload
from src.validation.experiment import ExperimentSpec, load_experiment_config
from src.validation.feasibility_audit import (
    audit_static_dca_window,
    resolve_dependency_profile,
)

_CALENDAR = load_calendar("XNYS")
_RETRIEVED_AT = datetime(2022, 6, 1, 5, 0, tzinfo=UTC)


def _payload() -> RawPayload:
    return RawPayload(
        provider="synthetic",
        endpoint="synthetic",
        request_params={},
        retrieved_at=_RETRIEVED_AT,
        extension="json",
        content=b"{}",
    )


def _prices_frame(days: tuple[date, ...], tickers: tuple[str, ...]) -> pl.DataFrame:
    spec = spec_for(Dataset.PRICES)
    rows_ticker: list[str] = []
    rows_date: list[date] = []
    closes: list[float] = []
    for ticker in tickers:
        for day in days:
            rows_ticker.append(ticker)
            rows_date.append(day)
            closes.append(100.0)
    n = len(rows_date)
    return pl.DataFrame(
        {
            "ticker": rows_ticker,
            "date": rows_date,
            "open": [c * 0.98 for c in closes],
            "high": [c * 1.02 for c in closes],
            "low": [c * 0.97 for c in closes],
            "close": closes,
            "volume": [10_000] * n,
            "adjusted_close": closes,
            "dividend": [0.0] * n,
            "split_factor": [1.0] * n,
            "source": ["synthetic"] * n,
            "retrieved_at": [_RETRIEVED_AT] * n,
        },
        schema=dict(spec.columns),
    )


def _fx_frame(days: tuple[date, ...]) -> pl.DataFrame:
    spec = spec_for(Dataset.FX)
    return pl.DataFrame(
        {"date": list(days), "usdkrw": [1300.0] * len(days), "source": ["synthetic"] * len(days), "retrieved_at": [_RETRIEVED_AT] * len(days)},
        schema=dict(spec.columns),
    )


def _cpi_frame(period_end: date) -> pl.DataFrame:
    spec = spec_for(Dataset.CPI)
    return pl.DataFrame({"period_end": [period_end], "value": [100.0], "source": ["synthetic"], "retrieved_at": [_RETRIEVED_AT]}, schema=dict(spec.columns))


def _catalog(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, days: tuple[date, ...], tickers: tuple[str, ...], cpi_period: date, name: str = "catalog") -> DataSettings:
    root = tmp_path / name
    root.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(root)
    settings = DataSettings(data_root="data")
    persist_ingest(_prices_frame(days, tickers), Dataset.PRICES, _payload(), settings)
    persist_ingest(_fx_frame(days), Dataset.FX, _payload(), settings)
    persist_ingest(_cpi_frame(cpi_period), Dataset.CPI, _payload(), settings)
    return settings


@pytest.mark.parametrize("scenario_id", ["WAV2-AUD-static-deps"])
def test_WAV2_AUD_static_deps(scenario_id: str) -> None:  # noqa: N802
    """WAV2-AUD-static-deps"""
    spec = load_experiment_config("configs/experiments/m_qqq_grid.json")
    profile = resolve_dependency_profile(spec)
    assert profile.profile == "static_dca"
    assert profile.requires_macro is False
    assert profile.required_datasets == ("prices", "fx", "cpi")


@pytest.mark.parametrize("scenario_id", ["WAV2-AUD-macro-deps"])
def test_WAV2_AUD_macro_deps(scenario_id: str) -> None:  # noqa: N802
    """WAV2-AUD-macro-deps"""
    spec = ExperimentSpec.model_validate(
        {
            "name": "macro_test",
            "start": "2024-01-01",
            "end": "2024-12-31",
            "contribution_krw": 1_000_000,
            "hurdle": 0.02,
            "horizon_months": 0,
            "objective": "adaptive_growth",
            "adaptive_contribution": {},
            "baseline": {"id": "m0", "policy": "s0_global", "modules": 0},
            "candidates": [{"id": "c1", "policy": "s1_us", "modules": 1}],
        }
    )
    profile = resolve_dependency_profile(spec)
    assert profile.requires_macro is True


def _month_ends(start: date, end: date) -> tuple[date, ...]:
    import calendar as _cal

    cur = date(start.year, start.month, 1)
    out: list[date] = []
    while cur <= end:
        last_day = _cal.monthrange(cur.year, cur.month)[1]
        me = date(cur.year, cur.month, last_day)
        if start <= me <= end:
            out.append(me)
        cur = date(cur.year + 1, 1, 1) if cur.month == 12 else date(cur.year, cur.month + 1, 1)
    return tuple(out)


@pytest.mark.parametrize("scenario_id", ["WAV2-AUD-grid-limit"])
def test_WAV2_AUD_grid_limit(scenario_id: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: N802
    """WAV2-AUD-grid-limit"""
    orig_close = TradingCalendar.close_ts
    orig_sessions = TradingCalendar.sessions
    orig_month_ends = TradingCalendar.month_end_sessions

    def _fake_close(self, session: date) -> datetime:
        try:
            return orig_close(self, session)
        except Exception:
            return datetime(session.year, session.month, session.day, 20, 0, tzinfo=UTC)

    def _fake_sessions(self, start: date, end: date) -> tuple[date, ...]:
        try:
            return orig_sessions(self, start, end)
        except Exception:
            cur = start
            out: list[date] = []
            from datetime import timedelta

            while cur <= end:
                out.append(cur)
                cur += timedelta(days=1)
            return tuple(out)

    def _fake_month_ends(self, start: date, end: date) -> tuple[date, ...]:
        try:
            return orig_month_ends(self, start, end)
        except Exception:
            return _month_ends(start, end)

    monkeypatch.setattr(TradingCalendar, "close_ts", _fake_close)
    monkeypatch.setattr(TradingCalendar, "sessions", _fake_sessions)
    monkeypatch.setattr(TradingCalendar, "month_end_sessions", _fake_month_ends)
    import src.data.quality as _q
    monkeypatch.setattr(_q, "_session_missing_finding", lambda *a, **k: None)
    # GRID listing 2009-11-17; requested_start 2000-01-01 should trigger grid_listing
    grid_first = date(2009, 11, 17)
    all_days = _month_ends(date(2000, 1, 1), date(2025, 6, 30))
    grid_days = _month_ends(grid_first, date(2025, 6, 30))
    qqq_days = all_days
    # Build combined price frame: QQQ for all_days, GRID for grid_days only
    spec = spec_for(Dataset.PRICES)
    rows_ticker: list[str] = []
    rows_date: list[date] = []
    for d in qqq_days:
        rows_ticker.append("QQQ")
        rows_date.append(d)
    for d in grid_days:
        rows_ticker.append("GRID")
        rows_date.append(d)
    n = len(rows_date)
    closes = [100.0] * n
    frame = pl.DataFrame(
        {
            "ticker": rows_ticker,
            "date": rows_date,
            "open": [c * 0.98 for c in closes],
            "high": [c * 1.02 for c in closes],
            "low": [c * 0.97 for c in closes],
            "close": closes,
            "volume": [10_000] * n,
            "adjusted_close": closes,
            "dividend": [0.0] * n,
            "split_factor": [1.0] * n,
            "source": ["synthetic"] * n,
            "retrieved_at": [_RETRIEVED_AT] * n,
        },
        schema=dict(spec.columns),
    )
    root = tmp_path / "grid_catalog"
    root.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(root)
    settings = DataSettings(data_root="data")
    persist_ingest(frame, Dataset.PRICES, _payload(), settings)
    # FX and CPI must cover from 2000
    persist_ingest(_fx_frame(all_days), Dataset.FX, _payload(), settings)
    persist_ingest(_cpi_frame(date(2023, 11, 1)), Dataset.CPI, _payload(), settings)

    spec2 = ExperimentSpec.model_validate(
        {
            "name": "grid_test",
            "start": "2000-01-01",
            "end": "2025-06-30",
            "contribution_krw": 1_000_000,
            "hurdle": 0.02,
            "horizon_months": 0,
            "baseline": {"id": "qqq_baseline", "policy": "qqq", "modules": 0},
            "candidates": [{"id": "qqq_grid_05", "policy": "qqq", "modules": 1, "targets": {"QQQ": 0.95, "GRID": 0.05}}],
        }
    )
    report = audit_static_dca_window(spec2, settings)
    assert "grid_listing" in report.limiting_factors


@pytest.mark.parametrize("scenario_id", ["WAV2-AUD-cohort-count"])
def test_WAV2_AUD_cohort_count(scenario_id: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: N802
    """WAV2-AUD-cohort-count"""
    orig_close = TradingCalendar.close_ts
    orig_sessions = TradingCalendar.sessions
    orig_month_ends = TradingCalendar.month_end_sessions

    def _fake_close(self, session: date) -> datetime:
        try:
            return orig_close(self, session)
        except Exception:
            return datetime(session.year, session.month, session.day, 20, 0, tzinfo=UTC)

    def _fake_sessions(self, start: date, end: date) -> tuple[date, ...]:
        try:
            return orig_sessions(self, start, end)
        except Exception:
            cur = start
            out: list[date] = []
            from datetime import timedelta

            while cur <= end:
                out.append(cur)
                cur += timedelta(days=1)
            return tuple(out)

    def _fake_month_ends(self, start: date, end: date) -> tuple[date, ...]:
        try:
            return orig_month_ends(self, start, end)
        except Exception:
            return _month_ends(start, end)

    monkeypatch.setattr(TradingCalendar, "close_ts", _fake_close)
    monkeypatch.setattr(TradingCalendar, "sessions", _fake_sessions)
    monkeypatch.setattr(TradingCalendar, "month_end_sessions", _fake_month_ends)
    import src.data.quality as _q2
    monkeypatch.setattr(_q2, "_session_missing_finding", lambda *a, **k: None)
    days = _month_ends(date(2000, 1, 1), date(2025, 6, 30))
    settings = _catalog(tmp_path, monkeypatch, days, ("QQQ",), date(1999, 11, 1), name="cohort_catalog")
    spec = ExperimentSpec.model_validate(
        {
            "name": "cohort_test",
            "start": "2000-01-01",
            "end": "2025-06-30",
            "contribution_krw": 1_000_000,
            "hurdle": 0.02,
            "horizon_months": 0,
            "baseline": {"id": "qqq_baseline", "policy": "qqq", "modules": 0},
            "candidates": [{"id": "qqq_dummy", "policy": "qqq", "modules": 1}],
        }
    )
    report = audit_static_dca_window(spec, settings)
    assert report.cohort_count_120m_step12 >= 10
