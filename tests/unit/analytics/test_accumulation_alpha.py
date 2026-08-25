"""Unit tests for the reporting-only QQQ accumulation screen."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from src.analytics.accumulation_alpha import (
    AccumulationScreenReport,
    ArmScreenRow,
    ArmVerdict,
    screen_qqq_accumulation,
)
from src.data.calendar import load_calendar

_SCREEN_START = date(2021, 1, 4)
_SCREEN_END = date(2021, 12, 31)
# 월별 성장률을 불균등하게 둬서 paired log-fill 통계에 유의미한 분산이 생기도록 한다.
_MONTH_GROWTH: tuple[float, ...] = (
    1.006,
    1.014,
    1.002,
    1.011,
    1.005,
    1.013,
    1.003,
    1.009,
    1.007,
    1.012,
    1.004,
    1.008,
)
# 인월 최저가가 세션 인덱스 4에서 유일하게 나오는 V-복원 형상 (모든 값 > 0.98 -> dip 미발동).
_K4_SHAPE: tuple[float, ...] = (
    1.0,
    0.998,
    0.996,
    0.995,
    0.987,
    0.992,
    0.995,
    1.0,
    1.006,
    1.012,
)


def _months(start: date, end: date) -> list[list[date]]:
    grouped: dict[tuple[int, int], list[date]] = {}
    for day in load_calendar("XNYS").sessions(start, end):
        grouped.setdefault((day.year, day.month), []).append(day)
    return list(grouped.values())


def _frame(days: list[date], closes: list[float]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ticker": ["QQQ"] * len(days),
            "date": days,
            "close": closes,
            "adjusted_close": closes,
        }
    )


def _monotone_prices() -> pl.DataFrame:
    days: list[date] = []
    closes: list[float] = []
    level = 100.0
    for index, month_days in enumerate(_months(_SCREEN_START, _SCREEN_END)):
        growth = _MONTH_GROWTH[index % len(_MONTH_GROWTH)]
        steps = len(month_days)
        for offset, day in enumerate(month_days):
            days.append(day)
            closes.append(level * growth ** ((offset + 1) / steps))
        level *= growth
    return _frame(days, closes)


def _k4_prices() -> pl.DataFrame:
    start = date(2021, 1, 4)
    end = date(2021, 10, 31)
    days: list[date] = []
    closes: list[float] = []
    for index, month_days in enumerate(_months(start, end)):
        base = 100.0 * (1.0 + 0.005 * index)
        for offset, day in enumerate(month_days):
            factor = (
                _K4_SHAPE[offset]
                if offset < len(_K4_SHAPE)
                else _K4_SHAPE[-1] + 0.006 * (offset - len(_K4_SHAPE) + 1)
            )
            days.append(day)
            closes.append(base * factor)
    return _frame(days, closes)


def _screen(
    prices: pl.DataFrame,
    *,
    start: date = _SCREEN_START,
    end: date = _SCREEN_END,
) -> AccumulationScreenReport:
    return screen_qqq_accumulation(
        prices=prices,
        ticker="QQQ",
        start=start,
        end=end,
        monthly_contribution=1_000_000.0,
        bootstrap_draws=64,
    )


def _rows(report: AccumulationScreenReport) -> dict[str, ArmScreenRow]:
    return {row.name: row for row in report.rows}


@pytest.mark.parametrize("scenario_id", ["ACC-A-monotone-open-beats-end"])
def test_acc_a_monotone_open_beats_end(scenario_id: str) -> None:
    """ACC-A-monotone-open-beats-end"""
    report = _screen(_monotone_prices())
    rows = _rows(report)

    assert rows["month_open"].ratio_vs_month_end > 1.0
    assert rows["session_k4"].ratio_vs_month_end < rows["month_open"].ratio_vs_month_end
    assert rows["dip2_wait5"].ratio_vs_month_end < rows["month_open"].ratio_vs_month_end
    assert rows["twice_monthly"].ratio_vs_month_end < rows["month_open"].ratio_vs_month_end
    assert report.operational_unlock is False
    assert rows["month_open"].verdict is ArmVerdict.RESEARCH_ONLY
    assert report.recommended_research_arm == "month_open"


@pytest.mark.parametrize("scenario_id", ["ACC-A-verdict-reject-dip-and-splits"])
def test_acc_a_verdict_reject_dip_and_splits(scenario_id: str) -> None:
    """ACC-A-verdict-reject-dip-and-splits"""
    report = _screen(_monotone_prices())
    rows = _rows(report)

    assert rows["month_end"].verdict is ArmVerdict.REJECT
    assert rows["month_end"].ratio_vs_month_end == 1.0
    for name in (
        "twice_monthly",
        "weekly4",
        "session_k1",
        "session_k9",
        "dip2_wait5",
        "dip3_wait10",
        "dip5_wait15",
    ):
        assert rows[name].verdict is ArmVerdict.REJECT, name
    assert all(row.verdict is not ArmVerdict.ADOPT for row in report.rows)


@pytest.mark.parametrize("scenario_id", ["ACC-A-paired-log-fill"])
def test_acc_a_paired_log_fill(scenario_id: str) -> None:
    """ACC-A-paired-log-fill"""
    rows = _rows(_screen(_monotone_prices()))

    month_open = rows["month_open"]
    assert isinstance(month_open.log_fill_p_value, float)
    assert 0.0 < month_open.log_fill_p_value < 1.0
    assert month_open.mean_log_fill_gap_vs_end is not None
    assert month_open.mean_log_fill_gap_vs_end > 0.0
    assert rows["twice_monthly"].log_fill_p_value is None
    assert rows["weekly4"].log_fill_p_value is None


@pytest.mark.parametrize("scenario_id", ["ACC-A-k4-midmonth-cheap"])
def test_acc_a_k4_midmonth_cheap(scenario_id: str) -> None:
    """ACC-A-k4-midmonth-cheap"""
    start = date(2021, 1, 4)
    end = date(2021, 10, 31)
    report = _screen(_k4_prices(), start=start, end=end)
    rows = _rows(report)

    assert rows["session_k4"].ratio_vs_month_end > rows["month_open"].ratio_vs_month_end
    assert rows["session_k4"].verdict is ArmVerdict.RESEARCH_ONLY
    assert report.recommended_research_arm == "session_k4"
    assert report.operational_unlock is False


@pytest.mark.parametrize("scenario_id", ["ACC-A-dip-wait-window"])
def test_acc_a_dip_wait_window(scenario_id: str) -> None:
    """Late-month dips outside max_wait must not move the dip arm to month-end fills."""
    start = date(2021, 1, 4)
    end = date(2021, 10, 31)
    days: list[date] = []
    closes: list[float] = []
    for month_days in _months(start, end):
        for offset, day in enumerate(month_days):
            days.append(day)
            closes.append(100.0 + offset if offset < len(month_days) - 1 else 80.0)
    report = _screen(_frame(days, closes), start=start, end=end)
    rows = _rows(report)

    assert rows["dip2_wait5"].ratio_vs_month_end < 1.0


@pytest.mark.parametrize("scenario_id", ["ACC-A-fail-closed"])
def test_acc_a_fail_closed(scenario_id: str) -> None:
    """ACC-A-fail-closed"""
    common = {
        "prices": _monotone_prices(),
        "start": _SCREEN_START,
        "end": _SCREEN_END,
    }
    with pytest.raises(ValueError, match="monthly_contribution"):
        screen_qqq_accumulation(**common, ticker="QQQ", monthly_contribution=0.0)
    with pytest.raises(ValueError, match="hurdle"):
        screen_qqq_accumulation(**common, ticker="QQQ", monthly_contribution=1_000_000.0, hurdle=-0.01)
    with pytest.raises(ValueError, match="bootstrap_draws"):
        screen_qqq_accumulation(**common, ticker="QQQ", monthly_contribution=1_000_000.0, bootstrap_draws=0)
    with pytest.raises(ValueError, match="missing from prices"):
        screen_qqq_accumulation(**common, ticker="SPY", monthly_contribution=1_000_000.0)

    solo_day = load_calendar("XNYS").sessions(date(2021, 1, 4), date(2021, 1, 4))
    assert len(solo_day) == 1
    with pytest.raises(ValueError, match="zero usable months"):
        screen_qqq_accumulation(
            prices=_frame(list(solo_day), [100.0]),
            ticker="QQQ",
            start=date(2021, 1, 4),
            end=date(2021, 1, 4),
            monthly_contribution=1_000_000.0,
        )
