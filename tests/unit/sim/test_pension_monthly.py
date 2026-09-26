"""Invariant guards for the monthly pension DCA simulator."""

from __future__ import annotations

import calendar as _calendar
from datetime import UTC, date, datetime, timedelta

import polars as pl
import pytest

from src.sim.pension_monthly import (
    MonthlyReturnPanel,
    WeightSchedule,
    block_bootstrap_panels,
    panel_from_prices,
    panel_from_research,
    simulate_cohorts,
    simulate_pension_dca,
)

_AS_OF = datetime(2021, 6, 15, tzinfo=UTC)


def _month_ends(year: int, first: int, last: int) -> tuple[date, ...]:
    return tuple(date(year, month, _calendar.monthrange(year, month)[1]) for month in range(first, last + 1))


def _panel(
    months: tuple[date, ...],
    returns: dict[str, tuple[float, ...]],
    tier: str = "modern",
) -> MonthlyReturnPanel:
    return MonthlyReturnPanel(tier=tier, months=months, returns=returns)


def _fixed_mix(weights: dict[str, float]) -> WeightSchedule:
    return WeightSchedule(start_weights=dict(weights), end_weights=dict(weights), glide_years=0)


def test_simulate_zero_market_conserves_contributions() -> None:
    """Zero returns and zero drag leave terminal equal to contributed."""
    months = _month_ends(2000, 1, 12) + _month_ends(2001, 1, 12)
    panel = _panel(months, {"SPY": (0.0,) * 24})
    result = simulate_pension_dca(
        panel,
        _fixed_mix({"SPY": 1.0}),
        start_month_index=0,
        years=2,
        pre_retirement_months=6,
        annual_drag={},
    )
    assert result.terminal_value == pytest.approx(2.0)
    assert result.contributed == pytest.approx(2.0)
    assert result.max_drawdown == pytest.approx(0.0)
    assert result.pre_retirement_drawdown == pytest.approx(0.0)


def test_simulate_single_sleeve_annuity_due() -> None:
    """Constant monthly compounding matches the closed-form annuity-due value."""
    months = _month_ends(2000, 1, 12) + _month_ends(2001, 1, 12)
    panel = _panel(months, {"SPY": (0.01,) * 24})
    result = simulate_pension_dca(
        panel,
        _fixed_mix({"SPY": 1.0}),
        start_month_index=0,
        years=2,
        pre_retirement_months=12,
        annual_drag={},
    )
    assert result.terminal_value == pytest.approx(1.01**24 + 1.01**12)


def test_simulate_annual_rebalance_restores_weights() -> None:
    """A diverging year followed by rebalancing matches the hand-computed path."""
    months = _month_ends(2000, 1, 12) + _month_ends(2001, 1, 12)
    panel = _panel(
        months,
        {"AAA": (0.10,) * 12 + (0.0,) * 12, "BBB": (0.0,) * 12 + (0.10,) * 12},
    )
    result = simulate_pension_dca(
        panel,
        _fixed_mix({"AAA": 0.5, "BBB": 0.5}),
        start_month_index=0,
        years=2,
        pre_retirement_months=12,
        annual_drag={},
    )
    grown_a = 0.5 * 1.1**12
    grown_b = 0.5
    rebalanced = grown_a + grown_b + 1.0
    assert result.terminal_value == pytest.approx(0.5 * rebalanced * (1.0 + 1.1**12))


def test_weight_schedule_glide_interpolates_linearly() -> None:
    """Glide years move linearly from the held mix to the terminal mix."""
    schedule = WeightSchedule(
        start_weights={"AAA": 0.7, "BBB": 0.3},
        end_weights={"AAA": 0.0, "BBB": 1.0},
        glide_years=10,
    )
    assert schedule.weights_for_year(19, 30) == pytest.approx({"AAA": 0.7, "BBB": 0.3})
    assert schedule.weights_for_year(20, 30) == pytest.approx({"AAA": 0.63, "BBB": 0.37})
    assert schedule.weights_for_year(25, 30) == pytest.approx({"AAA": 0.28, "BBB": 0.72})
    assert schedule.weights_for_year(29, 30) == pytest.approx({"AAA": 0.0, "BBB": 1.0})
    for index in (19, 20, 25, 29):
        weights = schedule.weights_for_year(index, 30)
        assert sum(weights.values()) == pytest.approx(1.0)
        assert all(value >= 0.0 for value in weights.values())


def test_simulate_drag_reduces_terminal_monotonically() -> None:
    """A positive annual drag strictly lowers the terminal value."""
    months = _month_ends(2000, 1, 12) + _month_ends(2001, 1, 12)
    panel = _panel(months, {"SPY": (0.01,) * 24})
    base = simulate_pension_dca(
        panel,
        _fixed_mix({"SPY": 1.0}),
        start_month_index=0,
        years=2,
        pre_retirement_months=12,
        annual_drag={},
    )
    dragged = simulate_pension_dca(
        panel,
        _fixed_mix({"SPY": 1.0}),
        start_month_index=0,
        years=2,
        pre_retirement_months=12,
        annual_drag={"SPY": 0.005},
    )
    assert dragged.terminal_value < base.terminal_value


def test_simulate_cohorts_share_identical_starts() -> None:
    """Every schedule runs over the same rolling cohort starts."""
    months = _month_ends(2000, 1, 12) + _month_ends(2001, 1, 12) + _month_ends(2002, 1, 12)
    panel = _panel(months, {"SPY": (0.005,) * 36, "BBB": (0.002,) * 36})
    results = simulate_cohorts(
        panel,
        {"fixed": _fixed_mix({"SPY": 1.0, "BBB": 0.0}), "balanced": _fixed_mix({"SPY": 0.5, "BBB": 0.5})},
        years=2,
        step_months=6,
        pre_retirement_months=12,
        annual_drag={},
    )
    assert len(results["fixed"]) == len(results["balanced"]) == 3
    for index, start in enumerate((0, 6, 12)):
        expected = simulate_pension_dca(
            panel,
            _fixed_mix({"SPY": 0.5, "BBB": 0.5}),
            start_month_index=start,
            years=2,
            pre_retirement_months=12,
            annual_drag={},
        )
        assert results["balanced"][index].terminal_value == pytest.approx(expected.terminal_value)


def test_block_bootstrap_deterministic_and_comoving() -> None:
    """Identical seeds repeat panels; sleeves in a path share source months."""
    months = _month_ends(2000, 1, 12) + _month_ends(2001, 1, 12)
    panel = _panel(
        months,
        {
            "AAA": tuple(0.01 * (index + 1) for index in range(24)),
            "BBB": tuple(100.0 + 0.01 * (index + 1) for index in range(24)),
        },
        tier="century",
    )
    first = block_bootstrap_panels(panel, n_paths=4, horizon_months=12, block_months=3, seed=7)
    second = block_bootstrap_panels(panel, n_paths=4, horizon_months=12, block_months=3, seed=7)
    assert len(first) == 4
    for left, right in zip(first, second, strict=True):
        assert left.months == right.months
        assert left.returns == right.returns
    for path in first:
        assert path.tier == "bootstrap"
        assert len(path.months) == 12
        gaps = [value_b - value_a for value_a, value_b in zip(path.returns["AAA"], path.returns["BBB"], strict=True)]
        assert gaps == pytest.approx((100.0,) * 12)


def _research_frame(rows: list[tuple[str, date, float]], lag_days: int = 60) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "series_id": [series for series, _, _ in rows],
            "period_end": [month for _, month, _ in rows],
            "simple_return": [value for _, _, value in rows],
            "label": ["research_proxy"] * len(rows),
            "source": ["synthetic"] * len(rows),
            "available_at": [
                datetime(month.year, month.month, month.day, tzinfo=UTC) + timedelta(days=lag_days)
                for _, month, _ in rows
            ],
        },
        schema={
            "series_id": pl.String,
            "period_end": pl.Date,
            "simple_return": pl.Float64,
            "label": pl.String,
            "source": pl.String,
            "available_at": pl.Datetime("us", "UTC"),
        },
    )


def test_panel_from_research_enforces_visibility() -> None:
    """Months available after as_of stay invisible; corrupting them is a no-op."""
    months = [date(2021, 1, 31), date(2021, 2, 28), date(2021, 3, 31), date(2021, 4, 30)]
    rows = [(series, month, 0.01) for series in ("ff_mkt_monthly", "ff_hitec_monthly") for month in months]
    frame = _research_frame(rows)
    mapping = {"SPY": "ff_mkt_monthly", "QQQ": "ff_hitec_monthly"}
    as_of = datetime(2021, 5, 15, tzinfo=UTC)
    panel = panel_from_research(frame, mapping, as_of)
    assert list(panel.months) == [date(2021, 1, 31), date(2021, 2, 28)]
    assert panel.tier == "century"
    corrupted = frame.with_columns(
        pl.when(pl.col("period_end") > date(2021, 2, 28))
        .then(0.99)
        .otherwise(pl.col("simple_return"))
        .alias("simple_return")
    )
    assert panel_from_research(corrupted, mapping, as_of).returns == panel.returns


def test_invalid_schedules_and_gaps_fail_closed() -> None:
    """Non-simplex weights, missing months, and wipeout returns raise ValueError."""
    with pytest.raises(ValueError, match="sum to 1"):
        WeightSchedule(start_weights={"AAA": 0.6, "BBB": 0.3}, end_weights={"AAA": 0.5, "BBB": 0.5}, glide_years=0)
    with pytest.raises(ValueError, match="non-negative"):
        WeightSchedule(start_weights={"AAA": 1.0}, end_weights={"AAA": 1.0}, glide_years=-1)
    with pytest.raises(ValueError, match="differ"):
        WeightSchedule(start_weights={"AAA": 1.0}, end_weights={"BBB": 1.0}, glide_years=0)
    with pytest.raises(ValueError, match="contiguous"):
        MonthlyReturnPanel(
            tier="modern",
            months=(date(2021, 1, 31), date(2021, 3, 31)),
            returns={"AAA": (0.01, 0.02)},
        )
    with pytest.raises(ValueError, match="above -1"):
        MonthlyReturnPanel(
            tier="modern",
            months=(date(2021, 1, 31), date(2021, 2, 28)),
            returns={"AAA": (0.01, -1.0)},
        )
    frame = _research_frame(
        [("ff_mkt_monthly", date(2021, 1, 31), 0.01), ("ff_hitec_monthly", date(2021, 2, 28), 0.02)]
    )
    with pytest.raises(ValueError, match=r"different months|no visible row"):
        panel_from_research(frame, {"SPY": "ff_mkt_monthly", "QQQ": "ff_hitec_monthly"}, _AS_OF)


def test_weight_schedule_rejects_degenerate_inputs() -> None:
    """Empty, non-numeric, non-finite, and negative weights fail closed."""
    with pytest.raises(ValueError, match="non-empty"):
        WeightSchedule(start_weights={}, end_weights={}, glide_years=0)
    with pytest.raises(ValueError, match="invalid sleeve id"):
        WeightSchedule(start_weights={"": 1.0}, end_weights={"": 1.0}, glide_years=0)
    with pytest.raises(ValueError, match="must be numeric"):
        WeightSchedule(start_weights={"AAA": True}, end_weights={"AAA": True}, glide_years=0)  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="must be finite"):
        WeightSchedule(start_weights={"AAA": float("inf")}, end_weights={"AAA": float("inf")}, glide_years=0)
    with pytest.raises(ValueError, match="non-negative"):
        WeightSchedule(start_weights={"AAA": 1.5, "BBB": -0.5}, end_weights={"AAA": 1.0}, glide_years=0)
    with pytest.raises(ValueError, match="integer"):
        WeightSchedule(start_weights={"AAA": 1.0}, end_weights={"AAA": 1.0}, glide_years=True)  # type: ignore[arg-type]
    schedule = _fixed_mix({"AAA": 1.0})
    with pytest.raises(ValueError, match="integer"):
        schedule.weights_for_year(True, 1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="positive integer"):
        schedule.weights_for_year(0, 0)
    with pytest.raises(ValueError, match="outside"):
        schedule.weights_for_year(1, 1)
    gliding = WeightSchedule(start_weights={"AAA": 1.0}, end_weights={"AAA": 1.0}, glide_years=3)
    with pytest.raises(ValueError, match="exceeds"):
        gliding.weights_for_year(0, 2)


def test_panel_rejects_degenerate_inputs() -> None:
    """Empty tiers, months, series, and bad scalars fail closed."""
    months = (date(2021, 1, 31), date(2021, 2, 28))
    with pytest.raises(ValueError, match="tier"):
        MonthlyReturnPanel(tier="", months=months, returns={"AAA": (0.01, 0.02)})
    with pytest.raises(ValueError, match="months must be non-empty"):
        MonthlyReturnPanel(tier="modern", months=(), returns={"AAA": ()})
    with pytest.raises(ValueError, match="must be a date"):
        MonthlyReturnPanel(
            tier="modern", months=(datetime(2021, 1, 31, tzinfo=UTC), date(2021, 2, 28)), returns={"AAA": (0.01, 0.02)}  # type: ignore[tuple-item]
        )
    with pytest.raises(ValueError, match="not a month-end"):
        MonthlyReturnPanel(tier="modern", months=(date(2021, 1, 15),), returns={"AAA": (0.01,)})
    with pytest.raises(ValueError, match="returns must be non-empty"):
        MonthlyReturnPanel(tier="modern", months=months, returns={})
    with pytest.raises(ValueError, match="invalid sleeve id"):
        MonthlyReturnPanel(tier="modern", months=months, returns={"": (0.01, 0.02)})
    with pytest.raises(ValueError, match="length"):
        MonthlyReturnPanel(tier="modern", months=months, returns={"AAA": (0.01,)})
    with pytest.raises(ValueError, match="must be numeric"):
        MonthlyReturnPanel(tier="modern", months=months, returns={"AAA": (0.01, True)})  # type: ignore[tuple-item]
    with pytest.raises(ValueError, match=r"naive|timezone-aware"):
        panel_from_research(_research_frame([]), {"SPY": "ff_mkt_monthly"}, datetime(2021, 5, 15))
    with pytest.raises(ValueError, match="non-empty"):
        panel_from_research(_research_frame([]), {}, _AS_OF)
    frame = _research_frame([("ff_mkt_monthly", date(2021, 1, 31), 0.01)])
    with pytest.raises(ValueError, match="required column"):
        panel_from_research(frame.drop("simple_return"), {"SPY": "ff_mkt_monthly"}, _AS_OF)
    with pytest.raises(ValueError, match="availability stamp"):
        panel_from_research(frame.drop("available_at"), {"SPY": "ff_mkt_monthly"}, _AS_OF)
    with pytest.raises(ValueError, match="invalid mapping"):
        panel_from_research(frame, {"": "ff_mkt_monthly"}, _AS_OF)
    with pytest.raises(ValueError, match="no visible row"):
        panel_from_research(frame, {"SPY": "ff_mkt_monthly"}, datetime(2021, 1, 1, tzinfo=UTC))
    doubled = pl.concat([frame, frame])
    with pytest.raises(ValueError, match="duplicate"):
        panel_from_research(doubled, {"SPY": "ff_mkt_monthly"}, _AS_OF)


def test_panel_from_prices_rejects_degenerate_inputs() -> None:
    """Bad windows, columns, closes, and missing prior months fail closed."""
    frame = _prices_frame([("AAA", date(2021, 1, 29), 110.0)])
    with pytest.raises(ValueError, match="after end"):
        panel_from_prices(frame, ["AAA"], _AS_OF, date(2021, 3, 31), date(2021, 1, 31))
    with pytest.raises(ValueError, match="no month-end"):
        panel_from_prices(frame, ["AAA"], _AS_OF, date(2021, 1, 15), date(2021, 1, 15))
    with pytest.raises(ValueError, match="non-empty and unique"):
        panel_from_prices(frame, ["AAA", "AAA"], _AS_OF, date(2021, 1, 31), date(2021, 1, 31))
    with pytest.raises(ValueError, match="required column"):
        panel_from_prices(frame.drop("adjusted_close"), ["AAA"], _AS_OF, date(2021, 1, 31), date(2021, 1, 31))
    bad = _prices_frame([("AAA", date(2021, 1, 29), 0.0), ("AAA", date(2020, 12, 31), 100.0)])
    with pytest.raises(ValueError, match="finite and positive"):
        panel_from_prices(bad, ["AAA"], _AS_OF, date(2021, 1, 31), date(2021, 1, 31))
    gapped = _prices_frame(
        [("AAA", date(2021, 1, 29), 110.0), ("AAA", date(2020, 12, 31), 100.0), ("AAA", date(2021, 3, 31), 120.0)]
    )
    with pytest.raises(ValueError, match="lack a month-end close"):
        panel_from_prices(gapped, ["AAA"], _AS_OF, date(2021, 1, 31), date(2021, 3, 31))
    noprior = _prices_frame([("AAA", date(2021, 1, 29), 110.0)])
    with pytest.raises(ValueError, match="prior month-end"):
        panel_from_prices(noprior, ["AAA"], _AS_OF, date(2021, 1, 31), date(2021, 1, 31))


def test_simulation_rejects_degenerate_inputs() -> None:
    """Bad cohorts, drags, steps, and bootstrap counts fail closed."""
    months = _month_ends(2000, 1, 12)
    panel = _panel(months, {"SPY": (0.01,) * 12})
    mix = _fixed_mix({"SPY": 1.0})
    with pytest.raises(ValueError, match="positive integer"):
        simulate_pension_dca(panel, mix, start_month_index=0, years=0, pre_retirement_months=0, annual_drag={})
    with pytest.raises(ValueError, match=r"\[0, 12\]"):
        simulate_pension_dca(panel, mix, start_month_index=0, years=1, pre_retirement_months=13, annual_drag={})
    with pytest.raises(ValueError, match="missing from panel"):
        simulate_pension_dca(panel, _fixed_mix({"BBB": 1.0}), start_month_index=0, years=1, pre_retirement_months=0, annual_drag={})
    with pytest.raises(ValueError, match="non-negative integer"):
        simulate_pension_dca(panel, mix, start_month_index=-1, years=1, pre_retirement_months=0, annual_drag={})
    with pytest.raises(ValueError, match="extends past"):
        simulate_pension_dca(panel, mix, start_month_index=1, years=1, pre_retirement_months=0, annual_drag={})
    with pytest.raises(ValueError, match="unknown sleeve"):
        simulate_pension_dca(panel, mix, start_month_index=0, years=1, pre_retirement_months=0, annual_drag={"BBB": 0.01})
    with pytest.raises(ValueError, match="must be numeric"):
        simulate_pension_dca(panel, mix, start_month_index=0, years=1, pre_retirement_months=0, annual_drag={"SPY": True})  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="finite and non-negative"):
        simulate_pension_dca(panel, mix, start_month_index=0, years=1, pre_retirement_months=0, annual_drag={"SPY": -0.01})
    with pytest.raises(ValueError, match="below 1"):
        simulate_pension_dca(panel, mix, start_month_index=0, years=1, pre_retirement_months=0, annual_drag={"SPY": 1.0})
    with pytest.raises(ValueError, match="non-empty"):
        simulate_cohorts(panel, {}, years=1, step_months=1, pre_retirement_months=0, annual_drag={})
    with pytest.raises(ValueError, match="step_months"):
        simulate_cohorts(panel, {"a": mix}, years=1, step_months=0, pre_retirement_months=0, annual_drag={})
    with pytest.raises(ValueError, match="positive integer"):
        simulate_cohorts(panel, {"a": mix}, years=0, step_months=1, pre_retirement_months=0, annual_drag={})
    with pytest.raises(ValueError, match="exceeds"):
        simulate_cohorts(panel, {"a": mix}, years=2, step_months=1, pre_retirement_months=0, annual_drag={})
    with pytest.raises(ValueError, match="positive integer"):
        block_bootstrap_panels(panel, n_paths=0, horizon_months=6, block_months=3, seed=1)
    with pytest.raises(ValueError, match="integer"):
        block_bootstrap_panels(panel, n_paths=1, horizon_months=6, block_months=3, seed=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="exceeds panel length"):
        block_bootstrap_panels(panel, n_paths=1, horizon_months=6, block_months=13, seed=1)
    with pytest.raises(ValueError, match="exceeds panel length"):
        block_bootstrap_panels(panel, n_paths=1, horizon_months=13, block_months=3, seed=1)
    with pytest.raises(ValueError, match="not usable"):
        block_bootstrap_panels(panel, n_paths=1, horizon_months=6, block_months=3, seed=-1)


def test_simulate_schedule_subset_of_panel_sleeves() -> None:
    """A schedule may use a subset of panel sleeves; missing ones fail closed."""
    months = _month_ends(2000, 1, 12)
    panel = _panel(months, {"SPY": (0.01,) * 12, "QQQ": (0.02,) * 12})
    result = simulate_pension_dca(
        panel, _fixed_mix({"SPY": 1.0}), start_month_index=0, years=1, pre_retirement_months=0, annual_drag={}
    )
    assert result.terminal_value == pytest.approx(1.01**12)
    with pytest.raises(ValueError, match="missing from panel"):
        simulate_pension_dca(
            panel, _fixed_mix({"BBB": 1.0}), start_month_index=0, years=1, pre_retirement_months=0, annual_drag={}
        )


def test_simulate_zero_pre_retirement_window() -> None:
    """A zero pre-retirement window reports no pre-retirement drawdown."""
    months = _month_ends(2000, 1, 12) + _month_ends(2001, 1, 12)
    panel = _panel(months, {"SPY": (-0.05,) * 12 + (0.02,) * 12})
    result = simulate_pension_dca(
        panel,
        _fixed_mix({"SPY": 1.0}),
        start_month_index=0,
        years=2,
        pre_retirement_months=0,
        annual_drag={},
    )
    assert result.pre_retirement_drawdown == pytest.approx(0.0)
    assert result.max_drawdown < 0.0


def _prices_frame(sessions: list[tuple[str, date, float]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ticker": [ticker for ticker, _, _ in sessions],
            "date": [day for _, day, _ in sessions],
            "adjusted_close": [close for _, _, close in sessions],
            "available_at": [datetime(2021, 6, 1, tzinfo=UTC)] * len(sessions),
        },
        schema={
            "ticker": pl.String,
            "date": pl.Date,
            "adjusted_close": pl.Float64,
            "available_at": pl.Datetime("us", "UTC"),
        },
    )


def test_panel_from_prices_uses_month_end_closes() -> None:
    """Each month return ratios consecutive last-session adjusted closes."""
    sessions = [
        ("AAA", date(2020, 12, 30), 90.0),
        ("AAA", date(2020, 12, 31), 100.0),
        ("AAA", date(2021, 1, 28), 100.0),
        ("AAA", date(2021, 1, 29), 110.0),
        ("AAA", date(2021, 2, 25), 121.0),
        ("AAA", date(2021, 2, 26), 132.0),
        ("BBB", date(2020, 12, 31), 50.0),
        ("BBB", date(2021, 1, 29), 55.0),
        ("BBB", date(2021, 2, 26), 60.5),
    ]
    panel = panel_from_prices(
        _prices_frame(sessions),
        ["AAA", "BBB"],
        _AS_OF,
        date(2021, 1, 31),
        date(2021, 2, 28),
    )
    assert panel.tier == "modern"
    assert list(panel.months) == [date(2021, 1, 31), date(2021, 2, 28)]
    assert panel.returns["AAA"] == pytest.approx((110.0 / 100.0 - 1.0, 132.0 / 110.0 - 1.0))
    assert panel.returns["BBB"] == pytest.approx((55.0 / 50.0 - 1.0, 60.5 / 55.0 - 1.0))
    with pytest.raises(ValueError, match=r"no visible row|month-end close"):
        panel_from_prices(
            _prices_frame([row for row in sessions if row[0] == "AAA"]),
            ["AAA", "BBB"],
            _AS_OF,
            date(2021, 1, 31),
            date(2021, 2, 28),
        )
