"""Invariant guards for standalone pension-selection scenario analytics."""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import replace
from datetime import date

import pytest

from src.analytics.pension_selection import (
    CapmScenarioModel,
    DcaTailStats,
    MonthlyReturnPanel,
    PensionSelectionDataError,
    bootstrap_dca_tail,
    build_monthly_krw_panel,
    delta_grid,
    estimate_tilt_delta,
    expected_returns,
    fit_capm_scenario_model,
    growth_regret_table,
)


def _month_date(index: int, day: int) -> date:
    year = 2000 + index // 12
    month = index % 12 + 1
    return date(year, month, day)


def _panel(tickers: tuple[str, ...], columns: tuple[tuple[float, ...], ...]) -> MonthlyReturnPanel:
    rows = tuple(tuple(values[index] for values in columns) for index in range(len(columns[0])))
    return MonthlyReturnPanel(
        tickers=tickers,
        month_ends=tuple(_month_date(index, 28) for index in range(1, len(columns[0]) + 1)),
        returns=rows,
    )


def _synthetic_marks(month_count: int = 30) -> tuple[dict[tuple[str, date], float], date, date]:
    marks: dict[tuple[str, date], float] = {}
    for month_index in range(month_count):
        first_day = _month_date(month_index, 10)
        last_day = _month_date(month_index, 20)
        market_last = 100.0 * (1.01**month_index) * (1.0 + 0.002 * month_index)
        anchor_last = 80.0 * (1.012**month_index) * (1.0 - 0.001 * month_index)
        for ticker, last_mark in (("M", market_last), ("A", anchor_last)):
            marks[(ticker, first_day)] = last_mark * (0.97 + 0.001 * month_index)
            marks[(ticker, last_day)] = last_mark
    return marks, date(2000, 1, 1), _month_date(month_count - 1, 28)


def test_panel_returns_match_month_end_mark_ratios() -> None:
    """Each return is the latest in-window mark ratio and the base month is excluded."""
    marks, start, end = _synthetic_marks()
    panel = build_monthly_krw_panel(marks, tickers=("M", "A"), start=start, end=end)

    assert len(panel.month_ends) == len(panel.returns) == 29
    assert panel.month_ends[0] == _month_date(1, 20)
    for month, row in enumerate(panel.returns, start=1):
        previous_day = _month_date(month - 1, 20)
        current_day = _month_date(month, 20)
        expected = (
            marks[("M", current_day)] / marks[("M", previous_day)] - 1.0,
            marks[("A", current_day)] / marks[("A", previous_day)] - 1.0,
        )
        assert row == pytest.approx(expected, abs=1e-12)


def test_panel_refuses_missing_month_without_forward_fill() -> None:
    """A missing interior ticker month aborts rather than borrowing an older mark."""
    marks, start, end = _synthetic_marks()
    missing_month = 10
    filtered = {
        key: value
        for key, value in marks.items()
        if not (key[0] == "A" and start <= key[1] <= end and key[1].month == missing_month)
    }
    with pytest.raises(PensionSelectionDataError, match="A has no mark"):
        build_monthly_krw_panel(filtered, tickers=("M", "A"), start=start, end=end)


def test_panel_ignores_marks_outside_window() -> None:
    """Before-window and after-window marks cannot alter the return panel."""
    marks, start, end = _synthetic_marks()
    expected = build_monthly_krw_panel(marks, tickers=("M", "A"), start=start, end=end)
    outside = dict(marks)
    outside[("M", date(1999, 12, 20))] = 1.0
    outside[("A", _month_date(30, 20))] = 1_000_000.0
    assert build_monthly_krw_panel(outside, tickers=("M", "A"), start=start, end=end) == expected


@pytest.mark.parametrize(
    ("tickers", "start", "end", "message"),
    [
        ((), date(2000, 1, 1), _month_date(29, 28), "non-empty"),
        (("M", "M"), date(2000, 1, 1), _month_date(29, 28), "unique"),
        ((" ",), date(2000, 1, 1), _month_date(29, 28), "blank"),
        (("M",), date(2001, 1, 1), date(2000, 1, 1), "not be after"),
    ],
)
def test_panel_rejects_invalid_window_or_tickers(
    tickers: tuple[str, ...], start: date, end: date, message: str
) -> None:
    """Empty, duplicate, blank, and reversed panel inputs fail closed."""
    marks, _, _ = _synthetic_marks()
    with pytest.raises(ValueError, match=message):
        build_monthly_krw_panel(marks, tickers=tickers, start=start, end=end)


def test_panel_requires_twenty_four_monthly_returns() -> None:
    """A short history cannot support the scenario model."""
    marks, start, end = _synthetic_marks(10)
    with pytest.raises(PensionSelectionDataError, match="at least 24 returns"):
        build_monthly_krw_panel(marks, tickers=("M", "A"), start=start, end=end)


def _capm_panel() -> MonthlyReturnPanel:
    market = tuple(0.01 + 0.002 * math.sin(2.0 * math.pi * index / 12.0) for index in range(36))
    residual = tuple(0.001 * math.cos(2.0 * math.pi * index / 12.0) for index in range(36))
    anchor = tuple(market_value + residual_value for market_value, residual_value in zip(market, residual, strict=True))
    clone = tuple(market_value + 2.0 * residual_value for market_value, residual_value in zip(market, residual, strict=True))
    return _panel(("M", "A", "C"), (market, anchor, clone))


def _fit(panel: MonthlyReturnPanel | None = None) -> CapmScenarioModel:
    return fit_capm_scenario_model(
        panel or _capm_panel(),
        market_weights={"M": 1.0},
        anchor_ticker="A",
        risk_free_annual=0.03,
        equity_risk_premium_annual=0.05,
    )


def test_market_proxy_has_unit_beta_and_zero_tilt() -> None:
    """The market leg has CAPM beta one and no direct tech-residual loading."""
    model = _fit()
    assert model.betas["M"] == pytest.approx(1.0, abs=1e-9)
    assert model.tilt_loadings["M"] == pytest.approx(0.0, abs=1e-9)


def test_anchor_loading_is_one_and_scaled_clone_doubles() -> None:
    """Residual-regression loadings recover one for the anchor and two for its clone."""
    model = _fit()
    assert model.tilt_loadings["A"] == pytest.approx(1.0, abs=1e-12)
    assert model.tilt_loadings["C"] == pytest.approx(2.0, abs=1e-9)


def test_expected_returns_are_linear_in_delta() -> None:
    """Changing delta by h changes each expected return by h times its loading."""
    model = _fit()
    base = expected_returns(model, 0.01)
    shifted = expected_returns(model, 0.03)
    for ticker in model.tickers:
        assert shifted[ticker] - base[ticker] == pytest.approx(0.02 * model.tilt_loadings[ticker])


def test_delta_shrinkage_stays_between_prior_and_sample() -> None:
    """Normal-normal shrinkage pulls a positive sample alpha toward a lower prior."""
    panel = _capm_panel()
    estimate = estimate_tilt_delta(panel, _fit(panel), prior_mean_annual=-0.01, prior_sd_annual=0.01)
    assert estimate.sample_alpha_annual > estimate.prior_mean_annual
    assert estimate.prior_mean_annual < estimate.posterior_mean_annual < estimate.sample_alpha_annual
    assert estimate.posterior_sd_annual < min(estimate.prior_sd_annual, estimate.sample_alpha_se_annual)


def test_tight_prior_dominates_delta_estimate() -> None:
    """A near-zero prior variance leaves posterior alpha effectively unchanged."""
    panel = _capm_panel()
    estimate = estimate_tilt_delta(panel, _fit(panel), prior_mean_annual=0.004, prior_sd_annual=1e-6)
    assert estimate.posterior_mean_annual == pytest.approx(0.004, abs=1e-8)


def test_delta_grid_is_symmetric_inclusive_and_uniform() -> None:
    """The odd-sized grid contains exact posterior endpoints and its center."""
    panel = _capm_panel()
    estimate = estimate_tilt_delta(panel, _fit(panel), prior_mean_annual=0.0, prior_sd_annual=0.01)
    grid = delta_grid(estimate, z=2.0, n_points=5)
    spacing = (grid[-1] - grid[0]) / 4.0
    assert len(grid) == 5
    assert grid[0] == estimate.posterior_mean_annual - 2.0 * estimate.posterior_sd_annual
    assert grid[-1] == estimate.posterior_mean_annual + 2.0 * estimate.posterior_sd_annual
    assert grid[2] == estimate.posterior_mean_annual
    assert all(grid[index + 1] - grid[index] == pytest.approx(spacing) for index in range(4))


def test_delta_grid_rejects_even_point_count() -> None:
    """Only odd grids preserve the posterior center."""
    panel = _capm_panel()
    estimate = estimate_tilt_delta(panel, _fit(panel), prior_mean_annual=0.0, prior_sd_annual=0.01)
    with pytest.raises(ValueError, match="odd integer"):
        delta_grid(estimate, z=2.0, n_points=4)


def _known_model() -> CapmScenarioModel:
    return CapmScenarioModel(
        tickers=("M", "A"),
        covariance_annual=((0.04, 0.01), (0.01, 0.09)),
        betas={"M": 1.0, "A": 1.2},
        tilt_loadings={"M": 0.0, "A": 1.0},
        market_weights={"M": 1.0},
        risk_free_annual=0.03,
        equity_risk_premium_annual=0.05,
        anchor_ticker="A",
    )


def _growth_at(model: CapmScenarioModel, weights: dict[str, float], delta: float) -> float:
    expected = expected_returns(model, delta)
    variance = math.fsum(
        left_weight * model.covariance_annual[model.tickers.index(left)][model.tickers.index(right)] * right_weight
        for left, left_weight in weights.items()
        for right, right_weight in weights.items()
    )
    return math.fsum(weight * expected[ticker] for ticker, weight in weights.items()) - 0.5 * variance


def test_growth_equals_closed_form() -> None:
    """Every grid growth matches the annual CAPM mean-variance closed form."""
    model = _known_model()
    arms = {
        "base": {"M": 1.0},
        "tech": {"A": 1.0},
        "blend": {"M": 0.5, "A": 0.5},
    }
    table = growth_regret_table(model, arms, (-0.01, 0.0, 0.02), baseline_arm_id="base")
    for row in table.rows:
        for delta, growth in zip(table.deltas, row.growth_by_delta, strict=True):
            assert growth == pytest.approx(_growth_at(model, arms[row.arm_id], delta), abs=1e-12)


def test_regret_is_nonnegative_with_zero_delta_winner() -> None:
    """Each scenario has a zero-regret winner and no arm has negative regret."""
    table = growth_regret_table(
        _known_model(),
        {"base": {"M": 1.0}, "tech": {"A": 1.0}, "blend": {"M": 0.5, "A": 0.5}},
        (-0.02, 0.0, 0.02),
        baseline_arm_id="base",
    )
    for delta_index in range(len(table.deltas)):
        growth = [row.growth_by_delta[delta_index] for row in table.rows]
        regrets = [max(growth) - value for value in growth]
        assert min(regrets) >= 0.0
        assert regrets.count(0.0) >= 1
    assert all(row.max_regret >= 0.0 and row.mean_regret >= 0.0 for row in table.rows)


def test_breakeven_delta_ties_growth_and_parallel_arm_is_none() -> None:
    """A tilted arm crosses the baseline exactly; equal exposure has no unique crossing."""
    model = _known_model()
    arms = {
        "base": {"M": 1.0},
        "market_copy": {"M": 1.0},
        "tech": {"A": 1.0},
    }
    table = growth_regret_table(model, arms, (-0.02, 0.0, 0.02), baseline_arm_id="base")
    breakeven = table.breakeven_vs_baseline["tech"]
    assert breakeven is not None
    assert _growth_at(model, arms["tech"], breakeven) == pytest.approx(
        _growth_at(model, arms["base"], breakeven), abs=1e-12
    )
    assert table.breakeven_vs_baseline["base"] is None
    assert table.breakeven_vs_baseline["market_copy"] is None


def _bootstrap_panel() -> MonthlyReturnPanel:
    market = tuple(0.006 + 0.003 * math.sin(2.0 * math.pi * index / 12.0) for index in range(36))
    anchor = tuple(0.008 + 0.005 * math.cos(2.0 * math.pi * index / 12.0) for index in range(36))
    return _panel(("M", "A"), (market, anchor))


def _bootstrap_kwargs() -> dict[str, object]:
    return {
        "delta": 0.0,
        "horizon_months": 12,
        "n_paths": 64,
        "block_months": 3,
        "pre_retirement_months": 4,
        "quantile": 0.05,
        "seed": 17,
    }


def test_bootstrap_is_deterministic_in_seed() -> None:
    """Identical seed and inputs reproduce complete tail-statistics rows."""
    panel, model, arms = _bootstrap_panel(), _known_model(), {"base": {"M": 1.0}, "tech": {"A": 1.0}}
    first = bootstrap_dca_tail(panel, model, arms, **_bootstrap_kwargs())  # type: ignore[arg-type]
    second = bootstrap_dca_tail(panel, model, arms, **_bootstrap_kwargs())  # type: ignore[arg-type]
    changed = bootstrap_dca_tail(panel, model, arms, **_bootstrap_kwargs() | {"seed": 18})  # type: ignore[arg-type]
    assert first == second
    assert first != changed


def test_scenario_premium_moves_only_loaded_tickers() -> None:
    """Changing delta lifts the anchor arm while market-only statistics stay identical."""
    panel, model = _bootstrap_panel(), _known_model()
    arms = {"market": {"M": 1.0}, "anchor": {"A": 1.0}}
    downside = bootstrap_dca_tail(panel, model, arms, **_bootstrap_kwargs() | {"delta": -0.02})  # type: ignore[arg-type]
    upside = bootstrap_dca_tail(panel, model, arms, **_bootstrap_kwargs() | {"delta": 0.02})  # type: ignore[arg-type]
    assert upside[1].median_terminal_multiple > downside[1].median_terminal_multiple
    market_down, market_up = downside[0], upside[0]
    assert market_up.low_quantile_terminal_multiple == market_down.low_quantile_terminal_multiple
    assert market_up.median_terminal_multiple == market_down.median_terminal_multiple
    assert market_up.high_quantile_pre_retirement_drawdown == market_down.high_quantile_pre_retirement_drawdown
    assert market_up.prob_below_principal == market_down.prob_below_principal


def test_paired_paths_make_identical_arms_identical() -> None:
    """Different arm ids with identical targets consume the same paired path."""
    stats = bootstrap_dca_tail(
        _bootstrap_panel(),
        _known_model(),
        {"first": {"A": 1.0}, "second": {"A": 1.0}},
        **_bootstrap_kwargs(),  # type: ignore[arg-type]
    )
    first, second = stats
    assert first.arm_id != second.arm_id
    assert replace(first, arm_id=second.arm_id) == second


def test_monotone_accounts_never_breach_principal() -> None:
    """Strictly positive adjusted returns preserve principal and create no drawdown."""
    positive = (0.01,) * 36
    panel = _panel(("M", "A"), (positive, positive))
    stats = bootstrap_dca_tail(
        panel,
        _known_model(),
        {"market": {"M": 1.0}, "anchor": {"A": 1.0}},
        **_bootstrap_kwargs(),  # type: ignore[arg-type]
    )
    assert all(row.prob_below_principal == 0.0 for row in stats)
    assert all(row.high_quantile_pre_retirement_drawdown == 0.0 for row in stats)


def test_horizon_longer_than_panel_is_rejected() -> None:
    """A DCA path cannot claim months beyond the certified panel."""
    with pytest.raises(ValueError, match="must not exceed panel length"):
        bootstrap_dca_tail(
            _bootstrap_panel(),
            _known_model(),
            {"base": {"M": 1.0}},
            **_bootstrap_kwargs() | {"horizon_months": 37},  # type: ignore[arg-type]
        )


def test_bootstrap_logs_progress_and_completion(caplog: pytest.LogCaptureFixture) -> None:
    """Long runs disclose the 500-path checkpoint and final completion."""
    caplog.set_level(logging.INFO, logger="src.analytics.pension_selection")
    bootstrap_dca_tail(
        _bootstrap_panel(),
        _known_model(),
        {"base": {"M": 1.0}},
        **_bootstrap_kwargs() | {"n_paths": 501, "horizon_months": 1, "pre_retirement_months": 1},  # type: ignore[arg-type]
    )
    messages = [record.getMessage() for record in caplog.records if "pension_tail_progress" in record.getMessage()]
    assert len(messages) == 2
    assert "paths_done=500 n_paths=501" in messages[0]
    assert "paths_done=501 n_paths=501" in messages[1]


@pytest.mark.parametrize(
    ("case_id", "mutate"),
    [
        ("empty_arms", lambda values: values.__setitem__("arms", {})),
        ("blank_arm", lambda values: values.__setitem__("arms", {" ": {"M": 1.0}})),
        ("boolean_weight", lambda values: values.__setitem__("arms", {"bad": {"M": True}})),
        ("negative_weight", lambda values: values.__setitem__("arms", {"bad": {"M": -0.1}})),
        ("weight_sum", lambda values: values.__setitem__("arms", {"bad": {"M": 0.5}})),
        ("unknown_ticker", lambda values: values.__setitem__("arms", {"bad": {"Z": 1.0}})),
        ("nonfinite_delta", lambda values: values.__setitem__("delta", math.nan)),
        ("zero_horizon", lambda values: values.__setitem__("horizon_months", 0)),
        ("long_horizon", lambda values: values.__setitem__("horizon_months", 37)),
        ("zero_paths", lambda values: values.__setitem__("n_paths", 0)),
        ("zero_block", lambda values: values.__setitem__("block_months", 0)),
        ("long_block", lambda values: values.__setitem__("block_months", 37)),
        ("zero_pre_retirement", lambda values: values.__setitem__("pre_retirement_months", 0)),
        ("long_pre_retirement", lambda values: values.__setitem__("pre_retirement_months", 13)),
        ("invalid_quantile", lambda values: values.__setitem__("quantile", 0.5)),
    ],
)
def test_bootstrap_rejects_invalid_inputs(
    case_id: str, mutate: Callable[[dict[str, object]], None]
) -> None:
    """Invalid scenario, sizing, probability, weight, and ticker inputs fail closed."""
    values = _bootstrap_kwargs()
    values["arms"] = {"base": {"M": 1.0}}
    mutate(values)
    with pytest.raises(ValueError, match=r"arm|weight|ticker|delta|months|paths|block|quantile"):
        bootstrap_dca_tail(_bootstrap_panel(), _known_model(), **values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("case_id", "market_weights", "anchor", "rate", "message"),
    [
        ("empty_market", {}, "A", 0.03, "non-empty"),
        ("boolean_market_weight", {"M": True}, "A", 0.03, "finite number"),
        ("negative_market_weight", {"M": -1.0}, "A", 0.03, "finite number"),
        ("market_weight_sum", {"M": 0.5}, "A", 0.03, "sum to 1"),
        ("unknown_market", {"Z": 1.0}, "A", 0.03, "outside the panel"),
        ("unknown_anchor", {"M": 1.0}, "Z", 0.03, "anchor ticker"),
        ("nonfinite_rate", {"M": 1.0}, "A", math.nan, "rates must be finite"),
    ],
)
def test_capm_fit_rejects_invalid_contract(
    case_id: str, market_weights: dict[str, float], anchor: str, rate: float, message: str
) -> None:
    """Market references, weights, anchor, and rates fail closed."""
    with pytest.raises(ValueError, match=message):
        fit_capm_scenario_model(
            _capm_panel(),
            market_weights=market_weights,
            anchor_ticker=anchor,
            risk_free_annual=rate,
            equity_risk_premium_annual=0.05,
        )


def test_capm_fit_rejects_insufficient_or_degenerate_panel() -> None:
    """Too few rows, a constant market, and a fully explained anchor are rejected."""
    short = _panel(("M", "A"), ((0.01, 0.02), (0.02, 0.01)))
    with pytest.raises(ValueError, match="at least three"):
        fit_capm_scenario_model(
            short,
            market_weights={"M": 1.0},
            anchor_ticker="A",
            risk_free_annual=0.03,
            equity_risk_premium_annual=0.05,
        )
    constant = _panel(("M", "A"), ((0.01,) * 24, (0.02,) * 24))
    with pytest.raises(ValueError, match="market proxy"):
        fit_capm_scenario_model(
            constant,
            market_weights={"M": 1.0},
            anchor_ticker="A",
            risk_free_annual=0.0,
            equity_risk_premium_annual=0.0,
        )
    market = tuple(0.01 + 0.001 * index for index in range(24))
    explained = _panel(("M", "A"), (market, market))
    with pytest.raises(ValueError, match="anchor CAPM residual"):
        fit_capm_scenario_model(
            explained,
            market_weights={"M": 1.0},
            anchor_ticker="A",
            risk_free_annual=0.0,
            equity_risk_premium_annual=0.0,
        )


@pytest.mark.parametrize("case_id", ["nonfinite_mean", "invalid_sd", "nonpositive_se"])
def test_delta_estimate_rejects_undefined_posterior(case_id: str) -> None:
    """Invalid priors or zero residual dispersion never produce a fake posterior."""
    panel = _bootstrap_panel()
    model = _known_model()
    if case_id == "nonfinite_mean":
        with pytest.raises(ValueError, match="prior_mean_annual"):
            estimate_tilt_delta(panel, model, prior_mean_annual=math.nan, prior_sd_annual=0.01)
    elif case_id == "invalid_sd":
        with pytest.raises(ValueError, match="prior_sd_annual"):
            estimate_tilt_delta(panel, model, prior_mean_annual=0.0, prior_sd_annual=True)
    else:
        constant = _panel(("M", "A"), ((0.01,) * 3, (0.02,) * 3))
        zero_beta = replace(model, betas={"M": 1.0, "A": 0.0})
        with pytest.raises(ValueError, match="standard error"):
            estimate_tilt_delta(constant, zero_beta, prior_mean_annual=0.0, prior_sd_annual=0.01)


def test_delta_estimate_requires_residual_degrees_of_freedom() -> None:
    """The ddof=2 anchor standard error needs more than two observations."""
    short = _panel(("M", "A"), ((0.01, 0.02), (0.02, 0.01)))
    with pytest.raises(ValueError, match="more than 2"):
        estimate_tilt_delta(short, _known_model(), prior_mean_annual=0.0, prior_sd_annual=0.01)


def test_scenario_and_grid_reject_nonfinite_values() -> None:
    """Delta and z cannot smuggle NaN or infinity into ranking outputs."""
    model = _known_model()
    with pytest.raises(ValueError, match="delta must be finite"):
        expected_returns(model, math.nan)
    panel = _capm_panel()
    estimate = estimate_tilt_delta(panel, _fit(panel), prior_mean_annual=0.0, prior_sd_annual=0.01)
    with pytest.raises(ValueError, match="z must be finite"):
        delta_grid(estimate, z=math.inf, n_points=5)


def test_growth_table_rejects_empty_or_unmatched_inputs() -> None:
    """Growth evaluation requires arms, deltas, and a declared baseline."""
    model = _known_model()
    with pytest.raises(ValueError, match="arms must be non-empty"):
        growth_regret_table(model, {}, (0.0,), baseline_arm_id="base")
    with pytest.raises(ValueError, match="deltas must be non-empty"):
        growth_regret_table(model, {"base": {"M": 1.0}}, (), baseline_arm_id="base")
    with pytest.raises(ValueError, match="not present"):
        growth_regret_table(model, {"base": {"M": 1.0}}, (0.0,), baseline_arm_id="ghost")


def test_tail_stats_retain_complete_scenario_metadata() -> None:
    """Returned rows retain arm, scenario, horizon, path count, and tail probability."""
    stats = bootstrap_dca_tail(
        _bootstrap_panel(),
        _known_model(),
        {"base": {"M": 1.0}},
        **_bootstrap_kwargs(),  # type: ignore[arg-type]
    )
    assert stats == (
        DcaTailStats(
            arm_id="base",
            delta=0.0,
            horizon_months=12,
            n_paths=64,
            quantile=0.05,
            low_quantile_terminal_multiple=stats[0].low_quantile_terminal_multiple,
            median_terminal_multiple=stats[0].median_terminal_multiple,
            high_quantile_pre_retirement_drawdown=stats[0].high_quantile_pre_retirement_drawdown,
            prob_below_principal=stats[0].prob_below_principal,
        ),
    )
