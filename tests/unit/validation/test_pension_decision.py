"""Invariant guards for robust pension decision evaluation."""

from __future__ import annotations

import calendar as _calendar
from datetime import date

import pytest

from src.sim.pension_monthly import MonthlyReturnPanel, WeightSchedule
from src.validation.pension_decision import (
    PensionDecisionReport,
    assert_tax_rank_neutrality,
    evaluate_pension_decision,
)
from src.validation.pension_decision_config import PensionDecisionSpec

_BENCH = "spy100_qqq0"


def _months(count: int) -> tuple[date, ...]:
    out: list[date] = []
    year, month = 2000, 1
    for _ in range(count):
        out.append(date(year, month, _calendar.monthrange(year, month)[1]))
        month += 1
        if month > 12:
            year += 1
            month = 1
    return tuple(out)


def _panel(months: tuple[date, ...], sleeves: dict[str, tuple[float, ...]], tier: str) -> MonthlyReturnPanel:
    return MonthlyReturnPanel(tier=tier, months=months, returns=dict(sleeves))


def _mix(spy: float, qqq: float) -> WeightSchedule:
    return WeightSchedule(start_weights={"SPY": spy, "QQQ": qqq}, end_weights={"SPY": spy, "QQQ": qqq}, glide_years=0)


def _spec(
    candidates: dict[str, WeightSchedule],
    neighbors: dict[str, list[str]],
    *,
    horizons: tuple[int, ...] = (2,),
    band: float = 0.005,
    floor: float = 0.6,
    sensitivity: tuple[float, ...] = (3.0,),
    paths: int = 30,
    pre: int = 12,
    reference: str | None = None,
    controls: dict[str, WeightSchedule] | None = None,
    lineage: int = 2,
    benchmark: str = _BENCH,
    realized_horizons: tuple[int, ...] = (),
) -> PensionDecisionSpec:
    return PensionDecisionSpec(
        name="probe",
        benchmark_id=benchmark,
        candidates=dict(candidates),
        neighbors={key: tuple(value) for key, value in neighbors.items()},
        century_series={"SPY": "ff_mkt_monthly", "QQQ": "ff_hitec_monthly"},
        modern_start=date(2000, 1, 31),
        modern_end=date(2001, 12, 31),
        century_start=date(2000, 1, 31),
        century_end=date(2001, 12, 31),
        horizons_years=horizons,
        step_months=12,
        pre_retirement_months=pre,
        primary_gamma=1.0,
        sensitivity_gammas=sensitivity,
        equivalence_band=band,
        min_bootstrap_win_share=floor,
        bootstrap_paths=paths,
        bootstrap_block_months=3,
        annual_drag_by_sleeve={},
        tax_crosscheck_campaign_path="experiments/pension_campaign_v2_dotcom.json",
        tax_crosscheck_arm_map={},
        review_every_months=12,
        lineage={"related_trial_count": lineage},
        modern_splices={},
        sleeve_products={},
        dominance_reference_id=reference,
        controls=dict(controls) if controls else {},
        realized_horizons_years=realized_horizons,
    )


def _flat_panel(months: tuple[date, ...], spy: float, qqq: float, tier: str) -> MonthlyReturnPanel:
    return _panel(months, {"SPY": (spy,) * len(months), "QQQ": (qqq,) * len(months)}, tier)


def test_growth_plateau_is_adopted() -> None:
    """Near-equal log growth across tiers and horizons adopts the plateau member."""
    months = _months(48)
    spec = _spec(
        {
            _BENCH: _mix(1.0, 0.0),
            "spy50_qqq50": _mix(0.5, 0.5),
            "spy20_qqq80": _mix(0.2, 0.8),
            "spy0_qqq100": _mix(0.0, 1.0),
        },
        {
            _BENCH: ["spy50_qqq50"],
            "spy50_qqq50": [_BENCH, "spy20_qqq80"],
            "spy20_qqq80": ["spy50_qqq50", "spy0_qqq100"],
            "spy0_qqq100": ["spy20_qqq80"],
        },
        horizons=(2, 3),
        paths=20,
    )
    modern = _flat_panel(months, 0.005, 0.0052, "modern")
    century = _flat_panel(months, 0.005, 0.0052, "century")
    report = evaluate_pension_decision(spec, modern, century, seed=11)
    assert report.status == "ADOPT_CANDIDATE"
    assert set(report.equivalent_ids) == {_BENCH, "spy50_qqq50", "spy20_qqq80", "spy0_qqq100"}
    assert report.selected_id == "spy0_qqq100"
    assert report.trial_count == 6


def test_equal_growth_prefers_milder_late_drawdown() -> None:
    """Within the band, the candidate with the milder pre-retirement drawdown wins."""
    months = _months(24)
    qqq = (0.008,) * 23 + (-0.01,)
    spec = _spec(
        {_BENCH: _mix(1.0, 0.0), "spy50_qqq50": _mix(0.5, 0.5), "spy80_qqq20": _mix(0.8, 0.2)},
        {_BENCH: ["spy80_qqq20"], "spy80_qqq20": ["spy50_qqq50"], "spy50_qqq50": ["spy80_qqq20"]},
        band=0.015,
    )
    panel = _panel(months, {"SPY": (0.005,) * 24, "QQQ": qqq}, "modern")
    twin = _panel(months, {"SPY": (0.005,) * 24, "QQQ": qqq}, "century")
    report = evaluate_pension_decision(spec, panel, twin, seed=11)
    assert report.status == "ADOPT_CANDIDATE"
    assert report.robust_scores["spy50_qqq50"] > report.robust_scores["spy80_qqq20"]
    assert report.selected_id == "spy80_qqq20"


def test_century_regime_vetoes_modern_evidence() -> None:
    """A modern winner that loses the century tier scores below 1 and is skipped."""
    months = _months(24)
    modern = _panel(months, {"SPY": (0.004,) * 24, "QQQ": (0.01,) * 24}, "modern")
    century = _panel(
        months,
        {"SPY": (0.004,) * 24, "QQQ": (0.02,) * 18 + (-0.04,) * 6},
        "century",
    )
    glide = WeightSchedule(
        start_weights={"SPY": 0.0, "QQQ": 1.0},
        end_weights={"SPY": 1.0, "QQQ": 0.0},
        glide_years=2,
    )
    spec = _spec(
        {_BENCH: _mix(1.0, 0.0), "glide_out": glide, "spy0_qqq100": _mix(0.0, 1.0)},
        {_BENCH: ["glide_out"], "glide_out": [], "spy0_qqq100": []},
    )
    report = evaluate_pension_decision(spec, modern, century, seed=11)
    assert report.robust_scores["spy0_qqq100"] < 1.0
    assert report.selected_id == "glide_out"
    assert report.status == "ADOPT_CANDIDATE"


def test_knife_edge_selection_blocked() -> None:
    """A selection whose declared neighbor loses to the benchmark is not adopted."""
    months = _months(24)
    spec = _spec(
        {
            _BENCH: _mix(1.0, 0.0),
            "spy50_qqq50": _mix(0.5, 0.5),
            "ttt_heavy": WeightSchedule(
                start_weights={"SPY": 0.2, "TTT": 0.8},
                end_weights={"SPY": 0.2, "TTT": 0.8},
                glide_years=0,
            ),
        },
        {_BENCH: ["spy50_qqq50"], "spy50_qqq50": ["ttt_heavy"], "ttt_heavy": ["spy50_qqq50"]},
    )
    sleeves = {"SPY": (0.004,) * 24, "QQQ": (0.008,) * 24, "TTT": (-0.05,) * 24}
    modern = _panel(months, dict(sleeves), "modern")
    century = _panel(months, dict(sleeves), "century")
    report = evaluate_pension_decision(spec, modern, century, seed=11)
    assert report.status == "NO_DECISION"
    assert report.selected_id is None
    assert "KNIFE_EDGE" in report.reasons


def test_weak_bootstrap_blocks_adoption() -> None:
    """A selection winning fewer than 60% of bootstrap paths is not adopted."""
    months = _months(36)
    qqq = tuple(0.10 if index % 2 == 0 else -0.0906 for index in range(36))
    spec = _spec({_BENCH: _mix(1.0, 0.0), "spy0_qqq100": _mix(0.0, 1.0)}, {_BENCH: [], "spy0_qqq100": []}, horizons=(2, 3), paths=200, band=0.002)
    modern = _flat_panel(months, 0.0, 0.02, "modern")
    century = _panel(months, {"SPY": (0.0,) * 36, "QQQ": qqq}, "century")
    report = evaluate_pension_decision(spec, modern, century, seed=11)
    assert report.robust_scores["spy0_qqq100"] > 1.0
    assert report.bootstrap_win_share["spy0_qqq100"] < 0.6
    assert report.status == "NO_DECISION"
    assert "BOOTSTRAP_WEAK" in report.reasons


def test_benchmark_best_keeps_benchmark() -> None:
    """No candidate beating the benchmark keeps the benchmark."""
    months = _months(24)
    spec = _spec(
        {_BENCH: _mix(1.0, 0.0), "spy50_qqq50": _mix(0.5, 0.5)},
        {_BENCH: ["spy50_qqq50"], "spy50_qqq50": [_BENCH]},
    )
    modern = _flat_panel(months, 0.004, 0.002, "modern")
    century = _flat_panel(months, 0.004, 0.002, "century")
    report = evaluate_pension_decision(spec, modern, century, seed=11)
    assert report.status == "KEEP_BENCHMARK"
    assert report.selected_id == _BENCH


def test_incumbent_inside_band_is_kept() -> None:
    """An incumbent within the band of a slightly better challenger is not switched."""
    months = _months(24)
    spec = _spec(
        {_BENCH: _mix(1.0, 0.0), "spy50_qqq50": _mix(0.5, 0.5), "spy0_qqq100": _mix(0.0, 1.0)},
        {_BENCH: ["spy50_qqq50"], "spy50_qqq50": [_BENCH], "spy0_qqq100": ["spy50_qqq50"]},
        band=0.01,
    )
    modern = _flat_panel(months, 0.005, 0.0055, "modern")
    century = _flat_panel(months, 0.005, 0.0055, "century")
    report = evaluate_pension_decision(spec, modern, century, seed=11, incumbent_id="spy50_qqq50")
    assert report.robust_scores["spy0_qqq100"] > report.robust_scores["spy50_qqq50"]
    assert report.status == "KEEP_INCUMBENT"
    assert report.selected_id == "spy50_qqq50"


def test_sensitivity_gammas_never_decide() -> None:
    """A gamma-3 ranking flip is reported but never changes status or selection."""
    months = _months(36)
    volatile = (0.11,) * 18 + (-0.06,) * 18
    spec = _spec(
        {_BENCH: _mix(1.0, 0.0), "steady": _mix(0.5, 0.5), "wild": WeightSchedule(
            start_weights={"SPY": 0.5, "VVV": 0.5},
            end_weights={"SPY": 0.5, "VVV": 0.5},
            glide_years=0,
        )},
        {_BENCH: ["steady"], "steady": [_BENCH, "wild"], "wild": ["steady"]},
    )
    sleeves = {"SPY": (0.005,) * 36, "QQQ": (0.011,) * 36, "VVV": volatile}
    modern = _panel(months, dict(sleeves), "modern")
    century = _panel(months, dict(sleeves), "century")
    full = evaluate_pension_decision(spec, modern, century, seed=11)
    assert 3.0 in full.sensitivity_robust_scores
    top_gamma1 = max(full.robust_scores, key=lambda candidate: full.robust_scores[candidate])
    top_gamma3 = max(full.sensitivity_robust_scores[3.0], key=lambda candidate: full.sensitivity_robust_scores[3.0][candidate])
    assert top_gamma1 != top_gamma3
    assert full.selected_id == top_gamma1
    gamma1_only = evaluate_pension_decision(_spec(
        {_BENCH: _mix(1.0, 0.0), "steady": _mix(0.5, 0.5), "wild": WeightSchedule(
            start_weights={"SPY": 0.5, "VVV": 0.5},
            end_weights={"SPY": 0.5, "VVV": 0.5},
            glide_years=0,
        )},
        {_BENCH: ["steady"], "steady": [_BENCH, "wild"], "wild": ["steady"]},
        sensitivity=(),
    ), modern, century, seed=11)
    assert (full.status, full.selected_id) == (gamma1_only.status, gamma1_only.selected_id)


def test_log_utility_equals_geometric_mean() -> None:
    """Gamma-1 certainty equivalents equal the geometric mean of paired ratios."""
    months = _months(24)
    spec = _spec(
        {_BENCH: _mix(1.0, 0.0), "spy0_qqq100": _mix(0.0, 1.0)},
        {_BENCH: [], "spy0_qqq100": []},
        horizons=(1,),
    )
    sleeves = {"SPY": (0.0,) * 24, "QQQ": (0.02,) * 12 + (0.04,) * 12}
    modern = _panel(months, dict(sleeves), "modern")
    century = _panel(months, dict(sleeves), "century")
    report = evaluate_pension_decision(spec, modern, century, seed=11)
    score = next(item for item in report.scores if item.candidate_id == "spy0_qqq100" and item.tier == "modern")
    assert score.gamma == pytest.approx(1.0)
    assert score.cohort_count == 2
    assert score.ce_ratio == pytest.approx((1.02 * 1.04) ** 6)


def test_benchmark_ratios_are_exactly_one() -> None:
    """Every benchmark cell ratio equals 1.0 exactly."""
    months = _months(24)
    spec = _spec(
        {_BENCH: _mix(1.0, 0.0), "spy50_qqq50": _mix(0.5, 0.5)},
        {_BENCH: ["spy50_qqq50"], "spy50_qqq50": [_BENCH]},
    )
    modern = _flat_panel(months, 0.004, 0.002, "modern")
    century = _flat_panel(months, 0.004, 0.002, "century")
    report = evaluate_pension_decision(spec, modern, century, seed=11)
    bench_scores = [item for item in report.scores if item.candidate_id == _BENCH]
    assert bench_scores
    for item in bench_scores:
        assert item.ce_ratio == 1.0
        assert item.median_ratio == 1.0
        assert item.worst_ratio == 1.0


def test_evaluation_is_deterministic_under_seed() -> None:
    """Identical inputs and seed give identical reports."""
    months = _months(48)
    spec = _spec(
        {_BENCH: _mix(1.0, 0.0), "spy50_qqq50": _mix(0.5, 0.5)},
        {_BENCH: ["spy50_qqq50"], "spy50_qqq50": [_BENCH]},
        horizons=(2, 3),
        paths=20,
    )
    modern = _flat_panel(months, 0.005, 0.0052, "modern")
    century = _flat_panel(months, 0.005, 0.0052, "century")
    first = evaluate_pension_decision(spec, modern, century, seed=11)
    second = evaluate_pension_decision(spec, modern, century, seed=11)
    assert first == second


def _neutrality_report() -> PensionDecisionReport:
    from src.validation.pension_decision import CandidateScore

    scores = (
        CandidateScore("aaa", "modern", 2, 1.0, 2.0, 2.0, 2.0, 3, 0.0),
        CandidateScore("bbb", "modern", 2, 1.0, 1.0, 1.0, 1.0, 3, 0.0),
    )
    return PensionDecisionReport(
        name="probe",
        status="ADOPT_CANDIDATE",
        selected_id="aaa",
        equivalent_ids=("aaa",),
        reasons=(),
        scores=scores,
        robust_scores={"aaa": 2.0, "bbb": 1.0},
        sensitivity_robust_scores={},
        bootstrap_win_share={"aaa": 1.0, "bbb": 0.0},
        tax_rank_agreement=True,
        manifest_hashes={},
        trial_count=4,
        dominance_min_ratio={},
        dominance_min_ratio_by_tier={},
        guard_excluded_ids=(),
        control_scores={},
        control_vs_reference={},
    )


def test_tax_rank_neutrality_detects_disagreement() -> None:
    """Opposite after-tax ordering returns False; matching ordering returns True."""
    arm_map = {"arm_a": "aaa", "arm_b": "bbb"}
    flipped = [
        {"arm_id": "arm_a", "horizon_months": 24, "median_wealth_ratio": 1.0},
        {"arm_id": "arm_b", "horizon_months": 24, "median_wealth_ratio": 2.0},
    ]
    assert assert_tax_rank_neutrality(_neutrality_report(), flipped, arm_map) is False
    aligned = [
        {"arm_id": "arm_a", "horizon_months": 24, "median_wealth_ratio": 2.0},
        {"arm_id": "arm_b", "horizon_months": 24, "median_wealth_ratio": 1.0},
    ]
    assert assert_tax_rank_neutrality(_neutrality_report(), aligned, arm_map) is True


def test_evaluation_rejects_unknown_sleeves_and_incumbent() -> None:
    """Panels lacking candidate sleeves and unknown incumbents fail closed."""
    months = _months(24)
    spec = _spec(
        {_BENCH: _mix(1.0, 0.0), "spy50_qqq50": _mix(0.5, 0.5)},
        {_BENCH: ["spy50_qqq50"], "spy50_qqq50": [_BENCH]},
    )
    thin = _panel(months, {"SPY": (0.005,) * 24}, "modern")
    full = _flat_panel(months, 0.005, 0.0052, "century")
    with pytest.raises(ValueError, match="lacks sleeves"):
        evaluate_pension_decision(spec, thin, full, seed=11)
    modern = _flat_panel(months, 0.005, 0.0052, "modern")
    with pytest.raises(ValueError, match="incumbent_id"):
        evaluate_pension_decision(spec, modern, full, seed=11, incumbent_id="ghost")


def test_evaluation_applies_universe_drag_per_schedule() -> None:
    """A universe-wide drag map prices each schedule only on its own sleeves."""
    from dataclasses import replace as _replace

    dividend = WeightSchedule(
        start_weights={"SCHD": 0.2, "QQQ": 0.8},
        end_weights={"SCHD": 0.2, "QQQ": 0.8},
        glide_years=0,
    )
    world = WeightSchedule(
        start_weights={"VT": 0.2, "QQQ": 0.8},
        end_weights={"VT": 0.2, "QQQ": 0.8},
        glide_years=0,
    )
    spec = _replace(
        _spec(
            {_BENCH: _mix(1.0, 0.0), "schd20_qqq80": dividend},
            {_BENCH: ["schd20_qqq80"], "schd20_qqq80": [_BENCH]},
            horizons=(2,),
            paths=10,
            controls={"vt20_qqq80": world},
        ),
        annual_drag_by_sleeve={"SPY": 0.001, "QQQ": 0.001, "SCHD": 0.002, "VT": 0.003},
    )
    months = _months(48)
    modern = _panel(
        months,
        {
            "SPY": (0.005,) * 48,
            "QQQ": (0.0052,) * 48,
            "SCHD": (0.0051,) * 48,
            "VT": (0.0049,) * 48,
        },
        "modern",
    )
    century = _panel(
        months,
        {
            "SPY": (0.005,) * 48,
            "QQQ": (0.0052,) * 48,
            "SCHD": (0.0051,) * 48,
        },
        "century",
    )
    report = evaluate_pension_decision(spec, modern, century, seed=11)
    assert set(report.robust_scores) == {_BENCH, "schd20_qqq80"}
    assert set(report.control_scores) == {"vt20_qqq80"}


def test_evaluation_rejects_drag_naming_unknown_sleeve() -> None:
    """A drag naming a sleeve no candidate or control uses fails closed."""
    from dataclasses import replace as _replace

    spec = _replace(
        _spec({_BENCH: _mix(1.0, 0.0)}, {_BENCH: []}, horizons=(2,), paths=10),
        annual_drag_by_sleeve={"BOND": 0.01},
    )
    months = _months(48)
    modern = _flat_panel(months, 0.005, 0.0052, "modern")
    century = _flat_panel(months, 0.005, 0.0052, "century")
    with pytest.raises(ValueError, match="unknown sleeves"):
        evaluate_pension_decision(spec, modern, century, seed=11)


def test_evaluation_skips_infeasible_horizons() -> None:
    """Horizons longer than a tier panel are skipped, not spliced."""
    spec = _spec(
        {_BENCH: _mix(1.0, 0.0), "spy0_qqq100": _mix(0.0, 1.0)},
        {_BENCH: [], "spy0_qqq100": []},
        horizons=(2, 3),
        paths=10,
    )
    modern = _flat_panel(_months(30), 0.005, 0.006, "modern")
    century = _flat_panel(_months(48), 0.005, 0.006, "century")
    report = evaluate_pension_decision(spec, modern, century, seed=11)
    cells = {(score.tier, score.horizon_years) for score in report.scores}
    assert ("modern", 3) not in cells
    assert ("century", 3) in cells
    assert ("modern", 2) in cells
    assert report.status == "ADOPT_CANDIDATE"


def test_evaluation_fails_without_feasible_cells() -> None:
    """No feasible tier-horizon cell and an oversized bootstrap horizon fail closed."""
    months = _months(24)
    spec = _spec(
        {_BENCH: _mix(1.0, 0.0), "spy0_qqq100": _mix(0.0, 1.0)},
        {_BENCH: [], "spy0_qqq100": []},
        horizons=(5,),
        paths=10,
    )
    modern = _flat_panel(months, 0.005, 0.006, "modern")
    century = _flat_panel(months, 0.005, 0.006, "century")
    with pytest.raises(ValueError, match="no feasible tier-horizon cell"):
        evaluate_pension_decision(spec, modern, century, seed=11)
    roomy = _flat_panel(_months(150), 0.005, 0.006, "modern")
    thin_century = _flat_panel(months, 0.005, 0.006, "century")
    with pytest.raises(ValueError, match="bootstrap horizon"):
        evaluate_pension_decision(spec, roomy, thin_century, seed=11)
    from dataclasses import replace as _replace

    bad_lineage = _replace(
        _spec({_BENCH: _mix(1.0, 0.0)}, {_BENCH: []}, horizons=(2,), paths=10),
        lineage={"related_trial_count": "many"},
    )
    with pytest.raises(ValueError, match="related_trial_count"):
        evaluate_pension_decision(
            bad_lineage,
            _flat_panel(months, 0.005, 0.006, "modern"),
            _flat_panel(months, 0.005, 0.006, "century"),
            seed=11,
        )


def test_tax_rank_neutrality_rejects_sparse_evidence() -> None:
    """Missing horizons, candidates, or malformed summaries cannot confirm neutrality."""
    arm_map = {"arm_a": "aaa", "arm_b": "bbb"}
    report = _neutrality_report()
    other_horizon = [{"arm_id": "arm_a", "horizon_months": 12, "median_wealth_ratio": 2.0}]
    assert assert_tax_rank_neutrality(report, other_horizon, arm_map) is False
    partial = [{"arm_id": "arm_a", "horizon_months": 24, "median_wealth_ratio": 2.0}]
    assert assert_tax_rank_neutrality(report, partial, arm_map) is False
    malformed: list[dict[str, object]] = [
        {"arm_id": "arm_a", "horizon_months": 24, "median_wealth_ratio": 2.0},
        {"arm_id": "arm_b", "horizon_months": 24},
        {"arm_id": "ghost", "horizon_months": 24, "median_wealth_ratio": 9.0},
        {"arm_id": "arm_b", "horizon_months": 24, "median_wealth_ratio": None},
    ]
    assert assert_tax_rank_neutrality(report, malformed, arm_map) is False


def _alt_mix(alt: float, spy: float = 0.0) -> WeightSchedule:
    return WeightSchedule(
        start_weights={"SPY": spy, "ALT": alt, "QQQ": 1.0 - spy - alt},
        end_weights={"SPY": spy, "ALT": alt, "QQQ": 1.0 - spy - alt},
        glide_years=0,
    )


def _guard_panels(alt_modern: float, alt_century: float) -> tuple[MonthlyReturnPanel, MonthlyReturnPanel]:
    months = _months(24)
    modern = _panel(
        months,
        {"SPY": (0.005,) * 24, "QQQ": (0.002,) * 24, "ALT": (alt_modern,) * 24},
        "modern",
    )
    century = _panel(
        months,
        {"SPY": (0.009,) * 24, "QQQ": (0.008,) * 24, "ALT": (alt_century,) * 24},
        "century",
    )
    return modern, century


def _guard_spec(alt: float = 1.0, spy: float = 0.0) -> PensionDecisionSpec:
    return _spec(
        {
            "bench": _mix(0.0, 1.0),
            "ref": _mix(1.0, 0.0),
            "tilt": _alt_mix(alt, spy),
        },
        {"bench": [], "ref": [], "tilt": []},
        horizons=(1,),
        paths=10,
        reference="ref",
        benchmark="bench",
    )


def test_no_reference_reproduces_v1_verdict() -> None:
    """Without a reference or controls the verdict matches v1 and new fields stay empty."""
    months = _months(48)
    spec = _spec(
        {
            _BENCH: _mix(1.0, 0.0),
            "spy50_qqq50": _mix(0.5, 0.5),
            "spy20_qqq80": _mix(0.2, 0.8),
            "spy0_qqq100": _mix(0.0, 1.0),
        },
        {
            _BENCH: ["spy50_qqq50"],
            "spy50_qqq50": [_BENCH, "spy20_qqq80"],
            "spy20_qqq80": ["spy50_qqq50", "spy0_qqq100"],
            "spy0_qqq100": ["spy20_qqq80"],
        },
        horizons=(2, 3),
        paths=20,
    )
    modern = _flat_panel(months, 0.005, 0.0052, "modern")
    century = _flat_panel(months, 0.005, 0.0052, "century")
    report = evaluate_pension_decision(spec, modern, century, seed=11)
    assert report.status == "ADOPT_CANDIDATE"
    assert set(report.equivalent_ids) == {_BENCH, "spy50_qqq50", "spy20_qqq80", "spy0_qqq100"}
    assert report.selected_id == "spy0_qqq100"
    assert report.dominance_min_ratio == {}
    assert report.dominance_min_ratio_by_tier == {}
    assert report.guard_excluded_ids == ()
    assert report.control_scores == {}
    assert report.control_vs_reference == {}


def test_guard_excludes_century_only_winner() -> None:
    """A tilt winning the century tier but losing the modern tier is guarded out."""
    modern, century = _guard_panels(0.00334, 0.0135)
    report = evaluate_pension_decision(_guard_spec(), modern, century, seed=11)
    assert report.dominance_min_ratio["tilt"] == pytest.approx(0.9804, abs=1e-4)
    assert "tilt" in report.guard_excluded_ids
    assert report.selected_id == "ref"
    assert "DOMINANCE_GUARD" in report.reasons


def test_guard_admits_dominant_challenger() -> None:
    """A challenger at or above the reference in every cell stays eligible and wins."""
    modern, century = _guard_panels(0.006, 0.010)
    report = evaluate_pension_decision(_guard_spec(), modern, century, seed=11)
    assert "tilt" not in report.guard_excluded_ids
    assert report.selected_id == "tilt"
    assert "DOMINANCE_GUARD" not in report.reasons


def test_guard_within_band_loss_stays_eligible() -> None:
    """A 0.3% loss in one cell is inside the band and keeps the challenger eligible."""
    modern, century = _guard_panels(0.00475, 0.0135)
    report = evaluate_pension_decision(_guard_spec(), modern, century, seed=11)
    assert report.dominance_min_ratio["tilt"] == pytest.approx(0.997, abs=1e-3)
    assert "tilt" not in report.guard_excluded_ids
    assert report.selected_id == "tilt"


def test_guard_reference_always_eligible() -> None:
    """With every other candidate excluded the reference is still selected."""
    months = _months(24)
    modern = _panel(
        months,
        {"SPY": (0.005,) * 24, "QQQ": (0.002,) * 24, "ALT": (0.00334,) * 24},
        "modern",
    )
    century = _panel(
        months,
        {"SPY": (0.009,) * 24, "QQQ": (0.008,) * 24, "ALT": (0.0135,) * 24},
        "century",
    )
    spec = _spec(
        {
            "bench": _mix(0.0, 1.0),
            "ref": _mix(1.0, 0.0),
            "tilt1": _alt_mix(1.0),
            "tilt2": _alt_mix(0.5, 0.5),
        },
        {"bench": [], "ref": [], "tilt1": [], "tilt2": []},
        horizons=(1,),
        paths=10,
        reference="ref",
        benchmark="bench",
    )
    report = evaluate_pension_decision(spec, modern, century, seed=11)
    assert set(report.guard_excluded_ids) >= {"tilt1", "tilt2"}
    assert report.selected_id == "ref"


def test_guard_controls_never_selected() -> None:
    """A control with the best modern score is reported but never selected."""
    months = _months(24)
    modern = _panel(
        months,
        {"SPY": (0.005,) * 24, "QQQ": (0.006,) * 24, "VT": (0.02,) * 24},
        "modern",
    )
    century = _panel(months, {"SPY": (0.005,) * 24, "QQQ": (0.006,) * 24}, "century")
    control = WeightSchedule(start_weights={"VT": 1.0}, end_weights={"VT": 1.0}, glide_years=0)
    spec = _spec(
        {_BENCH: _mix(1.0, 0.0), "spy0_qqq100": _mix(0.0, 1.0)},
        {_BENCH: [], "spy0_qqq100": []},
        horizons=(1,),
        paths=10,
        controls={"world": control},
    )
    report = evaluate_pension_decision(spec, modern, century, seed=11)
    assert report.control_scores["world"] > 1.0
    assert "world" not in report.robust_scores
    assert "world" not in report.equivalent_ids
    assert "world" not in report.bootstrap_win_share
    assert report.control_vs_reference == {}


def test_guard_control_without_century_proxy_scores_modern_only() -> None:
    """A control sleeve missing from the century panel evaluates without error."""
    months = _months(24)
    modern = _panel(months, {"SPY": (0.005,) * 24, "QQQ": (0.005,) * 24, "VT": (0.006,) * 24}, "modern")
    century = _panel(months, {"SPY": (0.005,) * 24, "QQQ": (0.005,) * 24}, "century")
    control = WeightSchedule(start_weights={"VT": 1.0}, end_weights={"VT": 1.0}, glide_years=0)
    spec = _spec(
        {"bench": _mix(1.0, 0.0)},
        {"bench": []},
        horizons=(1,),
        paths=10,
        reference="bench",
        benchmark="bench",
        controls={"world": control},
    )
    report = evaluate_pension_decision(spec, modern, century, seed=11)
    assert report.control_scores["world"] == pytest.approx(report.control_vs_reference["world"])


def test_guard_trial_count_includes_controls() -> None:
    """Lineage plus candidates plus controls equals the trial count."""
    months = _months(24)
    modern = _panel(
        months,
        {"SPY": (0.005,) * 24, "QQQ": (0.006,) * 24, "VT": (0.007,) * 24},
        "modern",
    )
    century = _flat_panel(months, 0.005, 0.006, "century")
    control = WeightSchedule(start_weights={"VT": 1.0}, end_weights={"VT": 1.0}, glide_years=0)
    spec = _spec(
        {_BENCH: _mix(1.0, 0.0), "spy50_qqq50": _mix(0.5, 0.5), "spy0_qqq100": _mix(0.0, 1.0)},
        {_BENCH: [], "spy50_qqq50": [], "spy0_qqq100": []},
        horizons=(1,),
        paths=10,
        lineage=4,
        controls={"world": control},
    )
    report = evaluate_pension_decision(spec, modern, century, seed=11)
    assert report.trial_count == 8


def test_evaluation_rejects_unknown_reference_and_century_gap() -> None:
    """A hand-built reference outside candidates and a thin century panel fail closed."""
    months = _months(24)
    modern = _flat_panel(months, 0.005, 0.0052, "modern")
    century = _flat_panel(months, 0.005, 0.0052, "century")
    bad_reference = _spec(
        {_BENCH: _mix(1.0, 0.0)},
        {_BENCH: []},
        horizons=(1,),
        paths=10,
        reference="ghost",
    )
    with pytest.raises(ValueError, match="dominance_reference_id"):
        evaluate_pension_decision(bad_reference, modern, century, seed=11)
    thin_century = _panel(months, {"SPY": (0.005,) * 24}, "century")
    spec = _spec({_BENCH: _mix(1.0, 0.0), "spy50_qqq50": _mix(0.5, 0.5)}, {_BENCH: ["spy50_qqq50"], "spy50_qqq50": [_BENCH]})
    with pytest.raises(ValueError, match="century panel lacks"):
        evaluate_pension_decision(spec, modern, thin_century, seed=11)


def _veto_spec() -> PensionDecisionSpec:
    return _spec(
        {
            "bench": _mix(0.0, 1.0),
            "ref": _mix(1.0, 0.0),
            "tilt": _alt_mix(1.0),
        },
        {"bench": [], "ref": [], "tilt": []},
        horizons=(1,),
        paths=10,
        reference="ref",
        benchmark="bench",
        realized_horizons=(1,),
    )


def _veto_panels(realized_alt: float) -> tuple[MonthlyReturnPanel, MonthlyReturnPanel, MonthlyReturnPanel]:
    months = _months(24)
    modern = _panel(
        months,
        {"SPY": (0.006,) * 24, "QQQ": (0.001,) * 24, "ALT": (0.010,) * 24},
        "modern",
    )
    century = _panel(
        months,
        {"SPY": (0.004,) * 24, "QQQ": (0.002,) * 24, "ALT": (0.011,) * 24},
        "century",
    )
    realized = _panel(
        months,
        {"SPY": (0.006,) * 24, "QQQ": (0.001,) * 24, "ALT": (realized_alt,) * 24},
        "realized",
    )
    return modern, century, realized


def test_realized_veto_blocks_proxy_only_winner() -> None:
    """A challenger winning the proxy tiers but losing the realized tier is vetoed."""
    modern, century, realized = _veto_panels(0.0036)
    report = evaluate_pension_decision(_veto_spec(), modern, century, seed=11, realized=realized)
    assert "tilt" in report.guard_excluded_ids
    assert report.selected_id == "ref"
    assert report.status == "ADOPT_CANDIDATE"
    assert "DOMINANCE_GUARD" in report.reasons
    assert "REALIZED_VETO" in report.reasons
    assert report.dominance_min_ratio_by_tier["realized"]["tilt"] < 0.995
    assert report.dominance_min_ratio["tilt"] < 0.995


def test_realized_cells_enter_robust_minimum() -> None:
    """A candidate whose only weak cell is realized scores that cell as its robust."""
    months = _months(24)
    spec = _spec(
        {_BENCH: _mix(1.0, 0.0), "spy0_qqq100": _mix(0.0, 1.0)},
        {_BENCH: [], "spy0_qqq100": []},
        horizons=(1,),
        paths=10,
        realized_horizons=(1,),
    )
    sleeves = {"SPY": (0.004,) * 24, "QQQ": (0.008,) * 24}
    modern = _panel(months, dict(sleeves), "modern")
    century = _panel(months, dict(sleeves), "century")
    realized = _panel(months, {"SPY": (0.005,) * 24, "QQQ": (0.006,) * 24}, "realized")
    report = evaluate_pension_decision(spec, modern, century, seed=11, realized=realized)
    realized_cell = next(
        item for item in report.scores if item.candidate_id == "spy0_qqq100" and item.tier == "realized"
    )
    assert report.robust_scores["spy0_qqq100"] == pytest.approx(realized_cell.ce_ratio)
    assert report.dominance_min_ratio_by_tier == {}


def test_challenger_winning_every_tier_is_adopted() -> None:
    """A challenger at or above the reference in all three tiers stays eligible and wins."""
    modern, century, realized = _veto_panels(0.008)
    report = evaluate_pension_decision(_veto_spec(), modern, century, seed=11, realized=realized)
    assert "tilt" not in report.guard_excluded_ids
    assert report.selected_id == "tilt"
    assert report.status == "ADOPT_CANDIDATE"
    assert "DOMINANCE_GUARD" not in report.reasons
    assert "REALIZED_VETO" not in report.reasons


def test_realized_required_when_declared() -> None:
    """Declared realized horizons without a realized panel fail closed."""
    months = _months(24)
    spec = _spec(
        {_BENCH: _mix(1.0, 0.0), "spy0_qqq100": _mix(0.0, 1.0)},
        {_BENCH: [], "spy0_qqq100": []},
        horizons=(1,),
        paths=10,
        realized_horizons=(1,),
    )
    modern = _flat_panel(months, 0.005, 0.0052, "modern")
    century = _flat_panel(months, 0.005, 0.0052, "century")
    with pytest.raises(ValueError, match="realized panel is required"):
        evaluate_pension_decision(spec, modern, century, seed=11)


def test_realized_rejected_when_undeclared() -> None:
    """A realized panel without declared realized horizons fails closed."""
    months = _months(24)
    spec = _spec(
        {_BENCH: _mix(1.0, 0.0), "spy0_qqq100": _mix(0.0, 1.0)},
        {_BENCH: [], "spy0_qqq100": []},
        horizons=(1,),
        paths=10,
    )
    modern = _flat_panel(months, 0.005, 0.0052, "modern")
    century = _flat_panel(months, 0.005, 0.0052, "century")
    realized = _flat_panel(months, 0.005, 0.0052, "realized")
    with pytest.raises(ValueError, match="while realized_horizons_years is empty"):
        evaluate_pension_decision(spec, modern, century, seed=11, realized=realized)


def test_infeasible_realized_horizon_fails_closed() -> None:
    """A realized horizon longer than the realized panel raises instead of skipping."""
    months = _months(24)
    spec = _spec(
        {_BENCH: _mix(1.0, 0.0), "spy0_qqq100": _mix(0.0, 1.0)},
        {_BENCH: [], "spy0_qqq100": []},
        horizons=(1,),
        paths=10,
        realized_horizons=(2,),
    )
    modern = _flat_panel(months, 0.005, 0.0052, "modern")
    century = _flat_panel(months, 0.005, 0.0052, "century")
    short = _flat_panel(_months(12), 0.005, 0.0052, "realized")
    with pytest.raises(ValueError, match="realized horizon"):
        evaluate_pension_decision(spec, modern, century, seed=11, realized=short)
    mistiered = _flat_panel(months, 0.005, 0.0052, "modern")
    with pytest.raises(ValueError, match="must be 'realized'"):
        evaluate_pension_decision(spec, modern, century, seed=11, realized=mistiered)
    thin = _panel(months, {"SPY": (0.005,) * 24}, "realized")
    with pytest.raises(ValueError, match="realized panel lacks"):
        evaluate_pension_decision(spec, modern, century, seed=11, realized=thin)


def test_controls_skip_realized_tier() -> None:
    """A control sleeve absent from the realized panel scores modern only without error."""
    months = _months(24)
    modern = _panel(
        months, {"SPY": (0.005,) * 24, "QQQ": (0.006,) * 24, "VT": (0.006,) * 24}, "modern"
    )
    century = _flat_panel(months, 0.005, 0.006, "century")
    realized = _flat_panel(months, 0.005, 0.006, "realized")
    control = WeightSchedule(start_weights={"VT": 1.0}, end_weights={"VT": 1.0}, glide_years=0)
    spec = _spec(
        {"bench": _mix(1.0, 0.0)},
        {"bench": []},
        horizons=(1,),
        paths=10,
        reference="bench",
        benchmark="bench",
        controls={"world": control},
        realized_horizons=(1,),
    )
    report = evaluate_pension_decision(spec, modern, century, seed=11, realized=realized)
    assert report.control_scores["world"] == pytest.approx(report.control_vs_reference["world"])
    assert "world" not in report.robust_scores
    assert {score.tier for score in report.scores} == {"modern", "century", "realized"}


def test_without_realized_tier_reports_match_prior_behavior() -> None:
    """Without realized horizons the verdict equals the pre-change expectations."""
    months = _months(48)
    spec = _spec(
        {
            _BENCH: _mix(1.0, 0.0),
            "spy50_qqq50": _mix(0.5, 0.5),
            "spy20_qqq80": _mix(0.2, 0.8),
            "spy0_qqq100": _mix(0.0, 1.0),
        },
        {
            _BENCH: ["spy50_qqq50"],
            "spy50_qqq50": [_BENCH, "spy20_qqq80"],
            "spy20_qqq80": ["spy50_qqq50", "spy0_qqq100"],
            "spy0_qqq100": ["spy20_qqq80"],
        },
        horizons=(2, 3),
        paths=20,
    )
    modern = _flat_panel(months, 0.005, 0.0052, "modern")
    century = _flat_panel(months, 0.005, 0.0052, "century")
    report = evaluate_pension_decision(spec, modern, century, seed=11)
    assert report.status == "ADOPT_CANDIDATE"
    assert set(report.equivalent_ids) == {_BENCH, "spy50_qqq50", "spy20_qqq80", "spy0_qqq100"}
    assert report.selected_id == "spy0_qqq100"
    assert report.trial_count == 6
    assert report.dominance_min_ratio_by_tier == {}
    assert all(score.tier in {"modern", "century"} for score in report.scores)


def test_fee_drag_breaks_exact_drawdown_ties() -> None:
    """Tied drawdowns within the band select the candidate with the lowest fee drag.

    Fee drags price into net wealth, so exactly equal scores with different drags are
    unreachable; instead the high-gross/high-fee leader stays within the band of the
    low-fee runner-up, drawdowns tie exactly at zero, and the fee level flips the
    selection away from the lexicographically smallest id.
    """
    from dataclasses import replace as _replace

    months = _months(24)
    alt_half = WeightSchedule(
        start_weights={"SPY": 0.5, "ALT": 0.5},
        end_weights={"SPY": 0.5, "ALT": 0.5},
        glide_years=0,
    )
    spec = _replace(
        _spec(
            {
                "bench": _mix(1.0, 0.0),
                "aaa": alt_half,
                "zzz": _mix(0.5, 0.5),
            },
            {"bench": [], "aaa": [], "zzz": []},
            horizons=(1,),
            paths=10,
            pre=0,
            benchmark="bench",
        ),
        annual_drag_by_sleeve={"SPY": 0.001, "QQQ": 0.001, "ALT": 0.02},
    )
    sleeves = {"SPY": (0.005,) * 24, "QQQ": (0.006,) * 24, "ALT": (0.008,) * 24}
    modern = _panel(months, dict(sleeves), "modern")
    century = _panel(months, dict(sleeves), "century")
    report = evaluate_pension_decision(spec, modern, century, seed=11)
    assert report.robust_scores["aaa"] > report.robust_scores["zzz"] > 1.0
    assert report.robust_scores["aaa"] - report.robust_scores["zzz"] < 0.005
    assert report.selected_id == "zzz"
    assert report.status == "ADOPT_CANDIDATE"


def test_reference_wins_full_ties() -> None:
    """Identical score, drawdown, and drag resolve to the reference holding."""
    months = _months(24)
    spec = _spec(
        {
            "bench": _mix(1.0, 0.0),
            "ref": _mix(1.0, 0.0),
            "twin": _mix(0.0, 1.0),
        },
        {"bench": [], "ref": [], "twin": []},
        horizons=(1,),
        paths=10,
        pre=0,
        floor=0.0,
        reference="ref",
        benchmark="bench",
    )
    sleeves = {"SPY": (0.005,) * 24, "QQQ": (0.005,) * 24}
    modern = _panel(months, dict(sleeves), "modern")
    century = _panel(months, dict(sleeves), "century")
    report = evaluate_pension_decision(spec, modern, century, seed=11)
    assert report.guard_excluded_ids == ()
    assert report.selected_id == "ref"
    assert "DOMINANCE_GUARD" not in report.reasons
