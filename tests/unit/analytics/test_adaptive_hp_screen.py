"""Unit tests for the reporting-only adaptive HP screen."""

from __future__ import annotations

from dataclasses import replace
from datetime import date

import pytest

from src.analytics.adaptive_hp_screen import (
    build_adaptive_hp_experiment,
    screen_adaptive_contribution_hp,
)
from src.policy.adaptive_contribution import (
    OPERATIONAL_ADAPTIVE_CONTRIBUTION,
    AdaptiveContributionConfig,
)
from src.policy.targets import PolicyId
from src.validation.campaign import CampaignReport, FoldOutcome
from src.validation.experiment import (
    ExperimentSpec,
    resolve_adaptive_contribution,
    resolve_baseline_adaptive_contribution,
)

_FOLD = FoldOutcome(
    train_start=date(2015, 6, 1),
    train_end=date(2020, 5, 31),
    test_start=date(2020, 6, 1),
    test_end=date(2023, 5, 31),
    train_adopted=True,
    chosen_policy=PolicyId.QQQ,
    baseline_test_wealth=100.0,
    candidate_test_wealth=100.0,
    chosen_test_wealth=100.0,
)


def _campaign_report(
    *,
    process_adopted: bool,
    chosen_wealth: float,
    baseline_wealth: float = 100.0,
) -> CampaignReport:
    fold = replace(
        _FOLD,
        baseline_test_wealth=baseline_wealth,
        candidate_test_wealth=chosen_wealth,
        chosen_test_wealth=chosen_wealth,
    )
    return CampaignReport(
        name="stub",
        candidate_id="stub",
        modules=1,
        folds=(fold,),
        baseline_test_ce={0.0: baseline_wealth},
        candidate_test_ce={0.0: chosen_wealth},
        chosen_test_ce={0.0: chosen_wealth},
        process_adopted_vs_baseline=process_adopted,
    )


def _candidate(*, dispersion: float = 1.4) -> AdaptiveContributionConfig:
    return replace(OPERATIONAL_ADAPTIVE_CONTRIBUTION, dispersion=dispersion)


def _matches_v5(cfg: AdaptiveContributionConfig) -> bool:
    lock = OPERATIONAL_ADAPTIVE_CONTRIBUTION
    return (
        cfg.rank_window == lock.rank_window
        and cfg.downside_power == lock.downside_power
        and cfg.upside_power == lock.upside_power
        and cfg.dispersion == lock.dispersion
        and cfg.neutral_deadband == lock.neutral_deadband
        and cfg.include_vol_dampener == lock.include_vol_dampener
    )


@pytest.mark.parametrize("scenario_id", ["AHS-A-build-spec"])
def test_ahs_a_build_spec(scenario_id: str) -> None:
    """AHS-A-build-spec"""
    candidate = _candidate(dispersion=1.4)
    spec = build_adaptive_hp_experiment(
        name="ahs_a",
        candidate=candidate,
        baseline=OPERATIONAL_ADAPTIVE_CONTRIBUTION,
        contribution_krw=1_000_000.0,
    )
    assert spec.objective == "adaptive_growth"
    assert spec.train_months == 60
    assert spec.test_months == 36
    assert spec.start == date(2015, 6, 1)
    assert spec.end == date(2026, 6, 30)
    assert spec.hurdle == pytest.approx(0.02)
    assert spec.candidates[0].policy == PolicyId.QQQ
    assert resolve_adaptive_contribution(spec) is not None
    assert resolve_adaptive_contribution(spec).dispersion == pytest.approx(1.4)
    baseline = resolve_baseline_adaptive_contribution(spec)
    assert baseline is not None
    lock = OPERATIONAL_ADAPTIVE_CONTRIBUTION
    assert baseline.rank_window == lock.rank_window
    assert baseline.downside_power == pytest.approx(lock.downside_power)
    assert baseline.upside_power == pytest.approx(lock.upside_power)
    assert baseline.dispersion == pytest.approx(lock.dispersion)
    assert baseline.neutral_deadband == pytest.approx(lock.neutral_deadband)
    assert baseline.include_vol_dampener == lock.include_vol_dampener


@pytest.mark.parametrize("scenario_id", ["AHS-B-unlock-false"])
def test_ahs_b_unlock_false(scenario_id: str) -> None:
    """AHS-B-unlock-false"""

    def wf_runner(spec: ExperimentSpec) -> CampaignReport:
        candidate = resolve_adaptive_contribution(spec)
        assert candidate is not None
        if _matches_v5(candidate):
            return _campaign_report(process_adopted=False, chosen_wealth=100.0)
        return _campaign_report(process_adopted=True, chosen_wealth=105.0)

    report = screen_adaptive_contribution_hp(
        contribution_krw=1_000_000.0,
        wf_runner=wf_runner,
        max_evaluations=5,
        parallel_workers=1,
    )
    assert report.operational_unlock is False
    assert report.champion is not None


@pytest.mark.parametrize("scenario_id", ["AHS-C-champion-gate"])
def test_ahs_c_champion_gate(scenario_id: str) -> None:
    """AHS-C-champion-gate"""
    ratio_by_dispersion = {
        1.275: 1.02,
        1.35: 1.0,
        1.425: 1.08,
        1.5: 1.01,
    }

    def wf_runner(spec: ExperimentSpec) -> CampaignReport:
        candidate = resolve_adaptive_contribution(spec)
        assert candidate is not None
        ratio = ratio_by_dispersion.get(round(candidate.dispersion, 3), 0.99)
        adopted = ratio > 1.0 and round(candidate.dispersion, 3) != 1.5
        chosen = 100.0 * ratio
        return _campaign_report(process_adopted=adopted, chosen_wealth=chosen)

    report = screen_adaptive_contribution_hp(
        contribution_krw=1_000_000.0,
        wf_runner=wf_runner,
        max_evaluations=20,
        parallel_workers=1,
    )
    assert report.evaluations <= 20
    assert any(_matches_v5(row.candidate) for row in report.rows)
    if report.champion is not None:
        assert report.champion.process_adopted_vs_baseline is True
        assert report.champion.pooled_tw_ratio > 1.0
        eligible = [
            row
            for row in report.rows
            if row.process_adopted_vs_baseline and row.pooled_tw_ratio > 1.0
        ]
        assert report.champion.pooled_tw_ratio == max(row.pooled_tw_ratio for row in eligible)

    all_fail_report = screen_adaptive_contribution_hp(
        contribution_krw=1_000_000.0,
        wf_runner=lambda _spec: _campaign_report(process_adopted=False, chosen_wealth=95.0),
        max_evaluations=5,
        parallel_workers=1,
    )
    assert all_fail_report.champion is None


@pytest.mark.parametrize("scenario_id", ["AHS-D-budget-and-bounds"])
def test_ahs_d_budget_and_bounds(scenario_id: str) -> None:
    """AHS-D-budget-and-bounds"""
    with pytest.raises(ValueError, match="contribution_krw"):
        screen_adaptive_contribution_hp(
            contribution_krw=0.0,
            wf_runner=lambda _spec: _campaign_report(process_adopted=False, chosen_wealth=100.0),
        )
    with pytest.raises(ValueError, match="max_evaluations"):
        screen_adaptive_contribution_hp(
            contribution_krw=1_000_000.0,
            wf_runner=lambda _spec: _campaign_report(process_adopted=False, chosen_wealth=100.0),
            max_evaluations=0,
        )

    calls = 0

    def counting_runner(_spec: ExperimentSpec) -> CampaignReport:
        nonlocal calls
        calls += 1
        return _campaign_report(process_adopted=False, chosen_wealth=100.0)

    report = screen_adaptive_contribution_hp(
        contribution_krw=1_000_000.0,
        wf_runner=counting_runner,
        max_evaluations=5,
        parallel_workers=1,
    )
    assert calls <= 5
    assert report.evaluations <= 5
    for row in report.rows:
        cfg = row.candidate
        assert 2.0 <= cfg.downside_power <= 5.0
        assert 0.15 <= cfg.upside_power <= 0.70
        assert 0.9 <= cfg.dispersion <= 1.6
        assert 0.0 <= cfg.neutral_deadband <= 8.0
        assert cfg.rank_window == 126
        assert cfg.include_vol_dampener is False
