from __future__ import annotations

import src.validation.strategy_selection  # noqa: F401  # co-mod anchor for lean_check AST linkage

def test_select_recommended_arm_picks_oos_eligible_max_gain() -> None:
    from src.validation.strategy_selection import StrategyArmRow, StrategyVerdict, select_recommended_arm

    rows = (
        StrategyArmRow(
            arm_id='soxx100_adaptive_v5',
            process_adopted_vs_baseline=False,
            pooled_oos_real_gain=9.0,
            pooled_oos_tw_ratio=1.05,
            in_sample_real_gain=10.0,
            verdict=StrategyVerdict.RESEARCH_ONLY,
            fold_count=3,
        ),
        StrategyArmRow(
            arm_id='qqq85_soxx15_adaptive_v5',
            process_adopted_vs_baseline=True,
            pooled_oos_real_gain=4.0,
            pooled_oos_tw_ratio=1.02,
            in_sample_real_gain=3.5,
            verdict=StrategyVerdict.OOS_ELIGIBLE,
            fold_count=3,
        ),
        StrategyArmRow(
            arm_id='qqq95_soxx5_adaptive_v5',
            process_adopted_vs_baseline=True,
            pooled_oos_real_gain=3.0,
            pooled_oos_tw_ratio=1.01,
            in_sample_real_gain=3.0,
            verdict=StrategyVerdict.OOS_ELIGIBLE,
            fold_count=3,
        ),
    )
    arm_id, reason = select_recommended_arm(rows, baseline_arm_id='qqq90_soxx10_adaptive_v5')
    assert arm_id == 'qqq85_soxx15_adaptive_v5'
    assert 'oos_eligible' in reason


def test_select_recommended_arm_falls_back_to_baseline() -> None:
    from src.validation.strategy_selection import StrategyArmRow, StrategyVerdict, select_recommended_arm

    rows = (
        StrategyArmRow(
            arm_id='soxx100_adaptive_v5',
            process_adopted_vs_baseline=False,
            pooled_oos_real_gain=8.0,
            pooled_oos_tw_ratio=1.04,
            in_sample_real_gain=9.0,
            verdict=StrategyVerdict.RESEARCH_ONLY,
            fold_count=3,
        ),
    )
    arm_id, reason = select_recommended_arm(rows, baseline_arm_id='qqq90_soxx10_adaptive_v5')
    assert arm_id == 'qqq90_soxx10_adaptive_v5'
    assert 'no_oos_eligible' in reason


def test_run_strategy_selection_integration_mock() -> None:
    from datetime import date

    from src.policy.targets import PolicyId
    from src.validation.experiment import CandidateSpec, ExperimentSpec
    from src.validation.strategy_selection import run_strategy_selection
    from src.validation.walk_forward import CampaignReport, FoldOutcome

    spec = ExperimentSpec(
        name='sel_mock',
        start=date(2016, 7, 1),
        end=date(2024, 6, 30),
        contribution_krw=1_000_000.0,
        hurdle=0.02,
        objective='compound_growth',
        horizon_months=0,
        train_months=36,
        test_months=24,
        baseline=CandidateSpec(id='qqq90_soxx10_adaptive_v5', policy=PolicyId.QQQ, modules=1),
        candidates=[
            CandidateSpec(id='soxx100_adaptive_v5', policy=PolicyId.QQQ, modules=2, targets={'SOXX': 1.0}),
            CandidateSpec(id='qqq85_soxx15_adaptive_v5', policy=PolicyId.QQQ, modules=2, targets={'QQQ': 0.85, 'SOXX': 0.15}),
        ],
    )
    fold = FoldOutcome(
        train_start=date(2016, 7, 1),
        train_end=date(2019, 6, 30),
        test_start=date(2019, 7, 1),
        test_end=date(2021, 6, 30),
        train_adopted=True,
        chosen_policy=PolicyId.QQQ,
        baseline_test_wealth=100.0,
        candidate_test_wealth=110.0,
        chosen_test_wealth=110.0,
        baseline_real_gain=10.0,
        candidate_real_gain=20.0,
        chosen_real_gain=20.0,
        baseline_xirr_real=0.1,
        candidate_xirr_real=0.12,
        chosen_xirr_real=0.12,
    )

    def _wf_runner(inner: ExperimentSpec) -> CampaignReport:
        adopted = inner.candidates[0].id != 'soxx100_adaptive_v5'
        return CampaignReport(
            name=inner.name,
            candidate_id=inner.candidates[0].id,
            modules=inner.candidates[0].modules,
            folds=(fold,),
            baseline_test_ce={2.0: 100.0, 5.0: 100.0, 10.0: 100.0},
            candidate_test_ce={2.0: 110.0, 5.0: 110.0, 10.0: 110.0},
            chosen_test_ce={2.0: 110.0, 5.0: 110.0, 10.0: 110.0},
            process_adopted_vs_baseline=adopted,
        )

    report = run_strategy_selection(spec, _wf_runner)
    assert report.operational_unlock is False
    assert report.in_sample_champion_arm_id is None
    assert 'soxx100_adaptive_v5' not in report.oos_eligible_arm_ids
    assert report.recommended_arm_id == 'qqq85_soxx15_adaptive_v5'
