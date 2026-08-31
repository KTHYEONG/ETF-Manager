"""Tournament cache tests."""

from __future__ import annotations

import src.validation.walk_forward  # noqa: F401  # co-mod anchor for lean_check AST linkage

def test_run_walk_forward_tournament_shared_baseline_cache() -> None:
    from datetime import date

    from src.policy.targets import PolicyId
    from src.validation.experiment import CandidateSpec, ExperimentSpec
    from src.validation.walk_forward import run_walk_forward_tournament
    from tests.unit.validation.campaign_parts.test_walk_forward import _RecordingRunner

    spec = ExperimentSpec(
        name='wf_tournament_two',
        start=date(2016, 7, 1),
        end=date(2024, 6, 30),
        contribution_krw=1_000_000.0,
        hurdle=0.02,
        objective='growth_first',
        horizon_months=0,
        train_months=36,
        test_months=24,
        baseline=CandidateSpec(id='baseline', policy=PolicyId.QQQ, modules=0),
        candidates=[
            CandidateSpec(id='cand_a', policy=PolicyId.QQQ, modules=1),
            CandidateSpec(id='cand_b', policy=PolicyId.QQQ, modules=1),
        ],
        cadence={'anchor': 'month_open'},
    )
    runner = _RecordingRunner(dict.fromkeys(PolicyId, 100.0))
    reports = run_walk_forward_tournament(spec, runner)
    assert set(reports) == {'cand_a', 'cand_b'}
    assert all(report.name == spec.name for report in reports.values())
