# ruff: noqa: F401, PT018, S101
"""Unit tests for compound DCA."""
from __future__ import annotations


def test_ccd_preregistered_arms() -> None:
    from datetime import date
    from src.analytics.compound_dca import (
        COMPOUND_DCA_ARM_IDS,
        COMPOUND_DCA_MIX_TARGETS,
        COMPOUND_DCA_ROLE_SWAP_TARGETS,
        COMPOUND_DCA_SOXX100_TARGETS,
        compare_compound_dca,
    )
    from src.policy.adaptive_contribution import OPERATIONAL_ADAPTIVE_CONTRIBUTION
    from src.policy.mix_risk_budget import OPERATIONAL_MIX_RISK_BUDGET
    from src.sim.allocation import AllocationConfig, AllocationResult, AllocationSnapshot

    configs: list[AllocationConfig] = []

    def runner(config: AllocationConfig) -> AllocationResult:
        configs.append(config)
        snap = AllocationSnapshot(session=date(2016, 1, 4), cash_krw=1.0, cash_usd=0.0, shares={}, mark_krw=1.0, contribution_krw=config.monthly_contribution_krw, fees_krw=0.0)
        return AllocationResult(config=config, snapshots=(snap,), terminal_wealth_krw=2.0, xirr=0.0, max_drawdown=0.0, terminal_wealth_real_krw=2.0, xirr_real=0.0, total_contribution_real_krw=1.0)

    report = compare_compound_dca(runner=runner, contribution_krw=1_000_000.0, start=date(2016, 1, 4), end=date(2016, 2, 1))
    assert COMPOUND_DCA_ARM_IDS == (
        'qqq_flat', 'qqq_adaptive_v5', 'qqq90_soxx10_flat', 'qqq90_soxx10_adaptive_v5',
        'qqq95_soxx5_adaptive_v5', 'qqq85_soxx15_adaptive_v5',
        'soxx90_qqq10_flat', 'soxx90_qqq10_adaptive_v5', 'soxx100_flat', 'soxx100_adaptive_v5',
        'qqq_soxx_riskbudget_flat', 'qqq_soxx_riskbudget_adaptive_v5',
    )
    assert tuple(row.arm_id for row in report.rows) == COMPOUND_DCA_ARM_IDS
    for cfg, arm in zip(configs, COMPOUND_DCA_ARM_IDS, strict=True):
        is_ad = arm.endswith('adaptive_v5')
        assert (cfg.adaptive_contribution is OPERATIONAL_ADAPTIVE_CONTRIBUTION) is is_ad
        assert cfg.rebalance_band is None
        if arm.startswith('qqq90_soxx10'):
            assert cfg.targets_override == COMPOUND_DCA_MIX_TARGETS == {'QQQ': 0.9, 'SOXX': 0.1}
            assert cfg.mix_risk_budget is None
        elif arm.startswith('qqq95_soxx5'):
            assert cfg.targets_override == {'QQQ': 0.95, 'SOXX': 0.05}
            assert cfg.mix_risk_budget is None
        elif arm.startswith('qqq85_soxx15'):
            assert cfg.targets_override == {'QQQ': 0.85, 'SOXX': 0.15}
            assert cfg.mix_risk_budget is None
        elif arm.startswith('soxx90_qqq10'):
            assert cfg.targets_override == COMPOUND_DCA_ROLE_SWAP_TARGETS == {'SOXX': 0.9, 'QQQ': 0.1}
            assert cfg.mix_risk_budget is None
        elif arm.startswith('soxx100'):
            assert cfg.targets_override == COMPOUND_DCA_SOXX100_TARGETS == {'SOXX': 1.0}
            assert cfg.mix_risk_budget is None
        elif arm.startswith('qqq_soxx_riskbudget'):
            assert cfg.targets_override is None
            assert cfg.mix_risk_budget == OPERATIONAL_MIX_RISK_BUDGET
        else:
            assert cfg.targets_override is None
            assert cfg.mix_risk_budget is None


def test_ccd_adaptive_lock_on_adaptive_arms_only() -> None:
    from datetime import date
    from src.analytics.compound_dca import compare_compound_dca
    from src.policy.adaptive_contribution import OPERATIONAL_ADAPTIVE_CONTRIBUTION
    from src.sim.allocation import AllocationConfig, AllocationResult, AllocationSnapshot

    seen: list[AllocationConfig] = []

    def runner(config: AllocationConfig) -> AllocationResult:
        seen.append(config)
        snap = AllocationSnapshot(session=date(2016, 1, 4), cash_krw=1.0, cash_usd=0.0, shares={}, mark_krw=1.0, contribution_krw=1.0, fees_krw=0.0)
        return AllocationResult(config=config, snapshots=(snap,), terminal_wealth_krw=2.0, xirr=0.0, max_drawdown=0.0, terminal_wealth_real_krw=2.0, xirr_real=0.0, total_contribution_real_krw=1.0)

    compare_compound_dca(runner=runner, contribution_krw=1_000_000.0, start=date(2016, 1, 4), end=date(2016, 2, 1))
    by_mix_ad = {(c.targets_override is not None, c.adaptive_contribution is not None): c for c in seen}
    assert by_mix_ad[(False, False)].adaptive_contribution is None
    assert by_mix_ad[(True, False)].adaptive_contribution is None
    assert by_mix_ad[(False, True)].adaptive_contribution == OPERATIONAL_ADAPTIVE_CONTRIBUTION
    assert by_mix_ad[(True, True)].adaptive_contribution == OPERATIONAL_ADAPTIVE_CONTRIBUTION
    assert all(c.rebalance_band is None for c in seen)


def test_ccd_i5_credits_match_within_sizing_family() -> None:
    from datetime import date
    import pytest
    from src.analytics.compound_dca import compare_compound_dca
    from src.sim.allocation import AllocationConfig, AllocationResult, AllocationSnapshot

    def _result(config: AllocationConfig, credit: float) -> AllocationResult:
        snap = AllocationSnapshot(session=date(2016, 1, 4), cash_krw=credit, cash_usd=0.0, shares={}, mark_krw=credit, contribution_krw=credit, fees_krw=0.0)
        return AllocationResult(config=config, snapshots=(snap,), terminal_wealth_krw=credit * 2, xirr=0.0, max_drawdown=0.0, terminal_wealth_real_krw=credit * 2, xirr_real=0.0, total_contribution_real_krw=credit)

    def aligned(config: AllocationConfig) -> AllocationResult:
        credit = 100.0 if config.adaptive_contribution is not None else 50.0
        return _result(config, credit)

    report = compare_compound_dca(runner=aligned, contribution_krw=1_000_000.0, start=date(2016, 1, 4), end=date(2016, 2, 1))
    assert len(report.rows) == 12

    def broken(config: AllocationConfig) -> AllocationResult:
        credit = 100.0 if config.adaptive_contribution is not None else 50.0
        if config.targets_override is not None and config.adaptive_contribution is not None:
            credit = 99.0
        return _result(config, credit)

    with pytest.raises(ValueError, match=r'identical|I5|family'):
        compare_compound_dca(runner=broken, contribution_krw=1_000_000.0, start=date(2016, 1, 4), end=date(2016, 2, 1))


def test_ccd_champion_max_real_gain_unlock_false() -> None:
    from datetime import date
    from src.analytics.compound_dca import compare_compound_dca
    from src.sim.allocation import AllocationConfig, AllocationResult, AllocationSnapshot

    def runner(config: AllocationConfig) -> AllocationResult:
        ad = config.adaptive_contribution is not None
        ov = dict(config.targets_override) if config.targets_override is not None else None
        rb = config.mix_risk_budget is not None
        if ov == {'QQQ': 0.9, 'SOXX': 0.1} and ad:
            tw, contrib = 240.0, 100.0
        elif ov == {'SOXX': 0.9, 'QQQ': 0.1}:
            tw, contrib = (200.0, 100.0) if ad else (160.0, 80.0)
        elif ov == {'SOXX': 1.0}:
            tw, contrib = (190.0, 100.0) if ad else (150.0, 80.0)
        elif rb:
            tw, contrib = (210.0, 100.0) if ad else (170.0, 80.0)
        elif ad:
            tw, contrib = 150.0, 100.0
        else:
            tw, contrib = 200.0, 80.0
        snap = AllocationSnapshot(session=date(2016, 1, 4), cash_krw=contrib, cash_usd=0.0, shares={}, mark_krw=tw, contribution_krw=contrib, fees_krw=0.0)
        return AllocationResult(config=config, snapshots=(snap,), terminal_wealth_krw=tw, xirr=0.0, max_drawdown=0.0, terminal_wealth_real_krw=tw, xirr_real=0.0, total_contribution_real_krw=contrib)

    report = compare_compound_dca(runner=runner, contribution_krw=1_000_000.0, start=date(2016, 1, 4), end=date(2016, 2, 1))
    assert report.champion_arm_id == 'qqq90_soxx10_adaptive_v5'
    assert report.operational_unlock is False


def test_ccd_reject_nonpositive_contribution() -> None:
    import math
    import pytest
    from src.analytics.compound_dca import compare_compound_dca

    def boom(config: object) -> object:
        raise AssertionError('runner must not run')

    for bad in (0.0, -1.0, math.nan):
        with pytest.raises(ValueError, match='contribution_krw'):
            compare_compound_dca(runner=boom, contribution_krw=bad)


def test_ccd_diverging_snapshots_fail_closed() -> None:
    from datetime import date, timedelta
    import pytest
    from src.analytics.compound_dca import compare_compound_dca
    from src.sim.allocation import AllocationConfig, AllocationResult, AllocationSnapshot

    def runner(config: AllocationConfig) -> AllocationResult:
        n = 2 if config.targets_override is not None else 1
        snaps = tuple(AllocationSnapshot(session=date(2016, 1, 4) + timedelta(days=i), cash_krw=1.0, cash_usd=0.0, shares={}, mark_krw=1.0, contribution_krw=1.0, fees_krw=0.0) for i in range(n))
        return AllocationResult(config=config, snapshots=snaps, terminal_wealth_krw=float(n), xirr=0.0, max_drawdown=0.0, terminal_wealth_real_krw=float(n), xirr_real=0.0, total_contribution_real_krw=float(n))

    with pytest.raises(ValueError, match='snapshot'):
        compare_compound_dca(runner=runner, contribution_krw=1_000_000.0, start=date(2016, 1, 4), end=date(2016, 2, 1))


def test_ccd_never_imports_adoption_passes() -> None:
    from pathlib import Path
    text = Path('src/analytics/compound_dca.py').read_text(encoding='utf-8')
    assert 'adoption_passes' not in text


def test_qqq_soxx_intensity_targets_preregistered_only() -> None:
    import pytest
    from src.analytics.compound_dca import qqq_soxx_intensity_targets
    from src.analytics.thesis.incremental import INCREMENTAL_SATELLITE_WEIGHTS

    assert INCREMENTAL_SATELLITE_WEIGHTS == (0.05, 0.10, 0.15)
    for w in INCREMENTAL_SATELLITE_WEIGHTS:
        targets = qqq_soxx_intensity_targets(w)
        assert targets == {'QQQ': pytest.approx(1.0 - w), 'SOXX': pytest.approx(w)}
        assert abs(sum(targets.values()) - 1.0) < 1e-12
    with pytest.raises(ValueError, match=r'INCREMENTAL_SATELLITE_WEIGHTS|satellite'):
        qqq_soxx_intensity_targets(0.12)


def test_select_mdd_feasible_rejects_deeper_drawdown() -> None:
    from datetime import date
    from src.analytics.compound_dca import (
        COMPOUND_MDD_SLACK,
        CompoundDcaArmRow,
        OPERATIONAL_COMPOUND_BASELINE_ARM_ID,
        select_mdd_feasible_champion,
    )
    from src.sim.allocation import AllocationConfig, AllocationResult, AllocationSnapshot
    from src.policy.targets import PolicyId

    def _row(arm_id: str, real_tw: float, contrib: float, mdd: float) -> CompoundDcaArmRow:
        cfg = AllocationConfig(policy=PolicyId.QQQ, start=date(2016, 7, 1), end=date(2026, 6, 30), monthly_contribution_krw=1_000_000.0)
        snap = AllocationSnapshot(session=date(2016, 7, 1), cash_krw=contrib, cash_usd=0.0, shares={}, mark_krw=real_tw, contribution_krw=contrib, fees_krw=0.0)
        res = AllocationResult(config=cfg, snapshots=(snap,), terminal_wealth_krw=real_tw, xirr=0.0, max_drawdown=mdd, terminal_wealth_real_krw=real_tw, xirr_real=0.0, total_contribution_real_krw=contrib)
        return CompoundDcaArmRow(arm_id=arm_id, config=cfg, result=res, terminal_wealth_krw=real_tw, terminal_wealth_real_krw=real_tw, total_contribution_real_krw=contrib, real_gain=real_tw - contrib, xirr=0.0, xirr_real=0.0, max_drawdown=mdd)

    rows = (
        _row('soxx100_adaptive_v5', 1000.0, 100.0, -0.294),
        _row(OPERATIONAL_COMPOUND_BASELINE_ARM_ID, 485.0, 129.0, -0.207),
        _row('qqq_adaptive_v5', 445.0, 129.0, -0.201),
    )
    assert COMPOUND_MDD_SLACK == 0.02
    assert select_mdd_feasible_champion(rows) == OPERATIONAL_COMPOUND_BASELINE_ARM_ID


def test_select_mdd_feasible_picks_max_gain_within_slack() -> None:
    from datetime import date
    from src.analytics.compound_dca import (
        CompoundDcaArmRow,
        OPERATIONAL_COMPOUND_BASELINE_ARM_ID,
        select_mdd_feasible_champion,
    )
    from src.sim.allocation import AllocationConfig, AllocationResult, AllocationSnapshot
    from src.policy.targets import PolicyId

    def _row(arm_id: str, real_tw: float, contrib: float, mdd: float) -> CompoundDcaArmRow:
        cfg = AllocationConfig(policy=PolicyId.QQQ, start=date(2016, 7, 1), end=date(2026, 6, 30), monthly_contribution_krw=1_000_000.0)
        snap = AllocationSnapshot(session=date(2016, 7, 1), cash_krw=contrib, cash_usd=0.0, shares={}, mark_krw=real_tw, contribution_krw=contrib, fees_krw=0.0)
        res = AllocationResult(config=cfg, snapshots=(snap,), terminal_wealth_krw=real_tw, xirr=0.0, max_drawdown=mdd, terminal_wealth_real_krw=real_tw, xirr_real=0.0, total_contribution_real_krw=contrib)
        return CompoundDcaArmRow(arm_id=arm_id, config=cfg, result=res, terminal_wealth_krw=real_tw, terminal_wealth_real_krw=real_tw, total_contribution_real_krw=contrib, real_gain=real_tw - contrib, xirr=0.0, xirr_real=0.0, max_drawdown=mdd)

    rows = (
        _row(OPERATIONAL_COMPOUND_BASELINE_ARM_ID, 485.0, 129.0, -0.207),
        _row('qqq85_soxx15_adaptive_v5', 520.0, 129.0, -0.220),
        _row('qqq95_soxx5_adaptive_v5', 470.0, 129.0, -0.205),
    )
    assert select_mdd_feasible_champion(rows) == 'qqq85_soxx15_adaptive_v5'


def test_select_mdd_feasible_fail_closed_inputs() -> None:
    import math
    import pytest
    from datetime import date
    from src.analytics.compound_dca import CompoundDcaArmRow, select_mdd_feasible_champion
    from src.sim.allocation import AllocationConfig, AllocationResult, AllocationSnapshot
    from src.policy.targets import PolicyId

    cfg = AllocationConfig(policy=PolicyId.QQQ, start=date(2016, 7, 1), end=date(2026, 6, 30), monthly_contribution_krw=1.0)
    snap = AllocationSnapshot(session=date(2016, 7, 1), cash_krw=1.0, cash_usd=0.0, shares={}, mark_krw=1.0, contribution_krw=1.0, fees_krw=0.0)
    res = AllocationResult(config=cfg, snapshots=(snap,), terminal_wealth_krw=1.0, xirr=0.0, max_drawdown=-0.1, terminal_wealth_real_krw=1.0, xirr_real=0.0, total_contribution_real_krw=1.0)
    row = CompoundDcaArmRow(arm_id='other', config=cfg, result=res, terminal_wealth_krw=1.0, terminal_wealth_real_krw=1.0, total_contribution_real_krw=1.0, real_gain=0.0, xirr=0.0, xirr_real=0.0, max_drawdown=-0.1)

    with pytest.raises(ValueError, match='rows'):
        select_mdd_feasible_champion(())
    with pytest.raises(ValueError, match='baseline'):
        select_mdd_feasible_champion((row,))
    with pytest.raises(ValueError, match='mdd_slack'):
        select_mdd_feasible_champion((row,), baseline_arm_id='other', mdd_slack=-0.01)
    with pytest.raises(ValueError, match='mdd_slack'):
        select_mdd_feasible_champion((row,), baseline_arm_id='other', mdd_slack=math.nan)


def test_ccd_intensity_arms_and_dual_champions() -> None:
    from datetime import date
    from src.analytics.compound_dca import (
        COMPOUND_DCA_ARM_IDS,
        COMPOUND_MDD_SLACK,
        OPERATIONAL_COMPOUND_BASELINE_ARM_ID,
        compare_compound_dca,
        qqq_soxx_intensity_targets,
    )
    from src.sim.allocation import AllocationConfig, AllocationResult, AllocationSnapshot

    assert 'qqq95_soxx5_adaptive_v5' in COMPOUND_DCA_ARM_IDS
    assert 'qqq85_soxx15_adaptive_v5' in COMPOUND_DCA_ARM_IDS
    assert OPERATIONAL_COMPOUND_BASELINE_ARM_ID in COMPOUND_DCA_ARM_IDS

    def runner(config: AllocationConfig) -> AllocationResult:
        ad = config.adaptive_contribution is not None
        ov = dict(config.targets_override) if config.targets_override is not None else {}
        soxx = float(ov.get('SOXX', 0.0))
        if soxx >= 0.99 and ad:
            tw, contrib, mdd = 1000.0, 100.0, -0.294
        elif ov == qqq_soxx_intensity_targets(0.10) and ad:
            tw, contrib, mdd = 485.0, 129.0, -0.207
        elif ov == qqq_soxx_intensity_targets(0.15) and ad:
            tw, contrib, mdd = 500.0, 129.0, -0.250
        elif ov == qqq_soxx_intensity_targets(0.05) and ad:
            tw, contrib, mdd = 460.0, 129.0, -0.205
        elif ad:
            tw, contrib, mdd = 445.0, 129.0, -0.201
        else:
            tw, contrib, mdd = 400.0, 110.0, -0.210
        snap = AllocationSnapshot(session=date(2016, 1, 4), cash_krw=contrib, cash_usd=0.0, shares={}, mark_krw=tw, contribution_krw=contrib, fees_krw=0.0)
        return AllocationResult(config=config, snapshots=(snap,), terminal_wealth_krw=tw, xirr=0.0, max_drawdown=mdd, terminal_wealth_real_krw=tw, xirr_real=0.0, total_contribution_real_krw=contrib)

    report = compare_compound_dca(runner=runner, contribution_krw=1_000_000.0, start=date(2016, 1, 4), end=date(2016, 2, 1))
    assert report.champion_arm_id == 'soxx100_adaptive_v5'
    assert report.mdd_feasible_champion_arm_id == OPERATIONAL_COMPOUND_BASELINE_ARM_ID
    assert report.mdd_baseline_arm_id == OPERATIONAL_COMPOUND_BASELINE_ARM_ID
    assert report.mdd_slack == COMPOUND_MDD_SLACK
    assert report.operational_unlock is False
    configs_by_arm = {row.arm_id: row.config for row in report.rows}
    assert configs_by_arm['qqq95_soxx5_adaptive_v5'].targets_override == qqq_soxx_intensity_targets(0.05)
    assert configs_by_arm['qqq85_soxx15_adaptive_v5'].targets_override == qqq_soxx_intensity_targets(0.15)


def test_wf_soxx_intensity_mdd_spec_loads() -> None:
    from datetime import date
    from src.validation.experiment import load_experiment_config
    from src.validation.windows import walk_forward_windows
    from src.analytics.compound_dca import COMPOUND_DCA_WINDOW, qqq_soxx_intensity_targets

    spec = load_experiment_config('configs/experiments/wf_qqq_soxx_intensity_mdd.json')
    assert spec.objective == 'adaptive_growth'
    assert spec.adaptive_contribution is not None
    assert spec.baseline.id == 'qqq90_soxx10_adaptive_v5'
    assert spec.baseline.targets == qqq_soxx_intensity_targets(0.10)
    cand_ids = {c.id for c in spec.candidates}
    assert cand_ids == {'qqq95_soxx5_adaptive_v5', 'qqq85_soxx15_adaptive_v5'}
    by_id = {c.id: c for c in spec.candidates}
    assert by_id['qqq95_soxx5_adaptive_v5'].targets == qqq_soxx_intensity_targets(0.05)
    assert by_id['qqq85_soxx15_adaptive_v5'].targets == qqq_soxx_intensity_targets(0.15)
    assert spec.preregistration is not None
    assert spec.preregistration.weights_locked is True
    assert spec.preregistration.universe_locked is True
    assert (spec.start, spec.end) == COMPOUND_DCA_WINDOW
    assert spec.train_months == 36
    assert spec.test_months == 24
    folds = walk_forward_windows(spec.start, spec.end, train_months=int(spec.train_months), test_months=int(spec.test_months))
    assert len(folds) >= 2
    assert spec.start == date(2016, 7, 1)
