# ruff: noqa: F401, PT018, S101
"""Unit tests for compound DCA."""
from __future__ import annotations


def test_ccd_preregistered_arms() -> None:
    from datetime import date, timedelta
    from src.analytics.compound_dca import COMPOUND_DCA_ARM_IDS, COMPOUND_DCA_MIX_TARGETS, compare_compound_dca
    from src.sim.allocation import AllocationConfig, AllocationResult, AllocationSnapshot

    configs: list[AllocationConfig] = []

    def runner(config: AllocationConfig) -> AllocationResult:
        configs.append(config)
        snap = AllocationSnapshot(session=date(2016, 1, 4), cash_krw=1.0, cash_usd=0.0, shares={}, mark_krw=1.0, contribution_krw=config.monthly_contribution_krw, fees_krw=0.0)
        return AllocationResult(config=config, snapshots=(snap,), terminal_wealth_krw=2.0, xirr=0.0, max_drawdown=0.0, terminal_wealth_real_krw=2.0, xirr_real=0.0, total_contribution_real_krw=1.0)

    report = compare_compound_dca(runner=runner, contribution_krw=1_000_000.0, start=date(2016, 1, 4), end=date(2016, 2, 1))
    assert tuple(row.arm_id for row in report.rows) == COMPOUND_DCA_ARM_IDS
    assert len(configs) == 4
    mix = [c for c in configs if c.targets_override is not None]
    core = [c for c in configs if c.targets_override is None]
    assert len(mix) == 2 and len(core) == 2
    for cfg in mix:
        assert cfg.targets_override == COMPOUND_DCA_MIX_TARGETS


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
    assert len(report.rows) == 4

    def broken(config: AllocationConfig) -> AllocationResult:
        credit = 100.0 if config.adaptive_contribution is not None else 50.0
        if config.targets_override is not None and config.adaptive_contribution is not None:
            credit = 99.0
        return _result(config, credit)

    with pytest.raises(ValueError, match=r'identical|I5'):
        compare_compound_dca(runner=broken, contribution_krw=1_000_000.0, start=date(2016, 1, 4), end=date(2016, 2, 1))


def test_ccd_champion_max_real_gain_unlock_false() -> None:
    from datetime import date
    import pytest
    from src.analytics.compound_dca import compare_compound_dca
    from src.sim.allocation import AllocationConfig, AllocationResult, AllocationSnapshot

    def runner(config: AllocationConfig) -> AllocationResult:
        mix = config.targets_override is not None
        ad = config.adaptive_contribution is not None
        tw, contrib = {(False, False): (200.0, 80.0), (False, True): (150.0, 100.0), (True, False): (180.0, 80.0), (True, True): (240.0, 100.0)}[(mix, ad)]
        snap = AllocationSnapshot(session=date(2016, 1, 4), cash_krw=contrib, cash_usd=0.0, shares={}, mark_krw=tw, contribution_krw=contrib, fees_krw=0.0)
        return AllocationResult(config=config, snapshots=(snap,), terminal_wealth_krw=tw, xirr=0.0, max_drawdown=0.0, terminal_wealth_real_krw=tw, xirr_real=0.0, total_contribution_real_krw=contrib)

    report = compare_compound_dca(runner=runner, contribution_krw=1_000_000.0, start=date(2016, 1, 4), end=date(2016, 2, 1))
    assert report.champion_arm_id == 'qqq90_soxx10_adaptive_v5'
    assert report.operational_unlock is False
    gains = {row.arm_id: row.real_gain for row in report.rows}
    assert gains['qqq_adaptive_v5'] == pytest.approx(50.0)
    assert gains['qqq90_soxx10_adaptive_v5'] == pytest.approx(140.0)


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
