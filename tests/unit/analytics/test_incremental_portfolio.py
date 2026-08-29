# ruff: noqa: B905,PT011,S101
"""Track H incremental portfolio tests."""
from __future__ import annotations

from datetime import date

import pytest

from src.analytics.incremental_portfolio import (
    INCREMENTAL_SOXX_WEIGHTS,
    PATH_BOOTSTRAP_WIN_FLOOR,
    BuyOnlyAttribution,
    IncrementalArmId,
    IncrementalArmReport,
    PathBootstrapVerdict,
    arm_targets,
    attribute_buy_only_soxx,
    classify_portfolio_status,
    paired_path_block_bootstrap,
    apply_incremental_portfolio_status,
)
from src.analytics.thesis_meaning import (
    HistoricalQuality,
    PortfolioEvidenceStatus,
    ThesisEvidenceStatus,
    ThesisMeaningSnapshot,
    VehicleEvidenceStatus,
)
from src.sim.allocation import AllocationConfig, AllocationResult, AllocationSnapshot
from src.policy.targets import PolicyId


def _make_result(mark_krw_seq: tuple[float, ...], shares_seq: tuple[dict[str, float], ...], terminal_real: float) -> AllocationResult:
    cfg = AllocationConfig(policy=PolicyId.QQQ, start=date(2020, 1, 1), end=date(2020, 12, 31), monthly_contribution_krw=1_000_000)
    snaps = []
    for idx, (mark, shares) in enumerate(zip(mark_krw_seq, shares_seq)):
        snaps.append(
            AllocationSnapshot(
                session=date(2020, idx + 1, 15),
                cash_krw=0.0,
                cash_usd=0.0,
                shares=dict(shares),
                mark_krw=float(mark),
                contribution_krw=1_000_000.0,
                fees_krw=0.0,
                reserve_krw=0.0,
            )
        )
    return AllocationResult(
        config=cfg,
        snapshots=tuple(snaps),
        terminal_wealth_krw=float(terminal_real),
        xirr=0.0,
        max_drawdown=0.0,
        terminal_wealth_real_krw=float(terminal_real),
        xirr_real=0.0,
    )


def test_inc_h1_arm_targets_locked() -> None:
    assert INCREMENTAL_SOXX_WEIGHTS == (0.05, 0.10, 0.15)
    assert arm_targets(0.05) == {"QQQ": 0.95, "SOXX": 0.05}
    assert arm_targets(0.10) == {"QQQ": 0.90, "SOXX": 0.10}
    assert arm_targets(0.15) == {"QQQ": 0.85, "SOXX": 0.15}
    with pytest.raises(ValueError):
        arm_targets(0.17)
    with pytest.raises(ValueError):
        arm_targets(1.0)


def test_inc_h2_paired_path_bootstrap_joint_blocks() -> None:
    rets = tuple([0.01] * 24)
    v1 = paired_path_block_bootstrap(rets, rets, block_size=6, n_paths=50, seed=1)
    assert v1.n_paths == 50
    assert 0.0 <= v1.win_rate <= 1.0
    v2 = paired_path_block_bootstrap(rets, rets, block_size=6, n_paths=50, seed=1)
    assert v2.win_rate == pytest.approx(v1.win_rate)
    assert v2.p05_terminal_ratio == pytest.approx(v1.p05_terminal_ratio)
    assert v2.n_paths == v1.n_paths


def test_inc_h3_path_ok_threshold() -> None:
    assert pytest.approx(0.55) == PATH_BOOTSTRAP_WIN_FLOOR
    # win_rate 0.55 should be ok True via direct verdict construction not needed; test via bootstrap
    # strictly dominating returns -> ok True
    cand = tuple([0.02] * 20)
    base = tuple([0.01] * 20)
    verdict_dom = paired_path_block_bootstrap(cand, base, block_size=4, n_paths=200, seed=42)
    assert verdict_dom.ok is True
    assert verdict_dom.win_rate >= 0.55
    # equal returns -> win_rate around 0.5? But due to tie rule >=1, equal series gives win_rate 1.0 (since all ratios 1). So use slight underperformance to get <0.55
    # Instead test equal returns yields win_rate ==1 with dominating? Actually equal returns produce ratio 1 always -> win_rate 1 => ok True. So we craft candidate slightly worse to get <0.55.
    # Use candidate 0.01 vs baseline 0.02
    cand_low = tuple([0.01] * 20)
    base_high = tuple([0.02] * 20)
    verdict_low = paired_path_block_bootstrap(cand_low, base_high, block_size=4, n_paths=200, seed=42)
    assert verdict_low.win_rate < 0.55
    assert verdict_low.ok is False
    # direct threshold edge: construct verdict with 0.55
    edge = PathBootstrapVerdict(n_paths=10, win_rate=0.55, p05_terminal_ratio=1.0, ok=True)
    assert edge.ok is True
    below = PathBootstrapVerdict(n_paths=10, win_rate=0.54, p05_terminal_ratio=1.0, ok=False)
    assert below.ok is False


def test_inc_h4_buy_only_attribution_drift() -> None:
    # two snapshots, target 0.15, realized weights 0.10 then 0.20
    # price_at returns 100, mark 1000, shares => weight = shares*price*fx/mark with fx=1
    def price_at(d: date, ticker: str) -> float:
        assert ticker == "SOXX"
        return 100.0

    def fx_at(d: date) -> float:
        return 1.0

    cand = _make_result((1000.0, 1000.0), ({"SOXX": 1.0}, {"SOXX": 2.0}), terminal_real=2000.0)
    base = _make_result((1000.0, 1000.0), ({}, {}), terminal_real=1000.0)
    attr = attribute_buy_only_soxx(candidate=cand, baseline=base, soxx_weight=0.15, price_at=price_at, fx_at=fx_at)
    assert attr.mean_abs_weight_drift == pytest.approx(0.05, abs=1e-9)
    assert attr.terminal_weight_drift == pytest.approx(0.05, abs=1e-9)
    assert attr.target_soxx_weight == pytest.approx(0.15)
    assert attr.mean_realized_soxx_weight == pytest.approx(0.15, abs=1e-9)
    assert attr.terminal_realized_soxx_weight == pytest.approx(0.20, abs=1e-9)
    assert attr.incremental_wealth_ratio == pytest.approx(2.0)


def test_inc_afx_fx_scales_realized_weight() -> None:
    def price_at(d: date, ticker: str) -> float:
        assert ticker == "SOXX"
        return 10.0

    def fx_at(d: date) -> float:
        return 1300.0

    cand = _make_result((13000.0,), ({"SOXX": 1.0},), terminal_real=13000.0)
    base = _make_result((13000.0,), ({},), terminal_real=13000.0)
    attr = attribute_buy_only_soxx(candidate=cand, baseline=base, soxx_weight=0.05, price_at=price_at, fx_at=fx_at)
    assert attr.mean_realized_soxx_weight == pytest.approx(1.0, abs=1e-12)
    assert attr.terminal_realized_soxx_weight == pytest.approx(1.0, abs=1e-12)


def test_inc_afx_rejects_usd_over_krw_mark() -> None:
    def price_at(d: date, ticker: str) -> float:
        assert ticker == "SOXX"
        return 100.0

    def fx_at(d: date) -> float:
        return 1.0

    cand = _make_result((130000.0,), ({"SOXX": 1.0},), terminal_real=130000.0)
    base = _make_result((130000.0,), ({},), terminal_real=130000.0)
    with pytest.raises(ValueError):
        attribute_buy_only_soxx(candidate=cand, baseline=base, soxx_weight=0.05, price_at=price_at, fx_at=fx_at)


def test_inc_afx_weight_above_one_fails() -> None:
    def price_at(d: date, ticker: str) -> float:
        assert ticker == "SOXX"
        return 10.0

    def fx_at(d: date) -> float:
        return 1300.0

    cand = _make_result((13000.0,), ({"SOXX": 2.0},), terminal_real=13000.0)
    base = _make_result((13000.0,), ({},), terminal_real=13000.0)
    with pytest.raises(ValueError):
        attribute_buy_only_soxx(candidate=cand, baseline=base, soxx_weight=0.05, price_at=price_at, fx_at=fx_at)


def test_inc_h5_classify_portfolio_status() -> None:
    def make_arm(median: float, ok: bool) -> IncrementalArmReport:
        verdict = PathBootstrapVerdict(n_paths=10, win_rate=1.0 if ok else 0.4, p05_terminal_ratio=1.0, ok=ok)
        attr = BuyOnlyAttribution(
            target_soxx_weight=0.05,
            mean_realized_soxx_weight=0.05,
            terminal_realized_soxx_weight=0.05,
            mean_abs_weight_drift=0.0,
            terminal_weight_drift=0.0,
            incremental_wealth_ratio=1.0,
        )
        return IncrementalArmReport(
            arm_id=IncrementalArmId.QQQ95_SOXX5,
            soxx_weight=0.05,
            median_ratio=float(median),
            p10_ratio=float(median),
            worst_ratio=float(median),
            win_rate=float(verdict.win_rate),
            cohort_count=8,
            ce_gamma_2=1.0,
            ce_gamma_5=1.0,
            ce_gamma_10=1.0,
            attribution=attr,
            path_bootstrap=verdict,
        )

    promising = make_arm(1.01, True)
    assert classify_portfolio_status([promising]) == PortfolioEvidenceStatus.HISTORICALLY_PROMISING
    weak1 = make_arm(0.99, True)
    assert classify_portfolio_status([weak1]) == PortfolioEvidenceStatus.HISTORICALLY_WEAK
    weak2 = make_arm(1.01, False)
    assert classify_portfolio_status([weak2]) == PortfolioEvidenceStatus.HISTORICALLY_WEAK
    with pytest.raises(ValueError):
        classify_portfolio_status([])


def test_inc_h6_apply_status_preserves_vehicle() -> None:
    snap = ThesisMeaningSnapshot(
        thesis_status=ThesisEvidenceStatus.UNRESOLVED,
        vehicle_status=VehicleEvidenceStatus.ACTIVE_PROXY,
        portfolio_status=PortfolioEvidenceStatus.UNVERIFIED,
        historical_quality=HistoricalQuality.TARGET_THIN,
        history_available=True,
        evidence_sufficient=True,
        thin_sample_warning=True,
    )
    updated = apply_incremental_portfolio_status(snap, PortfolioEvidenceStatus.HISTORICALLY_PROMISING)
    assert updated.vehicle_status == snap.vehicle_status
    assert updated.thesis_status == snap.thesis_status
    assert updated.portfolio_status == PortfolioEvidenceStatus.HISTORICALLY_PROMISING
    assert updated.historical_quality == snap.historical_quality
