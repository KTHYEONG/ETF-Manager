"""Regime proxy tests (Wave 7)."""

from __future__ import annotations

from datetime import date

import pytest

from src.analytics.regime_proxy import compute_regime_proxy_slot
from src.policy.thesis import ThesisId, ThesisSpec, Horizon
from src.sim.allocation import AllocationConfig, AllocationResult, AllocationSnapshot


@pytest.mark.parametrize("scenario_id", ["REG-A-proxy-windows"])
def test_reg_a_proxy_windows(scenario_id: str) -> None:
    """REG-A-proxy-windows"""
    thesis = ThesisSpec(
        id=ThesisId.AI_COMPUTE,
        version=1,
        title="test",
        status="research",
        horizon=Horizon(min_years=5, target_years=10),
        causal_chain=["a"],
        falsifiers=["f1"],
        candidate_sleeves=["ai_semiconductor"],
        historical_proxies=["SOXX"],
    )

    # Mock runner: proxy TW 1.1x QQQ on 2/3 windows, lose on 1
    # Windows are pre_ai, bear_2022, recent_2023_2026 in that order
    # We'll track calls
    call_index = {"count": 0}

    def runner(config: AllocationConfig) -> AllocationResult:
        # config.targets_override determines proxy vs QQQ
        is_proxy = config.targets_override is not None and "SOXX" in config.targets_override
        # Determine window by start date
        start = config.start
        # Assign win/lose per window
        # Let pre_ai and recent win (1.1x), bear lose (0.9x)
        if start == date(2010, 1, 4):  # pre_ai
            factor = 1.1 if is_proxy else 1.0
        elif start == date(2022, 1, 3):  # bear_2022
            factor = 0.9 if is_proxy else 1.0
        elif start == date(2023, 1, 3):  # recent
            factor = 1.1 if is_proxy else 1.0
        else:
            factor = 1.0
        wealth = 100.0 * factor
        return AllocationResult(
            config=config,
            snapshots=(AllocationSnapshot(session=date(2024, 1, 31), cash_krw=0, cash_usd=0, shares={}, mark_krw=wealth, contribution_krw=0, fees_krw=0),),
            terminal_wealth_krw=wealth,
            xirr=0.0,
            max_drawdown=0.0,
            terminal_wealth_real_krw=wealth,
            xirr_real=0.0,
        )

    slot = compute_regime_proxy_slot(thesis=thesis, runner=runner, contribution_krw=1_000_000)
    assert slot.status == "computed"
    assert slot.metrics["windows_beat_qqq"] == 2
    assert slot.metrics["windows_tested"] == 3
