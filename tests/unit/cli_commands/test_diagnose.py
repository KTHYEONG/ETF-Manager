"""Unit tests for diagnose CLI command runners."""

from __future__ import annotations

import logging
from datetime import date

import pytest

from src.cli_commands.diagnose import run_diagnose_compound_dca_command
from src.data.settings import DataSettings


def test_cli_compound_dca_logs_mdd_feasible_champion(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from src.analytics.compound_dca import (
        COMPOUND_MDD_SLACK,
        OPERATIONAL_COMPOUND_BASELINE_ARM_ID,
        CompoundDcaArmRow,
        CompoundDcaReport,
    )
    from src.sim.allocation import AllocationConfig, AllocationResult, AllocationSnapshot
    from src.policy.targets import PolicyId

    cfg = AllocationConfig(
        policy=PolicyId.QQQ,
        start=date(2016, 7, 1),
        end=date(2026, 6, 30),
        monthly_contribution_krw=1_000_000.0,
    )
    snap = AllocationSnapshot(
        session=date(2016, 7, 1),
        cash_krw=1.0,
        cash_usd=0.0,
        shares={},
        mark_krw=1.0,
        contribution_krw=1.0,
        fees_krw=0.0,
    )
    res = AllocationResult(
        config=cfg,
        snapshots=(snap,),
        terminal_wealth_krw=2.0,
        xirr=0.0,
        max_drawdown=-0.2,
        terminal_wealth_real_krw=2.0,
        xirr_real=0.0,
        total_contribution_real_krw=1.0,
    )
    row = CompoundDcaArmRow(
        arm_id=OPERATIONAL_COMPOUND_BASELINE_ARM_ID,
        config=cfg,
        result=res,
        terminal_wealth_krw=2.0,
        terminal_wealth_real_krw=2.0,
        total_contribution_real_krw=1.0,
        real_gain=1.0,
        xirr=0.0,
        xirr_real=0.0,
        max_drawdown=-0.2,
    )
    report = CompoundDcaReport(
        rows=(row,),
        champion_arm_id="soxx100_adaptive_v5",
        operational_unlock=False,
        start=date(2016, 7, 1),
        end=date(2026, 6, 30),
        contribution_krw=1_000_000.0,
        mdd_feasible_champion_arm_id=OPERATIONAL_COMPOUND_BASELINE_ARM_ID,
        mdd_baseline_arm_id=OPERATIONAL_COMPOUND_BASELINE_ARM_ID,
        mdd_slack=COMPOUND_MDD_SLACK,
    )

    def fake_compare(**kwargs: object) -> CompoundDcaReport:
        return report

    from src.analytics import compound_dca as compound_dca_mod

    monkeypatch.setattr(compound_dca_mod, "compare_compound_dca", fake_compare)

    caplog.set_level(logging.INFO, logger="src.cli_commands.diagnose")
    assert run_diagnose_compound_dca_command(contribution_krw=1_000_000.0, settings=DataSettings()) == 0
    done_logs = [r.message for r in caplog.records if "compound_dca_done" in r.message]
    assert len(done_logs) == 1
    assert "mdd_feasible_champion=" in done_logs[0]
    assert OPERATIONAL_COMPOUND_BASELINE_ARM_ID in done_logs[0]
    assert "operational_unlock=false" in done_logs[0]
