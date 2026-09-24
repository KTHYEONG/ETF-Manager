"""Invariant guard tests for the after-tax cohort campaign."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from src.data.settings import DataSettings
from src.sim.after_tax_engine import ExecutionMode
from src.validation.after_tax_campaign import (
    AfterTaxArmSummary,
    AfterTaxCampaignSpec,
    AfterTaxGateSpec,
    ArmRole,
    after_tax_gate_passes,
    load_after_tax_campaign_spec,
    run_after_tax_campaign,
    write_after_tax_campaign_report,
)


def _rule_static() -> dict:
    return {"rule_id": "static", "core_targets": {"QQQ": 0.9, "SOXX": 0.1}}


def _make_result(wealth: float, *, mdd: float = -0.05, taxes: float = 10.0, sells: int = 2):
    from src.sim.after_tax_engine import AfterTaxResult

    return AfterTaxResult(
        config=None,  # type: ignore[arg-type]
        snapshots=(),
        disposals=(),
        terminal_after_tax_krw=wealth,
        terminal_after_tax_real_krw=wealth,
        terminal_pre_liquidation_krw=wealth,
        liquidation_tax_krw=0.0,
        taxes_paid_krw=taxes,
        xirr_after_tax_real=0.0,
        total_contribution_real_krw=1.0,
        max_drawdown_after_tax=mdd,
        annual_financial_income_krw={},
        financial_income_breach_years=(),
        sell_count=sells,
    )


def _spec(
    *,
    start: date = date(2000, 4, 1),
    end: date = date(2022, 4, 30),
    horizons: tuple[int, ...] = (120,),
    primary: int = 120,
    crash_regimes: tuple[str, ...] = ("dot_com", "gfc"),
) -> AfterTaxCampaignSpec:
    from src.policy.after_tax_rules import parse_after_tax_rule_spec

    base_rule = parse_after_tax_rule_spec(_rule_static())
    cand_rule = parse_after_tax_rule_spec(_rule_static())
    from src.validation.after_tax_campaign import AfterTaxArmSpec

    return AfterTaxCampaignSpec(
        name="test_campaign",
        start=start,
        end=end,
        contribution_krw=1_000_000.0,
        commission_bps=10.0,
        fx_spread_bps=20.0,
        tax_regime_path="configs/tax/kr_overseas_equity.json",
        horizons_months=horizons,
        primary_horizon_months=primary,
        step_months=12,
        crash_regimes=crash_regimes,
        crash_window_months=36,
        fractional_shares=False,
        bootstrap_paths=50,
        baseline_arm_id="b1",
        arms=(
            AfterTaxArmSpec(
                arm_id="b1",
                role=ArmRole.BASELINE,
                rule=base_rule,
                mode=ExecutionMode.BUY_ONLY,
                rebalance_band=None,
                harvest_gains=False,
            ),
            AfterTaxArmSpec(
                arm_id="c1",
                role=ArmRole.OPERATIONAL_CANDIDATE,
                rule=cand_rule,
                mode=ExecutionMode.BUY_ONLY,
                rebalance_band=None,
                harvest_gains=True,
            ),
        ),
        gate=AfterTaxGateSpec(median_ratio_floor=1.0, worst_ratio_floor=0.97, bootstrap_p05_floor=1.0),
    )


def test_baseline_ratio_is_one() -> None:
    """Baseline rows carry ratio exactly 1.0."""

    def _runner(config):
        return _make_result(100.0)

    report = run_after_tax_campaign(_spec(), _runner, seed=7)
    baseline_rows = [row for row in report.rows if row.arm_id == "b1"]
    assert baseline_rows
    assert all(row.ratio == 1.0 for row in baseline_rows)
    assert all(row.frictionless_ratio == 1.0 for row in baseline_rows)


def test_frictionless_rerun_uses_zero_frictions() -> None:
    """Each cohort issues one frictionless call with zero tax and costs."""
    seen: list = []

    def _runner(config):
        seen.append(config)
        return _make_result(100.0)

    run_after_tax_campaign(_spec(), _runner, seed=7)
    free = [cfg for cfg in seen if not cfg.tax_enabled]
    assert free
    assert all(cfg.commission_bps == 0 for cfg in free)
    assert all(cfg.fx_spread_bps == 0 for cfg in free)
    frict = [cfg for cfg in seen if cfg.tax_enabled]
    assert frict
    assert all(cfg.commission_bps == 10.0 for cfg in frict)


def test_crash_first_flag() -> None:
    """Cohorts overlapping crash windows in their first months are flagged."""

    def _runner(config):
        return _make_result(100.0)

    report = run_after_tax_campaign(_spec(), _runner, seed=7)
    by_start = {(row.arm_id, row.cohort_start): row for row in report.rows}
    assert by_start[("b1", date(2000, 4, 1))].crash_first is True
    assert by_start[("b1", date(2012, 4, 1))].crash_first is False


def test_watch_arms_never_pass_the_gate() -> None:
    """A watch summary above every floor still fails the gate."""
    summary = AfterTaxArmSummary(
        arm_id="w",
        role=ArmRole.PROSPECTIVE_WATCH,
        horizon_months=120,
        cohort_count=4,
        median_ratio=2.0,
        worst_ratio=2.0,
        best_ratio=3.0,
        win_rate=1.0,
        frictionless_median_ratio=2.0,
        bootstrap_p05_ratio=2.0,
        crash_first_median_ratio=None,
        other_median_ratio=None,
        median_max_drawdown_after_tax=-0.05,
        median_taxes_paid_krw=0.0,
        median_sell_count=0.0,
        gate_passes=False,
    )
    gate = AfterTaxGateSpec(median_ratio_floor=1.0, worst_ratio_floor=0.97, bootstrap_p05_floor=1.0)
    assert after_tax_gate_passes(summary, gate) is False


def test_gate_boundaries() -> None:
    """Median at the floor is strict while worst at the floor is allowed."""
    gate = AfterTaxGateSpec(median_ratio_floor=1.0, worst_ratio_floor=0.97, bootstrap_p05_floor=1.0)

    def _summary(*, median: float, worst: float, p05: float) -> AfterTaxArmSummary:
        return AfterTaxArmSummary(
            arm_id="c1",
            role=ArmRole.OPERATIONAL_CANDIDATE,
            horizon_months=120,
            cohort_count=4,
            median_ratio=median,
            worst_ratio=worst,
            best_ratio=2.0,
            win_rate=1.0,
            frictionless_median_ratio=median,
            bootstrap_p05_ratio=p05,
            crash_first_median_ratio=None,
            other_median_ratio=None,
            median_max_drawdown_after_tax=-0.05,
            median_taxes_paid_krw=0.0,
            median_sell_count=0.0,
            gate_passes=False,
        )

    assert after_tax_gate_passes(_summary(median=1.0, worst=1.0, p05=1.0), gate) is False
    assert after_tax_gate_passes(_summary(median=1.01, worst=0.97, p05=1.0), gate) is True


def test_deterministic_bootstrap() -> None:
    """Identical inputs and seed reproduce the identical bootstrap tail."""

    def _runner(config):
        wealth = 100.0 + (config.start.day % 7)
        return _make_result(wealth)

    first = run_after_tax_campaign(_spec(), _runner, seed=11)
    second = run_after_tax_campaign(_spec(), _runner, seed=11)
    assert [s.bootstrap_p05_ratio for s in first.summaries] == [
        s.bootstrap_p05_ratio for s in second.summaries
    ]


def test_spec_validation_rejects_bad_campaigns(tmp_path: Path) -> None:
    """Duplicate ids, missing primary horizon, and unknown regimes fail closed."""
    base = {
        "name": "bad",
        "start": "2000-04-01",
        "end": "2022-04-30",
        "contribution_krw": 1000000,
        "commission_bps": 10,
        "fx_spread_bps": 20,
        "tax_regime_path": "configs/tax/kr_overseas_equity.json",
        "horizons_months": [120],
        "primary_horizon_months": 120,
        "step_months": 12,
        "crash_regimes": ["dot_com"],
        "crash_window_months": 36,
        "fractional_shares": False,
        "bootstrap_paths": 10,
        "baseline_arm_id": "b1",
        "gate": {"median_ratio_floor": 1.0, "worst_ratio_floor": 0.97, "bootstrap_p05_floor": 1.0},
        "arms": [
            {
                "arm_id": "b1",
                "role": "baseline",
                "mode": "buy_only",
                "rebalance_band": None,
                "harvest_gains": False,
                "rule": _rule_static(),
            },
            {
                "arm_id": "c1",
                "role": "operational_candidate",
                "mode": "buy_only",
                "rebalance_band": None,
                "harvest_gains": True,
                "rule": _rule_static(),
            },
        ],
    }

    def _write(payload: dict) -> str:
        path = tmp_path / f"bad_{len(list(tmp_path.iterdir()))}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return str(path)

    dup = json.loads(json.dumps(base))
    dup["arms"].append(dict(dup["arms"][0]))
    with pytest.raises(ValueError, match="duplicate arm id"):
        load_after_tax_campaign_spec(_write(dup))
    missing_primary = json.loads(json.dumps(base))
    missing_primary["primary_horizon_months"] = 240
    with pytest.raises(ValueError, match="absent from"):
        load_after_tax_campaign_spec(_write(missing_primary))
    unknown_regime = json.loads(json.dumps(base))
    unknown_regime["crash_regimes"] = ["nope"]
    with pytest.raises(ValueError, match="unknown crash regime"):
        load_after_tax_campaign_spec(_write(unknown_regime))


def test_unlock_is_always_false() -> None:
    """Campaign reports never unlock the operational policy."""

    def _runner(config):
        return _make_result(120.0 if config.harvest_gains else 100.0)

    report = run_after_tax_campaign(_spec(), _runner, seed=7)
    assert report.operational_unlock is False


def test_report_written_under_experiment_result_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """JSON and markdown reports land under the experiment result directory."""
    monkeypatch.chdir(Path.cwd())
    settings = DataSettings(data_root=tmp_path / "data")

    def _runner(config):
        return _make_result(100.0)

    report = run_after_tax_campaign(_spec(), _runner, seed=7)
    out = write_after_tax_campaign_report(report, settings, experiment_id="abc123")
    assert out.is_file()
    assert out.parent == tmp_path / "data" / "results" / report.name
    assert out.name.startswith("after_tax_")
    assert out.with_suffix(".md").is_file()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert len(payload["summaries"]) == len(report.summaries)


def test_operational_candidate_can_pass_gate_on_primary() -> None:
    """A candidate beating the baseline on every cohort passes the primary gate."""

    def _runner(config):
        return _make_result(130.0 if config.harvest_gains else 100.0)

    report = run_after_tax_campaign(_spec(), _runner, seed=7)
    cand = next(s for s in report.summaries if s.arm_id == "c1" and s.horizon_months == 120)
    assert cand.median_ratio > 1.0
    assert cand.gate_passes is True


def test_campaign_mode_band_validation(tmp_path: Path) -> None:
    """REBALANCE_BAND without a band and BUY_ONLY with a band are rejected."""
    import copy

    base = {
        "name": "bad",
        "start": "2000-04-01",
        "end": "2022-04-30",
        "contribution_krw": 1000000,
        "commission_bps": 10,
        "fx_spread_bps": 20,
        "tax_regime_path": "configs/tax/kr_overseas_equity.json",
        "horizons_months": [120],
        "primary_horizon_months": 120,
        "step_months": 12,
        "crash_regimes": ["dot_com"],
        "crash_window_months": 36,
        "fractional_shares": False,
        "bootstrap_paths": 10,
        "baseline_arm_id": "b1",
        "gate": {"median_ratio_floor": 1.0, "worst_ratio_floor": 0.97, "bootstrap_p05_floor": 1.0},
        "arms": [
            {
                "arm_id": "b1",
                "role": "baseline",
                "mode": "buy_only",
                "rebalance_band": None,
                "harvest_gains": False,
                "rule": _rule_static(),
            },
            {
                "arm_id": "c1",
                "role": "operational_candidate",
                "mode": "rebalance_band",
                "rebalance_band": None,
                "harvest_gains": True,
                "rule": _rule_static(),
            },
        ],
    }
    path = tmp_path / "band_missing.json"
    path.write_text(json.dumps(base), encoding="utf-8")
    with pytest.raises(ValueError, match="without a band"):
        load_after_tax_campaign_spec(str(path))
    flawed = copy.deepcopy(base)
    flawed["arms"][1] = dict(flawed["arms"][1], rebalance_band=0.05, mode="buy_only")
    path2 = tmp_path / "band_misplaced.json"
    path2.write_text(json.dumps(flawed), encoding="utf-8")
    with pytest.raises(ValueError, match="with a band"):
        load_after_tax_campaign_spec(str(path2))
