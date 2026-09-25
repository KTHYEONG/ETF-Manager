"""After-tax cohort campaign with frictionless decomposition and crash-first disclosure."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta
from typing import TYPE_CHECKING

from src.policy.after_tax_rules import build_weight_rule
from src.sim.tax import load_tax_regime
from src.validation.after_tax_campaign_config import (
    AfterTaxArmSpec,
    AfterTaxCampaignSpec,
    AfterTaxGateSpec,
    ArmRole,
    load_after_tax_campaign_spec,
)
from src.validation.after_tax_campaign_report import write_after_tax_campaign_report
from src.validation.bootstrap import moving_block_bootstrap
from src.validation.gate import wealth_quantile
from src.validation.historical_campaign import REGIME_COVERAGE_CATALOG, RegimeWindow
from src.validation.windows import add_calendar_months, rolling_cohorts

if TYPE_CHECKING:
    from src.sim.after_tax_engine import AfterTaxConfig, AfterTaxResult

logger = logging.getLogger(__name__)

__all__ = [
    "AfterTaxArmSpec",
    "AfterTaxArmSummary",
    "AfterTaxCampaignReport",
    "AfterTaxCampaignSpec",
    "AfterTaxCohortRow",
    "AfterTaxGateSpec",
    "ArmRole",
    "after_tax_gate_passes",
    "load_after_tax_campaign_spec",
    "run_after_tax_campaign",
    "write_after_tax_campaign_report",
]


@dataclass(frozen=True, slots=True)
class AfterTaxCohortRow:
    arm_id: str
    horizon_months: int
    cohort_start: date
    cohort_end: date
    after_tax_real_krw: float
    ratio: float
    frictionless_ratio: float
    max_drawdown_after_tax: float
    taxes_paid_krw: float
    sell_count: int
    crash_first: bool
    financial_income_breach_years: int


@dataclass(frozen=True, slots=True)
class AfterTaxArmSummary:
    arm_id: str
    role: ArmRole
    horizon_months: int
    cohort_count: int
    median_ratio: float
    worst_ratio: float
    best_ratio: float
    win_rate: float
    frictionless_median_ratio: float
    bootstrap_p05_ratio: float
    crash_first_median_ratio: float | None
    other_median_ratio: float | None
    median_max_drawdown_after_tax: float
    median_taxes_paid_krw: float
    median_sell_count: float
    gate_passes: bool


@dataclass(frozen=True, slots=True)
class AfterTaxCampaignReport:
    name: str
    rows: tuple[AfterTaxCohortRow, ...]
    summaries: tuple[AfterTaxArmSummary, ...]
    operational_unlock: bool


def after_tax_gate_passes(summary: AfterTaxArmSummary, gate: AfterTaxGateSpec) -> bool:
    """Primary-horizon verdict: median > floor, worst ≥ floor, bootstrap p05 ≥ floor.

    Only OPERATIONAL_CANDIDATE arms can pass; other roles always return False so watch
    and disclosure arms can never be mistaken for adoptable policies.
    """
    if summary.role is not ArmRole.OPERATIONAL_CANDIDATE:
        return False
    return (
        summary.median_ratio > gate.median_ratio_floor
        and summary.worst_ratio >= gate.worst_ratio_floor
        and summary.bootstrap_p05_ratio >= gate.bootstrap_p05_floor
    )


def _crash_first(
    cohort_start: date,
    cohort_end: date,
    crash_window_months: int,
    regime_windows: tuple[RegimeWindow, ...],
) -> bool:
    first_end = add_calendar_months(cohort_start, crash_window_months) - timedelta(days=1)
    first_end = min(first_end, cohort_end)
    return any(not (first_end < window.start or cohort_start > window.end) for window in regime_windows)


def run_after_tax_campaign(
    spec: AfterTaxCampaignSpec,
    runner: Callable[[AfterTaxConfig], AfterTaxResult],
    *,
    seed: int,
) -> AfterTaxCampaignReport:
    """Run every arm and the baseline over rolling cohorts for each horizon.

    Each cohort is simulated twice per arm: with the configured frictions and tax, and
    frictionless (``tax_enabled=False``, zero commission and spread) so reports separate
    timing effects from tax/cost drag. Ratios are arm/baseline real after-tax wealth on
    the identical cohort. A cohort is ``crash_first`` when its first
    ``crash_window_months`` overlap any configured crash regime window. Bootstrap p05
    uses the seeded moving-block resampling of cohort ratios (block = n // 2).

    Raises:
        ValueError: When no cohort fits a horizon.
        PitMarketError / AfterTaxDataError: Propagated fail-closed from the runner.
    """
    from src.sim.after_tax_engine import AfterTaxConfig

    regime = load_tax_regime(spec.tax_regime_path)
    catalog_by_name = {window.regime_name: window for window in REGIME_COVERAGE_CATALOG}
    regime_windows = tuple(catalog_by_name[name] for name in spec.crash_regimes)
    baseline = next(arm for arm in spec.arms if arm.arm_id == spec.baseline_arm_id)
    rows: list[AfterTaxCohortRow] = []
    summaries: list[AfterTaxArmSummary] = []
    baseline_cache: dict[tuple[int, str, str, str], AfterTaxResult] = {}

    def _baseline_result(horizon_months: int, c_start: date, c_end: date, frictionless: bool) -> AfterTaxResult:
        key = (horizon_months, c_start.isoformat(), c_end.isoformat(), "free" if frictionless else "frict")
        cached = baseline_cache.get(key)
        if cached is not None:
            return cached
        rule = build_weight_rule(baseline.rule, horizon_end=c_end)
        config = AfterTaxConfig(
            start=c_start,
            end=c_end,
            monthly_contribution_krw=spec.contribution_krw,
            tax_regime=regime,
            rule=rule,
            mode=baseline.mode,
            rebalance_band=baseline.rebalance_band,
            harvest_gains=baseline.harvest_gains,
            tax_enabled=not frictionless,
            commission_bps=0.0 if frictionless else spec.commission_bps,
            fx_spread_bps=0.0 if frictionless else spec.fx_spread_bps,
            fractional_shares=spec.fractional_shares,
        )
        result = runner(config)
        baseline_cache[key] = result
        return result

    for horizon in spec.horizons_months:
        cohorts = rolling_cohorts(spec.start, spec.end, horizon_months=horizon, step_months=spec.step_months)
        if not cohorts:
            raise ValueError(f"no cohorts fit horizon {horizon}")

        for arm in spec.arms:
            arm_ratios: list[float] = []
            arm_free_ratios: list[float] = []
            arm_rows: list[AfterTaxCohortRow] = []
            for c_start, c_end in cohorts:
                rule = build_weight_rule(arm.rule, horizon_end=c_end)
                config = AfterTaxConfig(
                    start=c_start,
                    end=c_end,
                    monthly_contribution_krw=spec.contribution_krw,
                    tax_regime=regime,
                    rule=rule,
                    mode=arm.mode,
                    rebalance_band=arm.rebalance_band,
                    harvest_gains=arm.harvest_gains,
                    tax_enabled=True,
                    commission_bps=spec.commission_bps,
                    fx_spread_bps=spec.fx_spread_bps,
                    fractional_shares=spec.fractional_shares,
                )
                free_config = AfterTaxConfig(
                    start=c_start,
                    end=c_end,
                    monthly_contribution_krw=spec.contribution_krw,
                    tax_regime=regime,
                    rule=rule,
                    mode=arm.mode,
                    rebalance_band=arm.rebalance_band,
                    harvest_gains=arm.harvest_gains,
                    tax_enabled=False,
                    commission_bps=0.0,
                    fx_spread_bps=0.0,
                    fractional_shares=spec.fractional_shares,
                )
                result = runner(config)
                free_result = runner(free_config)
                base = _baseline_result(horizon, c_start, c_end, False)
                base_free = _baseline_result(horizon, c_start, c_end, True)
                base_wealth = float(base.terminal_after_tax_real_krw)
                base_free_wealth = float(base_free.terminal_after_tax_real_krw)
                if base_wealth <= 0.0 or base_free_wealth <= 0.0:
                    raise ValueError("baseline wealth must be positive")
                if arm.arm_id == spec.baseline_arm_id:
                    ratio = 1.0
                    free_ratio = 1.0
                else:
                    ratio = float(result.terminal_after_tax_real_krw) / base_wealth
                    free_ratio = float(free_result.terminal_after_tax_real_krw) / base_free_wealth
                arm_ratios.append(ratio)
                arm_free_ratios.append(free_ratio)
                arm_rows.append(
                    AfterTaxCohortRow(
                        arm_id=arm.arm_id,
                        horizon_months=horizon,
                        cohort_start=c_start,
                        cohort_end=c_end,
                        after_tax_real_krw=float(result.terminal_after_tax_real_krw),
                        ratio=ratio,
                        frictionless_ratio=free_ratio,
                        max_drawdown_after_tax=float(result.max_drawdown_after_tax),
                        taxes_paid_krw=float(result.taxes_paid_krw),
                        sell_count=int(result.sell_count),
                        crash_first=_crash_first(c_start, c_end, spec.crash_window_months, regime_windows),
                        financial_income_breach_years=len(result.financial_income_breach_years),
                    )
                )
            rows.extend(arm_rows)
            block = max(1, len(arm_ratios) // 2)
            paths = moving_block_bootstrap(arm_ratios, block_size=block, n_paths=spec.bootstrap_paths, seed=seed)
            path_means = tuple(sum(p) / len(p) for p in paths)
            bootstrap_p05 = wealth_quantile(path_means, 0.05)
            crash_ratios = [row.ratio for row in arm_rows if row.crash_first]
            other_ratios = [row.ratio for row in arm_rows if not row.crash_first]
            median_mdd = wealth_quantile([row.max_drawdown_after_tax for row in arm_rows], 0.5)
            median_taxes = wealth_quantile([row.taxes_paid_krw for row in arm_rows], 0.5)
            median_sells = wealth_quantile([float(row.sell_count) for row in arm_rows], 0.5)
            provisional = AfterTaxArmSummary(
                arm_id=arm.arm_id,
                role=arm.role,
                horizon_months=horizon,
                cohort_count=len(arm_rows),
                median_ratio=wealth_quantile(arm_ratios, 0.5),
                worst_ratio=min(arm_ratios),
                best_ratio=max(arm_ratios),
                win_rate=sum(1 for value in arm_ratios if value > 1.0) / len(arm_ratios),
                frictionless_median_ratio=wealth_quantile(arm_free_ratios, 0.5),
                bootstrap_p05_ratio=bootstrap_p05,
                crash_first_median_ratio=wealth_quantile(crash_ratios, 0.5) if crash_ratios else None,
                other_median_ratio=wealth_quantile(other_ratios, 0.5) if other_ratios else None,
                median_max_drawdown_after_tax=median_mdd,
                median_taxes_paid_krw=median_taxes,
                median_sell_count=median_sells,
                gate_passes=False,
            )
            gate_ok = horizon == spec.primary_horizon_months and after_tax_gate_passes(provisional, spec.gate)
            summaries.append(
                AfterTaxArmSummary(
                    arm_id=provisional.arm_id,
                    role=provisional.role,
                    horizon_months=provisional.horizon_months,
                    cohort_count=provisional.cohort_count,
                    median_ratio=provisional.median_ratio,
                    worst_ratio=provisional.worst_ratio,
                    best_ratio=provisional.best_ratio,
                    win_rate=provisional.win_rate,
                    frictionless_median_ratio=provisional.frictionless_median_ratio,
                    bootstrap_p05_ratio=provisional.bootstrap_p05_ratio,
                    crash_first_median_ratio=provisional.crash_first_median_ratio,
                    other_median_ratio=provisional.other_median_ratio,
                    median_max_drawdown_after_tax=provisional.median_max_drawdown_after_tax,
                    median_taxes_paid_krw=provisional.median_taxes_paid_krw,
                    median_sell_count=provisional.median_sell_count,
                    gate_passes=gate_ok,
                )
            )
            logger.info(
                "[ALGO] event=after_tax_arm_done arm=%s horizon=%d cohorts=%d median_ratio=%.6f",
                arm.arm_id,
                horizon,
                len(arm_rows),
                provisional.median_ratio,
            )
    return AfterTaxCampaignReport(
        name=spec.name, rows=tuple(rows), summaries=tuple(summaries), operational_unlock=False
    )
