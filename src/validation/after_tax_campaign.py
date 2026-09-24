"""After-tax cohort campaign with frictionless decomposition and crash-first disclosure."""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

from src.policy.after_tax_rules import build_weight_rule, parse_after_tax_rule_spec
from src.sim.tax import load_tax_regime
from src.validation.bootstrap import moving_block_bootstrap
from src.validation.gate import wealth_quantile
from src.validation.historical_campaign import REGIME_COVERAGE_CATALOG, RegimeWindow
from src.validation.windows import add_calendar_months, rolling_cohorts

if TYPE_CHECKING:
    from src.data.settings import DataSettings
    from src.policy.after_tax_rules import AfterTaxRuleSpec
    from src.sim.after_tax_engine import AfterTaxConfig, AfterTaxResult, ExecutionMode

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


class ArmRole(StrEnum):
    """Campaign role; only the operational candidate can pass the adoption gate."""

    BASELINE = "baseline"
    OPERATIONAL_CANDIDATE = "operational_candidate"
    PROSPECTIVE_WATCH = "prospective_watch"
    DISCLOSURE = "disclosure"


@dataclass(frozen=True, slots=True)
class AfterTaxArmSpec:
    arm_id: str
    role: ArmRole
    rule: AfterTaxRuleSpec
    mode: ExecutionMode
    rebalance_band: float | None
    harvest_gains: bool


@dataclass(frozen=True, slots=True)
class AfterTaxGateSpec:
    """Adoption thresholds on the primary horizon; informational only."""

    median_ratio_floor: float
    worst_ratio_floor: float
    bootstrap_p05_floor: float


@dataclass(frozen=True, slots=True)
class AfterTaxCampaignSpec:
    name: str
    start: date
    end: date
    contribution_krw: float
    commission_bps: float
    fx_spread_bps: float
    tax_regime_path: str
    horizons_months: tuple[int, ...]
    primary_horizon_months: int
    step_months: int
    crash_regimes: tuple[str, ...]
    crash_window_months: int
    fractional_shares: bool
    bootstrap_paths: int
    baseline_arm_id: str
    arms: tuple[AfterTaxArmSpec, ...]
    gate: AfterTaxGateSpec


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


def _parse_date(value: object, name: str) -> date:
    try:
        return date.fromisoformat(str(value))
    except ValueError as exc:
        raise ValueError(f"{name} must be an ISO date, got {value!r}") from exc


def _parse_positive_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a number, got {value!r}")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite positive, got {value!r}")
    return result


def _parse_nonnegative_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a number, got {value!r}")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite nonnegative, got {value!r}")
    return result


def load_after_tax_campaign_spec(path: str | Path) -> AfterTaxCampaignSpec:
    """Load and validate campaign JSON.

    Raises:
        ValueError: On duplicate arm ids, a missing or non-BASELINE baseline arm, more
            than one BASELINE arm, ``primary_horizon_months`` absent from
            ``horizons_months``, unknown crash regime names (must exist in
            ``REGIME_COVERAGE_CATALOG``), REBALANCE_BAND without a band, or invalid fields.
    """
    from src.sim.after_tax_engine import ExecutionMode

    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("campaign JSON must be an object")
    try:
        name = str(document["name"]).strip()
        if not name:
            raise ValueError("name must be non-blank")
        start = _parse_date(document["start"], "start")
        end = _parse_date(document["end"], "end")
        if start > end:
            raise ValueError(f"start {start.isoformat()} is after end {end.isoformat()}")
        contribution = _parse_positive_float(document["contribution_krw"], "contribution_krw")
        commission = _parse_nonnegative_float(document["commission_bps"], "commission_bps")
        spread = _parse_nonnegative_float(document["fx_spread_bps"], "fx_spread_bps")
        tax_regime_path = str(document["tax_regime_path"]).strip()
        if not tax_regime_path:
            raise ValueError("tax_regime_path must be non-blank")
        raw_horizons = document["horizons_months"]
        if not isinstance(raw_horizons, list) or not raw_horizons:
            raise ValueError("horizons_months must be a nonempty list")
        horizons: list[int] = []
        for item in raw_horizons:
            if isinstance(item, bool) or not isinstance(item, int) or item < 1:
                raise ValueError(f"horizons_months entries must be positive integers, got {item!r}")
            horizons.append(item)
        primary = document["primary_horizon_months"]
        if isinstance(primary, bool) or not isinstance(primary, int) or primary < 1:
            raise ValueError(f"primary_horizon_months must be a positive integer, got {primary!r}")
        if primary not in horizons:
            raise ValueError(f"primary_horizon_months {primary} absent from horizons_months {horizons!r}")
        step = document["step_months"]
        if isinstance(step, bool) or not isinstance(step, int) or step < 1:
            raise ValueError(f"step_months must be a positive integer, got {step!r}")
        raw_crash = document.get("crash_regimes", [])
        if not isinstance(raw_crash, list):
            raise ValueError("crash_regimes must be a list")
        crash_regimes = tuple(str(item) for item in raw_crash)
        known_regimes = {window.regime_name for window in REGIME_COVERAGE_CATALOG}
        for regime_name in crash_regimes:
            if regime_name not in known_regimes:
                raise ValueError(f"unknown crash regime {regime_name!r}")
        crash_window = document["crash_window_months"]
        if isinstance(crash_window, bool) or not isinstance(crash_window, int) or crash_window < 1:
            raise ValueError(f"crash_window_months must be a positive integer, got {crash_window!r}")
        fractional = document["fractional_shares"]
        if not isinstance(fractional, bool):
            raise ValueError(f"fractional_shares must be a boolean, got {fractional!r}")
        bootstrap_paths = document["bootstrap_paths"]
        if isinstance(bootstrap_paths, bool) or not isinstance(bootstrap_paths, int) or bootstrap_paths < 1:
            raise ValueError(f"bootstrap_paths must be integer >= 1, got {bootstrap_paths!r}")
        baseline_arm_id = str(document["baseline_arm_id"]).strip()
        if not baseline_arm_id:
            raise ValueError("baseline_arm_id must be non-blank")
        raw_gate = document["gate"]
        if not isinstance(raw_gate, dict):
            raise ValueError("gate must be an object")
        gate = AfterTaxGateSpec(
            median_ratio_floor=float(raw_gate["median_ratio_floor"]),
            worst_ratio_floor=float(raw_gate["worst_ratio_floor"]),
            bootstrap_p05_floor=float(raw_gate["bootstrap_p05_floor"]),
        )
        for field_name in ("median_ratio_floor", "worst_ratio_floor", "bootstrap_p05_floor"):
            if not math.isfinite(getattr(gate, field_name)):
                raise ValueError(f"gate.{field_name} must be finite")
        raw_arms = document["arms"]
        if not isinstance(raw_arms, list) or not raw_arms:
            raise ValueError("arms must be a nonempty list")
        arms: list[AfterTaxArmSpec] = []
        seen_ids: set[str] = set()
        for entry in raw_arms:
            if not isinstance(entry, dict):
                raise ValueError("arm entries must be objects")
            arm_id = str(entry["arm_id"]).strip()
            if not arm_id:
                raise ValueError("arm_id must be non-blank")
            if arm_id in seen_ids:
                raise ValueError(f"duplicate arm id {arm_id!r}")
            seen_ids.add(arm_id)
            try:
                role = ArmRole(str(entry["role"]))
            except ValueError as exc:
                raise ValueError(f"unknown arm role {entry.get('role')!r}") from exc
            try:
                mode = ExecutionMode(str(entry["mode"]))
            except ValueError as exc:
                raise ValueError(f"unknown execution mode {entry.get('mode')!r}") from exc
            band_raw = entry.get("rebalance_band")
            band: float | None = None
            if band_raw is not None:
                if isinstance(band_raw, bool) or not isinstance(band_raw, int | float):
                    raise ValueError(f"rebalance_band must be numeric, got {band_raw!r}")
                band = float(band_raw)
                if not math.isfinite(band) or not 0.0 <= band <= 1.0:
                    raise ValueError(f"rebalance_band must lie in [0, 1], got {band_raw!r}")
            if mode is ExecutionMode.REBALANCE_BAND and band is None:
                raise ValueError(f"arm {arm_id!r} uses REBALANCE_BAND without a band")
            if mode is ExecutionMode.BUY_ONLY and band is not None:
                raise ValueError(f"arm {arm_id!r} uses BUY_ONLY with a band")
            harvest = entry.get("harvest_gains")
            if not isinstance(harvest, bool):
                raise ValueError(f"arm {arm_id!r} harvest_gains must be a boolean")
            rule_payload = entry.get("rule")
            if not isinstance(rule_payload, dict):
                raise ValueError(f"arm {arm_id!r} rule must be an object")
            rule = parse_after_tax_rule_spec(rule_payload)
            arms.append(
                AfterTaxArmSpec(
                    arm_id=arm_id, role=role, rule=rule, mode=mode, rebalance_band=band, harvest_gains=harvest
                )
            )
    except KeyError as exc:
        raise ValueError(f"campaign JSON missing field {exc}") from exc
    baseline_arms = [arm for arm in arms if arm.arm_id == baseline_arm_id]
    if not baseline_arms:
        raise ValueError(f"baseline arm {baseline_arm_id!r} not found")
    if baseline_arms[0].role is not ArmRole.BASELINE:
        raise ValueError(f"baseline arm {baseline_arm_id!r} role must be baseline")
    if sum(1 for arm in arms if arm.role is ArmRole.BASELINE) != 1:
        raise ValueError("exactly one BASELINE arm is required")
    if not Path(tax_regime_path).is_file():
        raise ValueError(f"tax_regime_path not found: {tax_regime_path!r}")
    return AfterTaxCampaignSpec(
        name=name,
        start=start,
        end=end,
        contribution_krw=contribution,
        commission_bps=commission,
        fx_spread_bps=spread,
        tax_regime_path=tax_regime_path,
        horizons_months=tuple(horizons),
        primary_horizon_months=primary,
        step_months=step,
        crash_regimes=crash_regimes,
        crash_window_months=crash_window,
        fractional_shares=fractional,
        bootstrap_paths=bootstrap_paths,
        baseline_arm_id=baseline_arm_id,
        arms=tuple(arms),
        gate=gate,
    )


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


def write_after_tax_campaign_report(
    report: AfterTaxCampaignReport, settings: DataSettings, experiment_id: str
) -> Path:
    """Write ``{name}_after_tax_{experiment_id}.json`` plus a markdown summary beside it under the experiment's result directory."""
    from src.data.result_store import ResultKind, write_result

    payload = {
        "name": report.name,
        "experiment_id": experiment_id,
        "operational_unlock": bool(report.operational_unlock),
        "summaries": [
            {
                "arm_id": summary.arm_id,
                "role": str(summary.role),
                "horizon_months": summary.horizon_months,
                "cohort_count": summary.cohort_count,
                "median_ratio": summary.median_ratio,
                "worst_ratio": summary.worst_ratio,
                "best_ratio": summary.best_ratio,
                "win_rate": summary.win_rate,
                "frictionless_median_ratio": summary.frictionless_median_ratio,
                "bootstrap_p05_ratio": summary.bootstrap_p05_ratio,
                "crash_first_median_ratio": summary.crash_first_median_ratio,
                "other_median_ratio": summary.other_median_ratio,
                "median_max_drawdown_after_tax": summary.median_max_drawdown_after_tax,
                "median_taxes_paid_krw": summary.median_taxes_paid_krw,
                "median_sell_count": summary.median_sell_count,
                "gate_passes": summary.gate_passes,
            }
            for summary in report.summaries
        ],
        "rows": [
            {
                "arm_id": row.arm_id,
                "horizon_months": row.horizon_months,
                "cohort_start": row.cohort_start.isoformat(),
                "cohort_end": row.cohort_end.isoformat(),
                "after_tax_real_krw": row.after_tax_real_krw,
                "ratio": row.ratio,
                "frictionless_ratio": row.frictionless_ratio,
                "max_drawdown_after_tax": row.max_drawdown_after_tax,
                "taxes_paid_krw": row.taxes_paid_krw,
                "sell_count": row.sell_count,
                "crash_first": row.crash_first,
                "financial_income_breach_years": row.financial_income_breach_years,
            }
            for row in report.rows
        ],
    }
    lines = [
        f"# After-tax campaign {report.name}",
        "",
        f"experiment_id: {experiment_id}",
        f"operational_unlock: {report.operational_unlock}",
        "",
        "| arm | horizon | cohorts | median | worst | p05 | gate |",
        "|---|---|---|---|---|---|---|",
    ]
    lines.extend(
        f"| {summary.arm_id} | {summary.horizon_months} | {summary.cohort_count} "
        f"| {summary.median_ratio:.4f} | {summary.worst_ratio:.4f} "
        f"| {summary.bootstrap_p05_ratio:.4f} | {summary.gate_passes} |"
        for summary in report.summaries
    )
    ref = write_result(
        settings,
        experiment=report.name,
        kind=ResultKind.AFTER_TAX,
        run_id=experiment_id,
        payload=payload,
        markdown="\n".join(lines) + "\n",
    )
    return ref.json_path
